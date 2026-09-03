"""功能性評測：對 HumanEval 算 pass@1，比較 prefix ON vs OFF。

確認套上安全 prefix 後，模型「把程式寫對」的能力有沒有退步。

流程（與安全評測對稱）：
  生成解答 → 在子行程執行 HumanEval 測試 → 算通過率
一次載入模型，同時跑 ON 與 OFF。

用法：
  python eval/run_humaneval.py \
      --model microsoft/Phi-3-mini-4k-instruct \
      --adapter ./outputs/phi3-prefix-dpo-full \
      --limit 20            # 先小規模試；正式跑拿掉

注意：會在子行程執行模型生成的程式碼（HumanEval 慣例），請在可信環境跑。
"""
import argparse
import json
import re
import subprocess
import tempfile
import os

import torch
# peft 0.19.1 會直接讀 torch.distributed.tensor.DTensor，但有些 torch build 不會
# 自動載入這個 submodule，於是 LoRA 掛到 nn.Linear 上時噴 AttributeError。
# 手動 import 一次把它掛上去（PrefixTuning 走不到這條路徑，所以之前沒踩到）。
try:
    import torch.distributed.tensor  # noqa: F401
except Exception:
    pass
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel




def adapter_kind(adapter_path):
    """從 adapter_config.json 讀出實際的 PEFT 型別，用於輸出標籤。

    腳本一開始只支援 prefix，標籤就寫死成 "prefix"；改成也能載 LoRA 之後
    這個標籤會誤導（兩臂的輸出看起來一模一樣，分不出跑的是哪個）。
    """
    import json as _json, os as _os
    for name in ("adapter_config.json",):
        path = _os.path.join(adapter_path, name)
        if _os.path.exists(path):
            try:
                t = _json.load(open(path)).get("peft_type", "")
                return {"LORA": "LoRA", "PREFIX_TUNING": "prefix"}.get(t, t.lower() or "adapter")
            except Exception:
                break
    return "adapter"

def build_messages(system_prompt, instruction):
    """訓練時帶了 system prompt，評測就必須帶同一句，否則 adapter 落在分布外。"""
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": instruction})
    return msgs

def extract_code(text):
    """從模型輸出抽出程式碼：優先取 ```python ...``` 區塊，否則取全文。"""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


def run_one(code, test, entry_point, timeout=10):
    """組成完整程式並在子行程執行，回傳是否通過。"""
    program = code + "\n\n" + test + f"\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        r = subprocess.run(["python", path], capture_output=True, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="./outputs/humaneval_result.json")
    ap.add_argument("--system_prompt", default=None,
                    help="套進 chat template 的 system 訊息。**必須與訓練時一致**：用 --system_prompt 訓練出來的 adapter，評測時不給就會 OOD。ProSec 論文管線用的是 \"You are helpful coding assistant.\"；早期的 prefix 實驗訓練時沒有 system prompt，那些要維持不給")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kind = adapter_kind(args.adapter)
    print(f"載入 base：{args.model} + {kind}：{args.adapter}")
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map={"": 0} if device == "cuda" else None,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    ds = load_dataset("openai_humaneval", split="test")
    if args.limit:
        ds = ds.select(range(args.limit))
    print(f"HumanEval 題數：{len(ds)}")

    @torch.no_grad()
    def gen(prompt_code):
        instr = ("Complete the following Python function. "
                 "Return the complete function in a single ```python code block.\n\n"
                 f"```python\n{prompt_code}\n```")
        text = tokenizer.apply_chat_template(
            build_messages(args.system_prompt, instr),
            tokenize=False, add_generation_prompt=True,
        )
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        out = model.generate(ids, do_sample=False, max_new_tokens=args.max_new_tokens,
                             pad_token_id=tokenizer.pad_token_id)
        new = out[0][ids.shape[1]:]
        # 撞到 max_new_tokens 的話函式會被切在半途，執行必定失敗 → pass@1 被系統性低估。
        # 這裡沒有事後補救的餘地（測試就是過或不過），所以一定要知道有沒有發生。
        truncated = bool((new != tokenizer.eos_token_id).all().item()) and \
            len(new) >= args.max_new_tokens
        return tokenizer.decode(new, skip_special_tokens=True), truncated

    def evaluate(use_prefix):
        passed = 0
        n_trunc = 0
        for ex in ds:
            if use_prefix:
                raw, trunc = gen(ex["prompt"])
            else:
                with model.disable_adapter():
                    raw, trunc = gen(ex["prompt"])
            n_trunc += int(trunc)
            code = extract_code(raw)
            if run_one(code, ex["test"], ex["entry_point"]):
                passed += 1
        return passed, n_trunc

    print(f"評測 {kind} ON ...")
    on_pass, on_trunc = evaluate(True)
    print(f"評測 {kind} OFF(base) ...")
    off_pass, off_trunc = evaluate(False)

    n = len(ds)
    result = {
        "n": n,
        "off_pass@1": round(100.0 * off_pass / n, 2),
        "on_pass@1": round(100.0 * on_pass / n, 2),
        "delta": round(100.0 * (on_pass - off_pass) / n, 2),
        "max_new_tokens": args.max_new_tokens,
        "off_truncated": off_trunc,
        "on_truncated": on_trunc,
    }
    json.dump(result, open(args.out, "w"), indent=2)
    print("\n" + "=" * 40)
    print(f"HumanEval pass@1（n={n}）")
    print(f"  OFF(base) : {result['off_pass@1']}%")
    print(f"  ON({kind}){' ' * max(0, 6 - len(kind))}: {result['on_pass@1']}%")
    print(f"  Δ         : {result['delta']:+}%  (≥0 代表功能性未退步)")
    # 撞到上限的生成一定執行失敗，pass@1 會被系統性低估。ON 若寫得比 OFF 長，
    # 低估的程度也會不對稱，Δ 因此失真。
    print(f"  撞 max_new_tokens={args.max_new_tokens}："
          f"OFF {off_trunc}/{n}、ON {on_trunc}/{n}")
    if max(on_trunc, off_trunc) > 0.02 * n:
        print(f"  ⚠️  截斷率超過 2%，pass@1 被低估，Δ 也失真。"
              f"請調高 --max_new_tokens 重跑")


if __name__ == "__main__":
    main()
