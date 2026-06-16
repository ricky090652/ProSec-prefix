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
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


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
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"載入 base：{args.model} + prefix：{args.adapter}")
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
            [{"role": "user", "content": instr}],
            tokenize=False, add_generation_prompt=True,
        )
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        out = model.generate(ids, do_sample=False, max_new_tokens=args.max_new_tokens,
                             pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    def evaluate(use_prefix):
        passed = 0
        for ex in ds:
            if use_prefix:
                raw = gen(ex["prompt"])
            else:
                with model.disable_adapter():
                    raw = gen(ex["prompt"])
            code = extract_code(raw)
            if run_one(code, ex["test"], ex["entry_point"]):
                passed += 1
        return passed

    print("評測 prefix ON ...")
    on_pass = evaluate(True)
    print("評測 prefix OFF(base) ...")
    off_pass = evaluate(False)

    n = len(ds)
    result = {
        "n": n,
        "off_pass@1": round(100.0 * off_pass / n, 2),
        "on_pass@1": round(100.0 * on_pass / n, 2),
        "delta": round(100.0 * (on_pass - off_pass) / n, 2),
    }
    json.dump(result, open(args.out, "w"), indent=2)
    print("\n" + "=" * 40)
    print(f"HumanEval pass@1（n={n}）")
    print(f"  OFF(base) : {result['off_pass@1']}%")
    print(f"  ON(prefix): {result['on_pass@1']}%")
    print(f"  Δ         : {result['delta']:+}%  (≥0 代表功能性未退步)")


if __name__ == "__main__":
    main()
