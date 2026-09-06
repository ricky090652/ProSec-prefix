"""對 PurpleLlama instruct benchmark 生成程式碼，輸出 ProSec detect_all.py 吃的 jsonl。

一次載入模型，同時產生 prefix ON 與 OFF 兩份(OFF = disable_adapter = base Phi-3)，
方便直接對照。每題抽 num_gen 個樣本(PurpleLlama 的做法：算所有樣本中安全碼比例)。

輸出兩個檔：<out_prefix>.on.jsonl / <out_prefix>.off.jsonl
每行：{"lang", "cwe", "prompt", "responses": [...]}  ← detect_all.py 需要的欄位

用法：
  python eval/gen_for_icd.py \
      --model microsoft/Phi-3-mini-4k-instruct \
      --adapter ./outputs/phi3-prefix-dpo-full \
      --instruct_json <PurpleLlama>/CybersecurityBenchmarks/datasets/instruct/instruct.json \
      --lang python --num_gen 10 --out_prefix ./outputs/icd_phi3
"""
import argparse
import json
import os
import re

import torch
# peft 0.19.1 會直接讀 torch.distributed.tensor.DTensor，但有些 torch build 不會
# 自動載入這個 submodule，於是 LoRA 掛到 nn.Linear 上時噴 AttributeError。
# 手動 import 一次把它掛上去（PrefixTuning 走不到這條路徑，所以之前沒踩到）。
try:
    import torch.distributed.tensor  # noqa: F401
except Exception:
    pass
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# SafeCoder 涵蓋的 CWE（掃 SafeCoder sec_eval 場景得到，共 35 個）。
# 搭配 --langs c,cpp,java,javascript,python + --safecoder_only，
# 可重建論文「PurpleLlama ∩ SafeCoder」子集（≈ 36 對 / 693 題，論文為 38 對 / 694 題）。
SAFECODER_CWES = {
    20, 22, 78, 79, 89, 94, 116, 117, 119, 125, 190, 200, 209, 215, 295,
    312, 326, 327, 338, 352, 377, 416, 476, 502, 611, 643, 676, 681, 732,
    777, 787, 798, 915, 916, 918,
}



def build_messages(system_prompt, instruction):
    """訓練時帶了 system prompt，評測就必須帶同一句，否則 adapter 落在分布外。"""
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": instruction})
    return msgs

def cwe_num(s):
    m = re.findall(r"\d+", str(s))
    return int(m[0]) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--instruct_json", required=True,
                    help="PurpleLlama datasets/instruct/instruct.json")
    ap.add_argument("--langs", default="c,cpp,java,javascript,python",
                    help="逗號分隔；對齊論文的 5 語言。可填 instruct.json 的 language 值："
                         "c,cpp,csharp,java,javascript,php,rust,python")
    ap.add_argument("--num_gen", type=int, default=10, help="每題抽幾個樣本")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 題(快速試)")
    ap.add_argument("--safecoder_only", action="store_true",
                    help="只保留 SafeCoder 涵蓋的 CWE；配 --langs 的 5 語言可重建論文子集")
    ap.add_argument("--seed", type=int, default=42,
                    help="取樣種子。不設(--seed -1)則每次結果不同——"
                         "base(OFF) 實測會漂移約 1 個百分點，小效果會被雜訊蓋掉")
    ap.add_argument("--out_prefix", default="./outputs/icd_phi3")
    ap.add_argument("--sides", choices=["both", "on", "off"], default="both",
                    help="要生成哪一側。OFF 是關掉 adapter 的 base model，**與 adapter 無關**，"
                         "所以多個臂共用同一份即可（實測四臂的 OFF 逐項相同）。"
                         "全量評測時先用 --sides off 生一次，各臂再用 --sides on，"
                         "生成量直接減半。逐題 seed 只取決於 --seed 與題目順序，"
                         "所以分開生成與一起生成的配對關係完全相同")
    ap.add_argument("--system_prompt", default=None,
                    help="套進 chat template 的 system 訊息。**必須與訓練時一致**：用 --system_prompt 訓練出來的 adapter，評測時不給就會 OOD。ProSec 論文管線用的是 \"You are helpful coding assistant.\"；早期的 prefix 實驗訓練時沒有 system prompt，那些要維持不給")
    args = ap.parse_args()
    if args.seed is not None and args.seed < 0:
        args.seed = None

    # 新開的 tmux pane 不會繼承 export，$ICD 為空時路徑會變成 /CybersecurityBenchmarks/...
    # 直接噴 FileNotFoundError 看不出原因，所以先擋下來講清楚。
    if not os.path.exists(args.instruct_json):
        hint = ("（路徑以 / 開頭 —— 看起來 $ICD 是空的。新開的 shell 要重新 export，"
                "或寫進 ~/.bashrc）" if args.instruct_json.startswith("/Cyber") else "")
        raise SystemExit(f"找不到 instruct.json：{args.instruct_json}{hint}\n"
                         "  export ICD=~/<你的路徑>/ProSec/PurpleLlama")


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

    langset = set(s.strip() for s in args.langs.split(",") if s.strip())
    data = json.load(open(args.instruct_json))
    data = [x for x in data if x.get("language") in langset]
    if args.safecoder_only:
        data = [x for x in data if cwe_num(x.get("cwe_identifier")) in SAFECODER_CWES]
    if args.limit:
        data = data[:args.limit]
    from collections import Counter
    print(f"語言={sorted(langset)}；題數={len(data)}；分布={dict(Counter(x['language'] for x in data))}")
    print(f"取樣：temperature={args.temperature} top_p={args.top_p} num_gen={args.num_gen} "
          f"seed={args.seed if args.seed is not None else '未固定(每次不同)'}")

    @torch.no_grad()
    def gen(instruction, seed=None):
        # 每題（ON 與 OFF）各自從同一個 RNG 狀態出發，讓兩邊看到同一組隨機性。
        # 這比只在程式開頭設一次 seed 更有用：ON/OFF 的差值是配對比較，
        # 消掉共同的抽樣雜訊之後，Δ 的變異會明顯小於各自的變異。
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        text = tokenizer.apply_chat_template(
            build_messages(args.system_prompt, instruction),
            tokenize=False, add_generation_prompt=True,
        )
        # Phi-3 的 pad_token_id 與 eos_token_id 同為 32000，HF 無法從 input_ids
        # 推斷 attention_mask（分不出「填充」與「真的結束符」），會印警告並退回全 1。
        # 目前 batch=1、無 padding，全 1 剛好正確，但明確傳入才不會在改成批次生成時出錯。
        enc = tokenizer(text, return_tensors="pt").to(device)
        ids = enc.input_ids
        out = model.generate(
            **enc, do_sample=True, num_return_sequences=args.num_gen,
            temperature=args.temperature, top_p=args.top_p,
            max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id,
        )
        # 記錄有沒有撞到 max_new_tokens。這一格資訊是必要的：ON 若寫得比 OFF 長，
        # 就會有更高比例被切在半句，括號配不起來 → 退化守門把「被截斷」誤判成
        # 「寫壞了」。分開這兩件事才能解讀語法完整率。
        texts, truncated = [], []
        for o in out:
            new = o[ids.shape[1]:]
            texts.append(tokenizer.decode(new, skip_special_tokens=True))
            # 沒有生出 eos 就代表是被長度上限切掉的
            truncated.append(bool((new != tokenizer.eos_token_id).all().item())
                             and len(new) >= args.max_new_tokens)
        return texts, truncated

    want_on = args.sides in ("both", "on")
    want_off = args.sides in ("both", "off")
    f_on = open(f"{args.out_prefix}.on.jsonl", "w") if want_on else None
    f_off = open(f"{args.out_prefix}.off.jsonl", "w") if want_off else None
    try:
        for i, x in enumerate(data, 1):
            prompt = x["test_case_prompt"]
            cwe = x.get("cwe_identifier", "")
            # 同一題的 ON / OFF 用同一個 seed
            qseed = None if args.seed is None else args.seed * 100003 + i
            # prefix ON
            pairs = []
            if want_on:
                pairs.append((f_on, *gen(prompt, qseed)))
            if want_off:
                # OFF = 關掉 adapter 的 base model
                with model.disable_adapter():
                    pairs.append((f_off, *gen(prompt, qseed)))
            for f, resp, trunc in pairs:
                f.write(json.dumps({
                    "lang": x["language"], "cwe": cwe, "prompt": prompt, "responses": resp,
                    "truncated": trunc, "max_new_tokens": args.max_new_tokens,
                }, ensure_ascii=False) + "\n")
                f.flush()
            if i % 10 == 0:
                print(f"  {i}/{len(data)}")
    finally:
        for f in (f_on, f_off):
            if f is not None:
                f.close()
    made = " / ".join(f"{args.out_prefix}.{s}.jsonl"
                      for s in (("on",) if want_on else ()) + (("off",) if want_off else ()))
    print(f"完成 → {made}")


if __name__ == "__main__":
    main()
