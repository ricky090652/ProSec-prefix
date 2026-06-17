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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


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
    ap.add_argument("--out_prefix", default="./outputs/icd_phi3")
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

    langset = set(s.strip() for s in args.langs.split(",") if s.strip())
    data = json.load(open(args.instruct_json))
    data = [x for x in data if x.get("language") in langset]
    if args.limit:
        data = data[:args.limit]
    from collections import Counter
    print(f"語言={sorted(langset)}；題數={len(data)}；分布={dict(Counter(x['language'] for x in data))}")

    @torch.no_grad()
    def gen(instruction):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False, add_generation_prompt=True,
        )
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        out = model.generate(
            ids, do_sample=True, num_return_sequences=args.num_gen,
            temperature=args.temperature, top_p=args.top_p,
            max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id,
        )
        return [tokenizer.decode(o[ids.shape[1]:], skip_special_tokens=True) for o in out]

    f_on = open(f"{args.out_prefix}.on.jsonl", "w")
    f_off = open(f"{args.out_prefix}.off.jsonl", "w")
    try:
        for i, x in enumerate(data, 1):
            prompt = x["test_case_prompt"]
            cwe = x.get("cwe_identifier", "")
            # prefix ON
            resp_on = gen(prompt)
            # prefix OFF = base
            with model.disable_adapter():
                resp_off = gen(prompt)
            for f, resp in ((f_on, resp_on), (f_off, resp_off)):
                f.write(json.dumps({
                    "lang": x["language"], "cwe": cwe, "prompt": prompt, "responses": resp,
                }, ensure_ascii=False) + "\n")
                f.flush()
            if i % 10 == 0:
                print(f"  {i}/{len(data)}")
    finally:
        f_on.close()
        f_off.close()
    print(f"完成 → {args.out_prefix}.on.jsonl / .off.jsonl")


if __name__ == "__main__":
    main()
