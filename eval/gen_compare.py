"""定性檢查：載入訓練好的 prefix，對同一組提示「prefix 開 vs 關」並排比較生成結果。

目的：在投入正式 benchmark 前，先用肉眼確認 prefix 真的把程式碼往「安全寫法」推。
零額外安裝，server 上直接可跑。

  prefix ON  = 套用訓練好的 adapter（你的安全 prefix）
  prefix OFF = model.disable_adapter()（等同原始 base Phi-3，即 DPO 的 reference）

用法：
  python eval/gen_compare.py \
      --model microsoft/Phi-3-mini-4k-instruct \
      --adapter ./outputs/phi3-prefix-dpo-full \
      --prompts eval/prompts_security.jsonl
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_prompts(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


@torch.no_grad()
def generate(model, tokenizer, instruction, device, max_new_tokens, use_chat_template):
    if use_chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False, add_generation_prompt=True,
        )
    else:
        text = instruction + "\n"
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    out = model.generate(
        ids, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--adapter", required=True, help="訓練好的 prefix adapter 目錄")
    ap.add_argument("--prompts", default="eval/prompts_security.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--no_chat_template", action="store_true")
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

    prompts = load_prompts(args.prompts)
    use_ct = not args.no_chat_template

    for i, ex in enumerate(prompts, 1):
        instr = ex["prompt"]
        cwe = ex.get("cwe", "")
        # prefix ON（套用 adapter）
        on = generate(model, tokenizer, instr, device, args.max_new_tokens, use_ct)
        # prefix OFF（停用 adapter = base）
        with model.disable_adapter():
            off = generate(model, tokenizer, instr, device, args.max_new_tokens, use_ct)

        print("\n" + "=" * 80)
        print(f"[{i}] {cwe}  |  {instr}")
        print("-" * 80)
        print(">>> prefix OFF（base Phi-3）:\n" + off)
        print("-" * 80)
        print(">>> prefix ON（安全 prefix）:\n" + on)


if __name__ == "__main__":
    main()
