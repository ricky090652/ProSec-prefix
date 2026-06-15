"""PoC 煙霧測試：把這條路【唯一的真風險】先驗掉，不依賴 TRL。

要驗證三件事（這三件成立，TRL DPOTrainer 才能正常運作）：
  1. CodeLlama-7B 能被 PEFT PrefixTuning 包起來，且只有 prefix 參數可訓練。
  2. 包了 prefix 之後 forward 拿得到 logits、generate 跑得動。
  3. 「啟用 adapter（policy）」vs「disable_adapter（reference=base）」會給出
     【不同】的 log-prob —— 這正是 DPO 的 policy/ref 機制，等同舊 SVEN 裡
     「有 sec prefix」vs「無 prefix 參考前向」。

跑法（先用小模型驗程式碼路徑會更快，再換 CodeLlama-7B）：
  python poc_prefix_smoke.py --model codellama/CodeLlama-7b-Instruct-hf
  # 快速驗管線（不下載 7B）：
  python poc_prefix_smoke.py --model sshleifer/tiny-gpt2 --no_chat_template
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PrefixTuningConfig, get_peft_model, TaskType


def seq_logprob(model, tokenizer, prompt, completion, device):
    """計算 log P(completion | prompt) 在 completion token 上的總和。"""
    full = prompt + completion
    full_ids = tokenizer(full, return_tensors="pt").input_ids.to(device)
    prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]

    with torch.no_grad():
        logits = model(full_ids).logits  # (1, T, V)
    logits = logits[:, :-1, :]
    labels = full_ids[:, 1:]
    logps = torch.log_softmax(logits.float(), dim=-1)
    token_logps = torch.gather(logps, 2, labels.unsqueeze(-1)).squeeze(-1)  # (1, T-1)
    # 只取 completion 段（labels index >= prompt_len-1）
    comp_logps = token_logps[:, prompt_len - 1:]
    return comp_logps.sum().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="codellama/CodeLlama-7b-Instruct-hf")
    ap.add_argument("--num_virtual_tokens", type=int, default=16)
    ap.add_argument("--no_chat_template", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[1] 載入 base model: {args.model} (device={device}, dtype={dtype})")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype,
        device_map={"": 0} if device == "cuda" else None,
    )
    if device != "cuda":
        base = base.to(device)

    print(f"[2] 套上 PEFT PrefixTuning (num_virtual_tokens={args.num_virtual_tokens})")
    peft_cfg = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=args.num_virtual_tokens,
    )
    model = get_peft_model(base, peft_cfg)
    model.print_trainable_parameters()  # 應只有 prefix 參數可訓練

    # 構造一個提示
    instr = "Write a Python function that lists files in a user-provided directory."
    if args.no_chat_template:
        prompt = instr + "\n"
    else:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": instr}],
            tokenize=False, add_generation_prompt=True,
        )
    safe = "import subprocess\ndef f(p):\n    return subprocess.run(['ls','-la',p], capture_output=True, text=True).stdout\n"
    vuln = "import os\ndef f(p):\n    return os.popen('ls -la ' + p).read()\n"

    print("[3] forward + generate 測試")
    model.eval()
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        _ = model(ids).logits
    gen = model.generate(ids, max_new_tokens=32, do_sample=False)
    print("    generate 輸出片段：",
          tokenizer.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)[:120].replace("\n", "\\n"))

    print("[4] policy(啟用 prefix) vs reference(disable_adapter=base) 的 log-prob")
    # policy
    pol_safe = seq_logprob(model, tokenizer, prompt, safe, device)
    pol_vuln = seq_logprob(model, tokenizer, prompt, vuln, device)
    # reference = base（停用 prefix）
    with model.disable_adapter():
        ref_safe = seq_logprob(model, tokenizer, prompt, safe, device)
        ref_vuln = seq_logprob(model, tokenizer, prompt, vuln, device)

    print(f"    policy   : logp(safe)={pol_safe:.3f}  logp(vuln)={pol_vuln:.3f}")
    print(f"    reference: logp(safe)={ref_safe:.3f}  logp(vuln)={ref_vuln:.3f}")
    dpo_margin = (pol_safe - ref_safe) - (pol_vuln - ref_vuln)
    print(f"    DPO reward margin (未訓練前，應接近 0):  {dpo_margin:.4f}")

    # 健全性檢查：未訓練的 prefix 初始化下，policy 與 reference 不該完全相等
    same = abs(pol_safe - ref_safe) < 1e-6 and abs(pol_vuln - ref_vuln) < 1e-6
    print("\n=== 結論 ===")
    if same:
        print("⚠️  policy 與 reference 完全相等 → prefix 沒有實際作用，需檢查 PEFT 版本/設定。")
    else:
        print("✅ prefix 生效、forward/generate 正常、policy≠reference。")
        print("   → PEFT PrefixTuning + 「disable_adapter 當 reference」機制在此模型上可用，")
        print("     可以放心接 TRL DPOTrainer（見 train_prefix.py）。")


if __name__ == "__main__":
    main()
