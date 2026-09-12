"""量 adapter 對「表示空間」的破壞程度——PT-PEFT 的 effective rank 診斷。

為什麼需要（S-rank）：
  PT-PEFT（Kim et al. 2024, arXiv:2411.00029）主張 prefix-tuning 不動權重，
  所以能保住 pre-training 學到的表示空間，而 LoRA / full-FT 會讓表示矩陣塌成
  低秩、丟掉語意細節。他們在 BLIP-2 上量到：pre-training 68.2%、prefix 68.2%、
  LoRA 52.0%、full-FT 47.0%（論文 Table 1）。

  但我們的 S3 實測是**反過來**的：prefix nvt=16 的 utility 掉 16.91 pt、
  nvt=8 掉 7.22 pt；LoRA qkv 量不到損失（+1.03，在誤差內）、all-linear 掉 4.12 pt。
  prefix 在兩個軸上都輸。若 PT-PEFT 的機制在 code generation 成立，
  prefix 的 rank 應該保住；若 prefix 的 rank 也塌了，代表「不動權重」不等於
  「不破壞表示空間」——KV cache 的全域 attention 擾動壓過了權重保留的好處，
  PT-PEFT 這條線就不必投入。

  PT-PEFT 的 Appendix D.3 其實只證明了 rank 的**上界**不降
  （min(|X|+|P|, ·) ≥ min(|X|, ·)），上界不降不代表實際 rank 不降。這支腳本
  就是去測實際值。

判讀（停損點）：
  prefix ≈ base，LoRA 明顯低   → PT-PEFT 前提成立，進 Prefix→LoRA 四臂實驗
  prefix 也明顯低              → 前提不成立，放棄該線（但這是可寫的 negative result）

方法（PT-PEFT §2.3，公式 1–5）：
  對每個樣本取最後一層 hidden states F ∈ R^{L×d} → SVD 取奇異值 → 除以總和正規化
  → 跨樣本平均 → 累積和 → 取累積和達 --threshold(0.9) 的 remaining rank ratio。

  與論文的一處偏離：論文沒說明不同長度的樣本怎麼平均。奇異值向量長度 = min(L, d)，
  長度不同就無法逐元素平均，所以這裡把所有樣本**截到同一個 token 長度**
  （--seq_len，預設 128），比插值到共同 grid 更乾淨、也完全可重現。

用法：
  # 主線：一次比 base / prefix / LoRA
  python eval/effective_rank.py \\
      --adapter prefix8=outputs/dpo-arms/prefix_nvt8 \\
      --adapter prefix16=outputs/dpo-arms/prefix \\
      --adapter lora_qkv=outputs/dpo-arms/lora_qkv \\
      --adapter lora_all=outputs/dpo-arms/lora \\
      --bf16 --csv_dir outputs/rank

  # 若 adapter 訓練時帶了 system prompt，要用同一句才不會 OOD（見 run_multipl_e.py）
  python eval/effective_rank.py --adapter lora_qkv=outputs/dpo-arms/lora_qkv \\
      --chat \\
      --system_prompt "You are helpful coding assistant."

  # 換到安全分布上看（預設是 utility 分布）
  python eval/effective_rank.py --adapter prefix8=outputs/dpo-arms/prefix_nvt8 \\
      --data eval/prompts_security.jsonl
"""
import argparse
import json
import os

import torch
# peft 0.19.1 會直接讀 torch.distributed.tensor.DTensor，但有些 torch build 不會
# 自動載入這個 submodule，於是 LoRA 掛到 nn.Linear 上時噴 AttributeError
# （PrefixTuning 走不到這條路徑，所以只有帶 LoRA 的那幾臂會踩到）。
try:
    import torch.distributed.tensor  # noqa: F401
except Exception:
    pass
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_texts(path, n, tokenizer, use_chat, system_prompt):
    """回傳一組固定文字。所有模型都吃完全相同的 token 序列，SVD 才可比。"""
    if path is None:
        from datasets import load_dataset
        ds = load_dataset("openai_humaneval", split="test")
        # prompt + canonical_solution：測的是「模型表示正確 code 的能力」，
        # 直接對應我們的 utility 指標（pass@1 on HumanEval/MultiPL-E）。
        raw = [ex["prompt"] + ex["canonical_solution"] for ex in ds]
    else:
        raw = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    o = json.loads(line)
                    raw.append(o.get("text") or o.get("prompt") or o["instruction"])

    if use_chat:
        out = []
        for t in raw:
            msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
            msgs.append({"role": "user", "content": t})
            out.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
        raw = out
    return raw[:n] if n > 0 else raw


@torch.no_grad()
def singular_spectrum(model, tokenizer, texts, seq_len, device):
    """對每段文字取最後一層 hidden states 做 SVD，回傳 (平均正規化奇異值, 用了幾個樣本)。

    batch_size 固定 1：有 padding 的話 pad 位置的 hidden states 會污染 SVD，
    而樣本數只有幾百個，不值得為了速度去處理 mask。
    """
    acc, used = None, 0
    for t in texts:
        ids = tokenizer(t, return_tensors="pt").input_ids
        if ids.shape[1] < seq_len:
            continue                      # 太短的丟掉，維持所有樣本 M 相同
        ids = ids[:, :seq_len].to(device)
        out = model(input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    output_hidden_states=True)
        # prefix 走 past_key_values，不佔 input 位置，所以 hidden_states 長度
        # 仍是 seq_len，與 base / LoRA 可以逐元素對齊。
        f = out.hidden_states[-1].squeeze(0).float()      # [L, d]
        s = torch.linalg.svdvals(f)                        # 降冪，長度 min(L, d)
        s = s / s.sum()                                    # 公式 (3)
        acc = s if acc is None else acc + s
        used += 1
    if used == 0:
        raise SystemExit(f"沒有樣本長度 ≥ --seq_len {seq_len}，把它調小再跑一次")
    return (acc / used).cpu(), used                        # 公式 (4)


def effective_rank(s_avg, threshold):
    """累積和第一次達到 threshold 的 remaining rank ratio（論文 Table 1 的定義）。"""
    y = torch.cumsum(s_avg, dim=0)                         # 公式 (5)
    k = int(torch.searchsorted(y, torch.tensor(threshold)).item()) + 1
    return 100.0 * k / len(s_avg), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--adapter", action="append", default=[], metavar="LABEL=PATH",
                    help="可重複。base 對照由 disable_adapter() 自動產生，不必另外指定")
    ap.add_argument("--data", default=None,
                    help="jsonl（吃 text/prompt/instruction 欄）。預設用 HumanEval")
    ap.add_argument("--n", type=int, default=164, help="取幾個樣本（0=全部）")
    ap.add_argument("--seq_len", type=int, default=128,
                    help="所有樣本統一截到這個 token 長度，短於此的跳過")
    ap.add_argument("--threshold", type=float, default=0.9, help="PT-PEFT 用 0.9")
    ap.add_argument("--chat", action="store_true",
                    help="套 chat template。adapter 訓練時若帶 system prompt 就該開，"
                         "否則 adapter 落在分布外（見 run_multipl_e.py 的同名警告）")
    ap.add_argument("--system_prompt", default=None)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--csv_dir", default=None, help="輸出累積曲線，用來畫 PT-PEFT Figure 4")
    args = ap.parse_args()

    if not args.adapter:
        raise SystemExit("至少要給一個 --adapter LABEL=PATH")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    texts = load_texts(args.data, args.n, tokenizer, args.chat, args.system_prompt)
    print(f"device={device}  樣本池={len(texts)}  seq_len={args.seq_len}  "
          f"threshold={args.threshold}  chat={args.chat}")

    from peft import PeftModel
    results, curves = {}, {}

    for spec in args.adapter:
        if "=" not in spec:
            raise SystemExit(f"--adapter 要寫成 LABEL=PATH，收到：{spec}")
        label, path = spec.split("=", 1)
        print(f"\n載入 {args.model} + {label}：{path}")
        base = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
            device_map={"": 0} if device == "cuda" else None)
        model = PeftModel.from_pretrained(base, path).eval()
        if device == "cpu":
            model = model.to(device)

        s_on, used = singular_spectrum(model, tokenizer, texts, args.seq_len, device)
        # OFF = disable_adapter()，與 gen_compare.py 的 base 對照同一條路徑。
        # 每個 adapter 都算一次：不同 adapter 的 OFF 應該完全相同，
        # 對不上就代表載入或 mask 有 bug，是免費的 sanity check。
        with model.disable_adapter():
            s_off, _ = singular_spectrum(model, tokenizer, texts, args.seq_len, device)

        r_on, y_on = effective_rank(s_on, args.threshold)
        r_off, y_off = effective_rank(s_off, args.threshold)
        # rank 是量化的（k/min(seq_len,d)，本設定下一格 = 0.78 pt），所以「+0.0」
        # 分不出「沒變」和「變動小於一格」。累積曲線的最大偏離是連續的，
        # 補上它才看得出實際擾動量——實測它的排序與 utility 損害一致。
        dev = (y_on - y_off).abs().max().item()
        results[label] = (r_on, r_off, used, dev)
        curves[label], curves[f"base@{label}"] = y_on, y_off
        print(f"  {label} ON = {r_on:.1f}%   OFF(base) = {r_off:.1f}%   "
              f"曲線偏離 = {dev:.2e}   （{used} 個樣本）")
        del model, base
        if device == "cuda":
            torch.cuda.empty_cache()

    bases = [v[1] for v in results.values()]
    print(f"\n{'='*74}\neffective rank @ 累積奇異值 {args.threshold}\n{'='*74}")
    print(f"{'臂':<16}{'rank(%)':>10}{'vs base':>12}{'曲線偏離':>14}")
    print(f"{'base':<16}{bases[0]:>10.1f}{'—':>12}{'—':>14}")
    for label, (r_on, r_off, _, dev) in sorted(results.items(), key=lambda kv: -kv[1][3]):
        print(f"{label:<16}{r_on:>10.1f}{r_on - r_off:>+12.1f}{dev:>14.2e}")
    print("\n「曲線偏離」= 累積奇異值曲線與 base 的最大差距，連續值。rank 一格 = "
          f"{100.0 / min(args.seq_len, 3072):.2f} pt，")
    print("小於一格的變動在 rank 欄會顯示 +0.0，要看這一欄才分得出來。")

    if len(bases) > 1 and max(bases) - min(bases) > 0.5:
        print(f"\n⚠️  各臂的 OFF(base) 不一致（{min(bases):.1f}–{max(bases):.1f}%）—— "
              "理論上應相同，先查 adapter 載入是否正確再看結論")

    print("\n判讀：")
    print("  2026-09-13 實測（Phi-3 + ProSec DPO 四臂）否證了 PT-PEFT 的兩半前提——")
    print("  LoRA 沒有讓 rank 塌（+0.0），prefix 反而把 rank 推高（+1.6 / +9.4）。")
    print("  但「曲線偏離」的排序與 utility 損害一致（4/4）：")
    print("    prefix16 0.148 > prefix8 0.0254 > lora_all 0.00805 > lora_qkv 0.00644")
    print("    utility  −16.91  >  −7.22       >  −4.12          >  +1.03")
    print("  → 決定 utility 的是**擾動表示空間的幅度**（往哪個方向都算），")
    print("    不是 PT-PEFT 假設的「rank 塌陷」。rank 高不等於語意豐富。")

    if args.csv_dir:
        os.makedirs(args.csv_dir, exist_ok=True)
        p = os.path.join(args.csv_dir, "cumulative_singular_values.csv")
        keys = list(curves)
        with open(p, "w") as f:
            f.write("rank_ratio_pct," + ",".join(keys) + "\n")
            for i in range(len(curves[keys[0]])):
                ratio = 100.0 * (i + 1) / len(curves[keys[0]])
                f.write(f"{ratio:.4f}," + ",".join(f"{curves[k][i]:.6f}" for k in keys) + "\n")
        print(f"\n曲線已寫入 {p}")


if __name__ == "__main__":
    main()
