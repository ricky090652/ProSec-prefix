"""量測 ProSec 偏好對的「局部性」，並推算 SimPO 在這份資料上的 margin 可達上限。

兩個用途：
1. **S5 的前置**（計畫 §9.7.4）。masked SimPO 的前提是 chosen 與 rejected 高度相似、
   差異集中。若 edit ratio 其實很高，masking 就沒有意義，H3 直接不成立。
2. **診斷 S1 的 SimPO 失敗**。SimPO 的 margin 是「每 token 平均 log-prob 的差」，
   而完全相同的 token 對這個差幾乎沒有貢獻。所以可達的平均 margin 大約被
   edit ratio 壓住：

       可達 margin ≈ ρ x Δ      （ρ = 變動 token 比例，Δ = 那些 token 上的領先幅度）

   而 TRL 的 SimPO 要求 margin 超過 γ/β 才算滿足。若 ρxΔ 遠小於 γ/β，
   這個目標**結構上不可能達成**，loss 會一直卡在高梯度區把兩側一起往下推。

用法：
  python data/edit_locality.py --train_file data/train_pref.jsonl \
      --tokenizer microsoft/Phi-3-mini-4k-instruct --limit 3000

  # 想順便檢查目前的 beta/gamma 設定是否可達
  python data/edit_locality.py ... --beta 1.5 --gamma 0.5
"""
import argparse
import difflib
import json
import statistics as st
from collections import Counter


def pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


def summarize(name, vals, fmt="{:.3f}"):
    if not vals:
        print(f"{name:<22} —")
        return
    print(f"{name:<22} mean {fmt.format(st.mean(vals)):>9}  "
          f"p25 {fmt.format(pct(vals,25)):>9}  "
          f"p50 {fmt.format(pct(vals,50)):>9}  "
          f"p75 {fmt.format(pct(vals,75)):>9}  "
          f"p95 {fmt.format(pct(vals,95)):>9}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", default="data/train_pref.jsonl")
    ap.add_argument("--tokenizer", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--limit", type=int, default=3000,
                    help="隨機抽樣幾筆（tokenize 全量很慢；3000 筆已足夠估分布）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_length", type=int, default=1024,
                    help="訓練用的 max_length，用來估算截斷比例")
    ap.add_argument("--max_prompt_length", type=int, default=512)
    ap.add_argument("--beta", type=float, default=1.5, help="SimPO beta")
    ap.add_argument("--gamma", type=float, default=0.5, help="SimPO target margin")
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    rows = [json.loads(l) for l in open(args.train_file) if l.strip()]
    print(f"讀入 {len(rows)} 筆")
    if args.limit and args.limit < len(rows):
        import random
        random.Random(args.seed).shuffle(rows)
        rows = rows[:args.limit]
        print(f"隨機抽樣 {len(rows)} 筆（seed {args.seed}）")

    enc = lambda s: tok(s, add_special_tokens=False).input_ids  # noqa: E731

    lc, lr_, ldiff = [], [], []
    rho, hunks, lcp, lcs = [], [], [], []
    n_trunc_c = n_trunc_r = n_trunc_p = 0
    n_single_hunk = n_empty_diff = 0
    per_lang = Counter()

    for i, e in enumerate(rows, 1):
        if i % 500 == 0:
            print(f"  {i}/{len(rows)}")
        a, b = enc(e["chosen"]), enc(e["rejected"])
        p = enc(e["prompt"])
        lc.append(len(a)); lr_.append(len(b)); ldiff.append(len(a) - len(b))
        if len(p) > args.max_prompt_length:
            n_trunc_p += 1
        if len(p) + len(a) > args.max_length:
            n_trunc_c += 1
        if len(p) + len(b) > args.max_length:
            n_trunc_r += 1

        sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        ops = sm.get_opcodes()
        changed_a = sum(i2 - i1 for tag, i1, i2, _, _ in ops if tag != "equal")
        changed_b = sum(j2 - j1 for tag, _, _, j1, j2 in ops if tag != "equal")
        denom = max(len(a), len(b)) or 1
        rho.append(max(changed_a, changed_b) / denom)
        nh = sum(1 for tag, *_ in ops if tag != "equal")
        hunks.append(nh)
        if nh == 0:
            n_empty_diff += 1
        if nh == 1:
            n_single_hunk += 1
        # 頭尾相同的長度 = 完全浪費梯度的部分
        head = ops[0]
        lcp.append(head[2] - head[1] if head[0] == "equal" else 0)
        tail = ops[-1]
        lcs.append(tail[2] - tail[1] if tail[0] == "equal" else 0)
        per_lang[e.get("lang", "?")] += 1

    n = len(rows)
    print("\n" + "=" * 92)
    print("【長度與截斷】單位：token")
    print("=" * 92)
    summarize("chosen 長度", lc, "{:.0f}")
    summarize("rejected 長度", lr_, "{:.0f}")
    summarize("長度差 (c-r)", ldiff, "{:+.0f}")
    print(f"\nprompt 超過 max_prompt_length={args.max_prompt_length}："
          f"{n_trunc_p}/{n} ({100*n_trunc_p/n:.1f}%)")
    print(f"prompt+chosen 超過 max_length={args.max_length}："
          f"{n_trunc_c}/{n} ({100*n_trunc_c/n:.1f}%)")
    print(f"prompt+rejected 超過 max_length={args.max_length}："
          f"{n_trunc_r}/{n} ({100*n_trunc_r/n:.1f}%)")
    if max(n_trunc_c, n_trunc_r) > 0.1 * n:
        print("⚠️  截斷比例偏高。SimPO 除以 |y| 對截斷特別敏感："
              "chosen 通常較長、被砍得較多，會系統性扭曲比較。")

    print("\n" + "=" * 92)
    print("【局部性】")
    print("=" * 92)
    summarize("edit ratio ρ", rho)
    summarize("edit hunk 數", hunks, "{:.0f}")
    summarize("共同前綴長度", lcp, "{:.0f}")
    summarize("共同後綴長度", lcs, "{:.0f}")
    print(f"\n單一 contiguous diff（hunk=1）：{n_single_hunk}/{n} "
          f"({100*n_single_hunk/n:.1f}%)")
    print(f"token 化後完全相同（應丟棄）：{n_empty_diff}/{n} "
          f"({100*n_empty_diff/n:.1f}%)")
    wasted = (st.mean(lcp) + st.mean(lcs)) / max(st.mean(lc), 1)
    print(f"僅頭尾共同部分就佔 chosen 平均長度的 {100*wasted:.1f}%"
          "（完整序列 loss 花在這些 token 上的梯度全是浪費）")

    print("\n" + "=" * 92)
    print(f"【SimPO margin 可達性】β={args.beta}  γ={args.gamma}")
    print("=" * 92)
    target = args.gamma / args.beta
    print(f"TRL 的目標 margin = γ/β = {target:.3f} nats/token\n")
    print(f"{'diff token 上的領先 Δ':<24}" + "".join(f"{d:>12}" for d in (0.5, 1.0, 2.0, 4.0)))
    print("-" * 72)
    for label, r in (("ρ 中位數", pct(rho, 50)), ("ρ 平均", st.mean(rho)),
                     ("ρ p95（最樂觀）", pct(rho, 95))):
        cells = []
        for d in (0.5, 1.0, 2.0, 4.0):
            v = r * d
            cells.append(f"{v:.3f}{'✓' if v >= target else '✗'}")
        print(f"{label + f' ({r:.3f})':<24}" + "".join(f"{c:>12}" for c in cells))
    print("\n✓ = 該情境下可達成目標 margin；✗ = 結構上不可能")
    print("若整列都是 ✗，代表 γ 設得太高：loss 永遠停在高梯度區，"
          "優化器會持續往一個無法擴大 margin 的方向推，把兩側 log-prob 一起壓低。")
    print(f"→ 建議的 γ 上限 ≈ ρ_中位數 x Δ(≈1~2) x β = "
          f"{pct(rho,50)*1.5*args.beta:.3f} ~ {pct(rho,50)*2.0*args.beta:.3f}")

    print("\n【語言分布】" + "  ".join(f"{k}:{v}" for k, v in per_lang.most_common()))

    if args.json_out:
        out = dict(n=n, rho_mean=st.mean(rho), rho_p50=pct(rho, 50), rho_p95=pct(rho, 95),
                   hunks_p50=pct(hunks, 50), single_hunk_rate=n_single_hunk / n,
                   empty_diff_rate=n_empty_diff / n,
                   len_chosen_mean=st.mean(lc), len_rejected_mean=st.mean(lr_),
                   trunc_chosen_rate=n_trunc_c / n, trunc_rejected_rate=n_trunc_r / n,
                   simpo_target=target)
        json.dump(out, open(args.json_out, "w"), indent=2)
        print(f"\n已存 → {args.json_out}")


if __name__ == "__main__":
    main()
