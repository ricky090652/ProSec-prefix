"""安全性改善有多少來自「學會訓練過的 CWE」、多少來自「泛化到沒見過的 CWE」？

ProSec 訓練資料涵蓋 12 個 CWE，CyberSecEval 的 instruct 集涵蓋 50 個。
交集大約只有評測題目的三成——也就是說**大部分測到的改善都是泛化**，
不是學會訓練標的。這件事在 ProSec 論文裡沒有拆開報告。

而 prefix 與 LoRA 的泛化能力是否相同，沒有人量過。這正是本工作能補的洞，
而且**不需要任何新訓練或新生成**：四臂的全量評測結果都已算好，每題都有 CWE 標籤，
重新拆帳就有答案。

兩種可能的結果都有價值：
  seen 與 unseen 的 Δ 差不多
      → 兩種方法都在學「通用的安全寫作習慣」，而非個別 CWE 的樣板
  某一方在 unseen 上明顯較差
      → 那一方比較像在背訓練過的 CWE，泛化較弱。這是關於**方法機制**的發現，
        而不只是「誰的分數高」

用法：
  python eval/score_generalization.py \\
      --off outputs/full_shared.off.norm.jsonl.detected.jsonl \\
      --on outputs/full_lora.on.norm.jsonl.detected.jsonl \\
      --on outputs/full_lora_qkv.on.norm.jsonl.detected.jsonl \\
      --on outputs/full_prefix.on.norm.jsonl.detected.jsonl \\
      --on outputs/full_prefix_nvt8.on.norm.jsonl.detected.jsonl
"""
import argparse
import json
import os
import re
from collections import defaultdict


def norm_cwe(s):
    """'CWE-338' / 'cwe-0338' / '338' → 338。格式不一致時仍能對得起來。"""
    if s is None:
        return None
    m = re.search(r"(\d+)", str(s))
    return int(m.group(1)) if m else None


def trained_cwes(train_file):
    s = set()
    with open(train_file) as f:
        for line in f:
            if line.strip():
                c = norm_cwe(json.loads(line).get("cwe"))
                if c is not None:
                    s.add(c)
    return s


def load(path, seen):
    """回傳 {'seen': [vul, total], 'unseen': [...]} 與逐 CWE 的統計。"""
    grp = {"seen": [0, 0], "unseen": [0, 0]}
    per_cwe = defaultdict(lambda: [0, 0])
    n_nocwe = 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            c = norm_cwe(e.get("cwe"))
            if c is None:
                n_nocwe += 1
                continue
            is_vul = int(len(e.get("detection_results", [])) > 0)
            k = "seen" if c in seen else "unseen"
            grp[k][1] += 1
            grp[k][0] += is_vul
            per_cwe[c][1] += 1
            per_cwe[c][0] += is_vul
    return grp, per_cwe, n_nocwe


def rate(vt):
    v, t = vt
    return 100.0 * v / t if t else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True, help="共用的 OFF .detected.jsonl")
    ap.add_argument("--on", action="append", required=True,
                    help="各臂的 ON .detected.jsonl，可重複給")
    ap.add_argument("--train_file", default="data/train_pref.jsonl")
    ap.add_argument("--top_cwe", type=int, default=12,
                    help="逐 CWE 表列出題數最多的前幾個")
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    seen = trained_cwes(args.train_file)
    print(f"訓練涵蓋 {len(seen)} 個 CWE：" +
          ", ".join(f"CWE-{c}" for c in sorted(seen)))

    off_grp, off_cwe, n_nocwe = load(args.off, seen)
    ns, nu = off_grp["seen"][1], off_grp["unseen"][1]
    tot = ns + nu
    if n_nocwe:
        print(f"⚠️  {n_nocwe:,} 筆沒有 cwe 欄位，已排除")
    if tot == 0:
        raise SystemExit("OFF 檔裡沒有任何帶 cwe 的紀錄——檢查 --off 路徑")
    print(f"評測題數（含樣本）：seen {ns:,}（{ns/tot:.1%}）"
          f"　unseen {nu:,}（{nu/tot:.1%}）　共 {tot:,}")
    print(f"→ **{nu/tot:.0%} 的評測題目屬於訓練時沒看過的 CWE**")

    rows = []
    print("\n" + "=" * 78)
    print("安全性改善的拆帳（漏洞率 %，越低越好；Δ 為 ON − OFF）")
    print("=" * 78)
    print(f"{'臂':<18}{'seen Δ':>11}{'unseen Δ':>11}{'全體 Δ':>11}"
          f"{'unseen/seen':>13}")
    off_s, off_u = rate(off_grp["seen"]), rate(off_grp["unseen"])
    off_all = rate([off_grp["seen"][0] + off_grp["unseen"][0], tot])
    print(f"{'(base 漏洞率)':<18}{off_s:>10.2f}%{off_u:>10.2f}%{off_all:>10.2f}%")
    print("-" * 78)
    for p in args.on:
        g, pc, _ = load(p, seen)
        ds = rate(g["seen"]) - off_s
        du = rate(g["unseen"]) - off_u
        da = rate([g["seen"][0] + g["unseen"][0],
                   g["seen"][1] + g["unseen"][1]]) - off_all
        name = os.path.basename(p).replace(".on.norm.jsonl.detected.jsonl", "") \
                                  .replace("full_", "")
        ratio = f"{du/ds:>12.2f}" if abs(ds) > 1e-9 else f"{'—':>12}"
        print(f"{name:<18}{ds:>+10.2f}{du:>+10.2f}{da:>+10.2f}{ratio}")
        rows.append({"arm": name, "seen_delta": ds, "unseen_delta": du,
                     "all_delta": da,
                     "per_cwe": {f"CWE-{c}": [v, t] for c, (v, t) in pc.items()}})

    print("\n判讀 unseen/seen 這一欄（兩個 Δ 的比值）：")
    print("  ≈ 1.0   兩邊改善幅度相當 → 學到的是通用的安全寫作習慣")
    print("  < 1.0   unseen 改善較少 → 較依賴訓練過的 CWE，泛化較弱")
    print("  > 1.0   unseen 改善反而更多 → 訓練標的上已接近飽和，或 base 在該處較差")

    # 逐 CWE：題數最多的前幾個，標出有沒有訓練過
    big = sorted(off_cwe.items(), key=lambda kv: -kv[1][1])[:args.top_cwe]
    print("\n" + "=" * 78)
    print(f"逐 CWE（題數前 {args.top_cwe} 名；★ = 訓練集內）")
    print("=" * 78)
    on_maps = []
    for p in args.on:
        g, pc, _ = load(p, seen)
        on_maps.append((os.path.basename(p).replace(".on.norm.jsonl.detected.jsonl", "")
                        .replace("full_", ""), pc))
    hdr = f"{'CWE':<12}{'題數':>7}{'base':>9}"
    for n, _ in on_maps:
        hdr += f"{n[:11]:>12}"
    print(hdr)
    for c, vt in big:
        mark = "★" if c in seen else " "
        line = f"{mark}CWE-{c:<7}{vt[1]:>7,}{rate(vt):>8.1f}%"
        for _, pc in on_maps:
            line += f"{rate(pc[c]) - rate(vt):>+12.2f}" if c in pc else f"{'—':>12}"
        print(line)

    if args.json_out:
        json.dump({"seen_cwes": sorted(seen), "arms": rows},
                  open(args.json_out, "w"), indent=2, ensure_ascii=False)
        print(f"\n→ {args.json_out}")


if __name__ == "__main__":
    main()
