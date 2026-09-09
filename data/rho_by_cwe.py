"""依 CWE 拆開量編輯局部性——ProSec 是不是「修補」與「重寫」的混合體？

`edit_locality.py` 在全體資料上量到 ρ 中位數 0.358、平均 17 個 hunk，我們據此
說「ProSec 的配對是重寫，不像 SVEN 的修補」。但那是**混出來的平均值**。

CWE-338（弱亂數）的修補通常就是一行替換——`random` → `secrets`、
`rand()` → `arc4random()`——本質上和 SVEN 的資料同型。而它佔訓練資料的 46.4%。
反過來 CWE-502（反序列化）、CWE-22（路徑穿越）往往要在外面包一整層防護，
那才是重寫。

若真是混合體，那麼 SVEN / PTC 那一套（token-level masking、per-CWE prefix）
應該只在 patch-like 的子集上生效，而不是整份資料。這支腳本把 ρ 依 CWE 拆開，
用的是與 `edit_locality.py` 完全相同的定義，數字可以直接並排。

用法：
  python data/rho_by_cwe.py                        # 每個 CWE 抽 800 筆
  python data/rho_by_cwe.py --per_cwe_limit 0      # 全部（慢）
"""
import argparse
import difflib
import json
from collections import defaultdict


def pct(v, p):
    s = sorted(v)
    return s[max(0, min(len(s) - 1, int(round(p / 100 * (len(s) - 1)))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", default="data/train_pref.jsonl")
    ap.add_argument("--tokenizer", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--per_cwe_limit", type=int, default=800,
                    help="每個 CWE 最多抽幾筆（0 = 全部）。ρ 的分位數在幾百筆就穩了")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    enc = lambda s: tok(s, add_special_tokens=False).input_ids  # noqa: E731

    by = defaultdict(list)
    for l in open(args.train_file):
        if l.strip():
            e = json.loads(l)
            by[e.get("cwe", "?")].append(e)

    import random
    rng = random.Random(args.seed)
    rows = []
    for cwe, es in by.items():
        n_all = len(es)
        if args.per_cwe_limit and n_all > args.per_cwe_limit:
            es = rng.sample(es, args.per_cwe_limit)
        rho, hunks, n1 = [], [], 0
        for e in es:
            a, b = enc(e["chosen"]), enc(e["rejected"])
            ops = difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
            ca = sum(i2 - i1 for t, i1, i2, _, _ in ops if t != "equal")
            cb = sum(j2 - j1 for t, _, _, j1, j2 in ops if t != "equal")
            rho.append(max(ca, cb) / max(1, max(len(a), len(b))))
            nh = sum(1 for t, *_ in ops if t != "equal")
            hunks.append(nh)
            n1 += (nh == 1)
        rows.append({"cwe": cwe, "n_all": n_all, "n_sampled": len(es),
                     "rho_p50": pct(rho, 50), "rho_p25": pct(rho, 25),
                     "hunks_mean": sum(hunks) / len(hunks),
                     "single_hunk": n1 / len(es)})
        print(f"  {cwe} 完成（{len(es)}/{n_all}）", flush=True)

    rows.sort(key=lambda r: r["rho_p50"])
    print("\n" + "=" * 84)
    print("依 CWE 的編輯局部性（ρ 越小越像「修補」，越大越像「重寫」）")
    print("=" * 84)
    print(f"{'CWE':<11}{'筆數':>8}{'ρ p25':>9}{'ρ p50':>9}{'hunk':>8}{'單一hunk':>10}   判讀")
    for r in rows:
        v = ("← patch-like" if r["rho_p50"] < 0.15 else
             "  中間" if r["rho_p50"] < 0.35 else "← rewrite-like")
        print(f"{r['cwe']:<11}{r['n_all']:>8,}{r['rho_p25']:>9.3f}{r['rho_p50']:>9.3f}"
              f"{r['hunks_mean']:>8.1f}{r['single_hunk']:>9.1%}   {v}")
    print("\n對照：SVEN 同尺 ρ p50 = 0.058、hunk 4.1、單一 hunk 21.0%")
    print("      ProSec 全體  ρ p50 = 0.358、hunk 17.0、單一 hunk  0.1%")
    print("\nρ 低的 CWE 是 SVEN/PTC 那套方法（masking、per-CWE prefix）最可能生效之處；")
    print("ρ 高的則是「包一層防護」型的重寫，那套方法的前提在那裡不成立。")

    if args.json_out:
        json.dump(rows, open(args.json_out, "w"), indent=2, ensure_ascii=False)
        print(f"\n→ {args.json_out}")


if __name__ == "__main__":
    main()
