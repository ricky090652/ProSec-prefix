"""把 ProSec 偏好資料的 D_sec 轉成 DuoSteer 的 intra-prompt pairs。

D_sec（benign=False）每筆是同一條誘發漏洞的指令底下的兩份 Phi-3 程式碼：
chosen = y_f（修好的碼）、rejected = y_v（漏洞碼）。這正好對應 DuoSteer 的
intra-prompt pair（同一個 prompt 的 safe / vuln），差別只在 DuoSteer 用 benign
prompt，這裡是 vulnerability-inducing prompt。D_norm 不是 safe/vuln 對比，不收。

切分以「指令」為單位：同一條指令平均對應 2.3 筆 pair（27,400 筆 / 11,939 條），
按 pair 切會讓同一題同時出現在兩邊（DuoSteer 釋出資料沒有 src_id，就是這樣洩漏的）。

輸出（--out_dir 底下）：
  intra.jsonl    算 steering 向量 / 訓練 probe 用
  heldout.jsonl  指令與 intra 完全不重疊，留給之後 steering 的設定篩選
每行：{"id", "src_id", "cwe_id", "lang", "prompt", "safe_code", "vuln_code", "pref_id"}

⚠️ 抽樣照原始分布，不做 CWE 分層 —— 這與 LoRA 看到的資料分布相同，
   但也代表合併向量會被 CWE-338（約 47%）主導。

用法：
  python steering/build_pairs.py --train_file data/train_pref.jsonl \
      --out_dir data/steering --heldout_ratio 0.05 --max_pairs 4000
"""
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def cwe_key(cwe):
    """'CWE-022' / 'cwe-22' / '22' → '22'，與 DuoSteer 的 _cwe_key 一致。"""
    return str(cwe).lower().removeprefix("cwe-").lstrip("0") or "0"


def prompt_hash(prompt):
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]


def load_dsec(train_file):
    rows = []
    with open(train_file) as f:
        for line in f:
            r = json.loads(line)
            if "benign" not in r:
                raise SystemExit(
                    "資料沒有 benign 欄位，分不出 D_sec；請用新版 "
                    "data/convert_prosec_to_pref.py 重新轉檔")
            if r["benign"]:
                continue
            rows.append({
                "src_id":    prompt_hash(r["prompt"]),
                "cwe_id":    cwe_key(r["cwe"]),
                "lang":      r.get("lang", ""),
                "prompt":    r["prompt"],
                "safe_code": r["chosen"],
                "vuln_code": r["rejected"],
                "pref_id":   r.get("id"),
            })
    return rows


def split_by_prompt(rows, heldout_ratio, seed):
    groups = defaultdict(list)
    for r in rows:
        groups[r["src_id"]].append(r)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_held = round(len(keys) * heldout_ratio)
    held = [r for k in keys[:n_held] for r in groups[k]]
    intra = [r for k in keys[n_held:] for r in groups[k]]
    return intra, held


def write(rows, path, tag):
    with open(path, "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"id": f"{tag}-{i}", **r}, ensure_ascii=False) + "\n")


def summary(name, rows):
    n_prompts = len({r["src_id"] for r in rows})
    by_cwe = Counter(r["cwe_id"] for r in rows).most_common()
    by_lang = Counter(r["lang"] for r in rows).most_common()
    print(f"{name}: {len(rows)} pairs / {n_prompts} 條指令")
    print(f"  CWE : {dict(by_cwe)}")
    print(f"  lang: {dict(by_lang)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", default="data/train_pref.jsonl")
    ap.add_argument("--out_dir", default="data/steering")
    ap.add_argument("--heldout_ratio", type=float, default=0.05,
                    help="以指令為單位留出的比例")
    ap.add_argument("--max_pairs", type=int, default=None,
                    help="intra 隨機抽 N 筆（照原始分布）；held-out 不抽")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load_dsec(args.train_file)
    summary("D_sec", rows)

    intra, held = split_by_prompt(rows, args.heldout_ratio, args.seed)
    if args.max_pairs and len(intra) > args.max_pairs:
        intra = random.Random(args.seed).sample(intra, args.max_pairs)

    assert not ({r["src_id"] for r in intra} & {r["src_id"] for r in held}), \
        "intra 與 held-out 的指令重疊"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write(intra, out / "intra.jsonl", "intra")
    write(held, out / "heldout.jsonl", "heldout")
    summary("intra", intra)
    summary("heldout", held)
    print(f"→ {out}/intra.jsonl、{out}/heldout.jsonl")


if __name__ == "__main__":
    main()
