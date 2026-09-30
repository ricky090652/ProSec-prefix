"""Turn ProSec D_sec into DuoSteer-style intra-prompt pairs.

A D_sec row is one vulnerability-inducing instruction with two Phi-3 outputs:
chosen = y_f (fixed), rejected = y_v (vulnerable). That is an intra-prompt pair
(p, r_safe, r_vuln). D_norm is not a safe/vuln contrast and is skipped.

Split is by instruction: 27,400 pairs share 11,939 instructions, so a pair-level
split would leak prompts across sides.

Outputs in --out_dir:
  intra.jsonl    vectors / probes / knockout
  heldout.jsonl  instructions disjoint from intra, for steering config selection
Row: {"id", "src_id", "cwe_id", "lang", "prompt", "safe_code", "vuln_code", "pref_id"}

Sampling keeps the natural CWE distribution (no stratification), as LoRA sees it.

Usage:
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
    """'CWE-022' / 'cwe-22' / '22' -> '22'."""
    return str(cwe).lower().removeprefix("cwe-").lstrip("0") or "0"


def prompt_hash(prompt):
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]


def load_dsec(train_file):
    rows = []
    with open(train_file) as f:
        for line in f:
            r = json.loads(line)
            if "benign" not in r:
                raise SystemExit("no 'benign' field; re-run data/convert_prosec_to_pref.py")
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
    print(f"{name}: {len(rows)} pairs / {n_prompts} prompts")
    print(f"  CWE : {dict(Counter(r['cwe_id'] for r in rows).most_common())}")
    print(f"  lang: {dict(Counter(r['lang'] for r in rows).most_common())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", default="data/train_pref.jsonl")
    ap.add_argument("--out_dir", default="data/steering")
    ap.add_argument("--heldout_ratio", type=float, default=0.05, help="fraction of prompts held out")
    ap.add_argument("--max_pairs", type=int, default=None,
                    help="random subsample of intra (natural distribution); held-out is not subsampled")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load_dsec(args.train_file)
    summary("D_sec", rows)

    intra, held = split_by_prompt(rows, args.heldout_ratio, args.seed)
    if args.max_pairs and len(intra) > args.max_pairs:
        intra = random.Random(args.seed).sample(intra, args.max_pairs)
    assert not ({r["src_id"] for r in intra} & {r["src_id"] for r in held}), "prompt overlap"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write(intra, out / "intra.jsonl", "intra")
    write(held, out / "heldout.jsonl", "heldout")
    summary("intra", intra)
    summary("heldout", held)
    print(f"-> {out}/intra.jsonl, {out}/heldout.jsonl")


if __name__ == "__main__":
    main()
