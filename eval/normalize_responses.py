"""Normalize generated responses so ProSec's detect_all.py extracts code correctly.

Problem found: detect_all.py's parse_code_blocks only keeps lines INSIDE ``` fences.
Many instruct-model responses output RAW code with no fences (e.g. Phi-3 often emits
`#include ...` directly). Those get dropped -> counted as "safe" -> the vulnerable
ratio is diluted ~6x (in our run, 84% of code blocks were empty for this reason).

Fix: for any response that contains no fenced code block, wrap the whole response in a
code fence so detect_all.py analyzes the actual code. Responses that already contain a
fenced block are left unchanged.

Usage:
  python eval/normalize_responses.py --in_file outputs/icd_paper.off.jsonl \
                                     --out      outputs/icd_paper.off.norm.jsonl
Then run detect_all.py on the .norm.jsonl and score as usual.
"""
import argparse
import json

FENCE = "```"


def has_code_block(text):
    """True if text has a non-empty fenced code block."""
    in_block = False
    for line in text.split("\n"):
        if FENCE in line:
            in_block = not in_block
        elif in_block and line.strip():
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n_resp, n_wrapped = 0, 0
    with open(args.out, "w") as fo:
        for line in open(args.in_file):
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            new = []
            for resp in e.get("responses", []):
                n_resp += 1
                if has_code_block(resp):
                    new.append(resp)
                else:
                    new.append(f"{FENCE}\n{resp}\n{FENCE}")  # wrap raw code
                    n_wrapped += 1
            e["responses"] = new
            fo.write(json.dumps(e, ensure_ascii=False) + "\n")

    pct = 100.0 * n_wrapped / n_resp if n_resp else 0.0
    print(f"responses={n_resp}, wrapped fence-less={n_wrapped} ({pct:.1f}%) -> {args.out}")


if __name__ == "__main__":
    main()
