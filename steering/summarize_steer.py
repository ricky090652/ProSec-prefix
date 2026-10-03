"""Vulnerable-code ratio for every steer_generate.py output in a directory.

Same metric as eval/score_detected.py (a code block is vulnerable if it has any ICD
detection), reusing its load_stats. SE = sqrt(p(1-p)/n) over code blocks.

Usage (after steering/score_steer.sh has run detection):
  python steering/summarize_steer.py --dir outputs/steer/cwe-502 \
      [--off outputs/full_shared.off.norm.jsonl.detected.jsonl]
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from score_detected import load_stats, ratio  # noqa: E402


def row(name, total, vul, per_lang, trunc):
    p = vul / total if total else 0.0
    se = 100 * math.sqrt(p * (1 - p) / total) if total else 0.0
    rates = {k: ratio(*per_lang[k]) for k in sorted(per_lang)}
    # ProSec reports the mean of per-language rates (Table 1 style)
    avg = f"{sum(rates.values()) / len(rates):8.2f}" if rates else f"{'':8}"
    langs = " ".join(f"{k}={v:.1f}" for k, v in rates.items())
    t = "" if trunc is None else f"{100 * trunc:5.1f}"
    print(f"  {name:<40}{total:>6}{ratio(vul, total):>8.2f}{se:>6.2f}{avg}{t:>7}   {langs}")


def main(args):
    d = Path(args.dir)
    files = sorted(d.glob("*.norm.jsonl.detected.jsonl"))
    if not files:
        raise SystemExit(f"no *.detected.jsonl in {d}; run steering/score_steer.sh first")
    print(f"{'':2}{'setting':<40}{'n':>6}{'V%':>8}{'SE':>6}{'langavg':>8}{'trunc%':>7}   per-lang V%")

    cwes = set()
    stats = []
    for f in files:
        name = f.name.split(".norm.jsonl")[0]
        total, vul, per_cwe, per_lang = load_stats(f)
        cwes |= set(per_cwe)
        src = d / f"{name}.jsonl"
        flags = [t for l in open(src) if l.strip() for t in json.loads(l).get("truncated", [])]
        stats.append((name, total, vul, per_lang, sum(flags) / len(flags) if flags else None))

    if args.off:
        # OFF restricted to the CWEs present here (valid for --instruct_json runs)
        _, _, off_cwe, _ = load_stats(args.off)
        v = sum(off_cwe[c][0] for c in cwes if c in off_cwe)
        t = sum(off_cwe[c][1] for c in cwes if c in off_cwe)
        row(f"OFF (shared) {sorted(cwes)}", t, v, {}, None)

    stats.sort(key=lambda s: (s[0] != "baseline", ratio(s[2], s[1])))
    for s in stats:
        row(*s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--off", default=None, help="shared OFF .detected.jsonl (eval-set runs only)")
    main(ap.parse_args())
