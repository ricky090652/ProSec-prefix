"""Safety vs correctness vectors at the causal heads (DuoSteer §3.5, §7, Appendix D).

Reports what the paper reports:
  overlap   |top-k safety-causal heads ∩ top-k correctness-causal heads| (§3.5)
  cosine    mean cos(v_safety, v_correctness) over the top-k safety-causal heads
            (Appendix D, Table 7) and over every head (§7)
Paper (Llama, per CWE): Table 7 cosines lie in [-0.24, +0.16]; all-head means in
[-0.16, +0.08]. Anti-aligned or near-zero CWEs gained the most from DuoSteer.

Head ranking is the one steering uses (hooks.rank_heads: delta_vs_baseline ascending).

Usage:
  python steering/vector_cosine.py \
      --safety_vectors $REP/vectors --safety_heads $REP/causal/head_causal_results.json \
      --correct_vectors $REP_C/vectors --correct_heads $REP_C/causal/head_causal_results.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hooks import rank_heads


def unit(vector_dir, layer, head, method):
    path = Path(vector_dir) / "head" / method / f"steering_vector_head_{layer:02d}_{head:02d}.pt"
    return torch.load(path, map_location="cpu", weights_only=True)["sv_unit"].float()


def cos(a, b):
    return float(a @ b / (a.norm() * b.norm() + 1e-12))


def main(args):
    safety, _ = rank_heads(args.safety_heads)
    correct, _ = rank_heads(args.correct_heads)
    s_keys = [(h["layer"], h["head"]) for h in safety]
    c_keys = [(h["layer"], h["head"]) for h in correct]

    def head_cos(l, h):
        return cos(unit(args.safety_vectors, l, h, args.method),
                   unit(args.correct_vectors, l, h, args.method))

    print(f"{'k':>5}{'overlap':>10}{'mean cos @ top-k safety heads':>32}")
    for k in args.ks:
        if k > len(s_keys):
            break
        overlap = len(set(s_keys[:k]) & set(c_keys[:k]))
        mean = sum(head_cos(l, h) for l, h in s_keys[:k]) / k
        print(f"{k:>5}{overlap:>6} / {k:<4}{mean:>+20.3f}")

    files = sorted((Path(args.safety_vectors) / "head" / args.method).glob("steering_vector_head_*.pt"))
    all_heads = [tuple(int(x) for x in f.stem.split("_")[-2:]) for f in files]
    all_cos = [head_cos(l, h) for l, h in all_heads]
    print(f"all {len(all_cos)} heads: mean cos {sum(all_cos) / len(all_cos):+.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--safety_vectors", required=True, help="<rep_dir>/vectors of the D_sec pairs")
    ap.add_argument("--safety_heads", required=True, help="safety head_causal_results.json")
    ap.add_argument("--correct_vectors", required=True, help="<rep_dir>/vectors of the correctness pairs")
    ap.add_argument("--correct_heads", required=True, help="correctness head_causal_results.json")
    ap.add_argument("--method", default="mean_diff", choices=["mean_diff", "probe"])
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256],
                    help="Appendix D budgets")
    main(ap.parse_args())
