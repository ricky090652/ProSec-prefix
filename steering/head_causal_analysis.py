"""Causal head knockout (paper §3.3, Eq. 1-2) on ProSec intra pairs.

Port of DuoSteer localization/head_causal_analysis.py. For each of the top-k
probe-ranked heads, zero its o_proj input slice at every response position and
teacher-force both sides under the same prompt p:
    delta = L(r_safe | p) - L(r_vuln | p)       L = mean token log-prob
    Delta = mean over pairs of (delta_ko - delta_base)
Delta < 0 => safe-promoting head. causal_rank 0 = most negative Delta.

Differences from the original:
  1. Pairs are D_sec intra pairs scored under their own instruction. The original
     needs cross pairs only because p^b rarely yields vulnerable code (§3.2);
     D_sec has thousands of vulnerable samples per CWE.
  2. Pairs are the probe's validation split ("averaged over validation pairs",
     §3.3), disjoint from the pairs that build the steering vectors. The original
     uses every cross pair of the CWE.
  3. Tokenization and response mask are shared with extract_representations.py.
  4. --top_k defaults to 256 (paper); original default is 16.
  5. Encodings are built once and reused for every head.
  6. Reports the Spearman rho between probe rank and Delta, and the probe rank of
     the most causal head (§5).

Usage:
  python steering/head_causal_analysis.py --rep_dir <extract output dir> --bf16
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from extract_representations import prepare_batch
from train_probe import make_split_indices


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_split_pairs(metadata, pairs_file, split, val_ratio, seed, max_pairs):
    """Recreate the probe split and return the full pair records of one side."""
    train_idx, val_idx = make_split_indices(metadata, val_ratio, seed)
    idx = {"val": val_idx, "train": train_idx, "all": sorted(train_idx + val_idx)}[split]
    wanted = [metadata["pairs"][i]["id"] for i in idx]
    by_id = {}
    with open(pairs_file) as f:
        for line in f:
            p = json.loads(line)
            by_id[p["id"]] = p
    missing = [i for i in wanted if i not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} pair ids not in {pairs_file}, e.g. {missing[:3]}")
    pairs = [by_id[i] for i in wanted]
    return pairs[:max_pairs] if max_pairs else pairs


def build_batches(tokenizer, pairs, side, system_prompt, batch_size, device):
    return [prepare_batch(tokenizer,
                          [{"prompt": p["prompt"], "code": p[f"{side}_code"]}
                           for p in pairs[i:i + batch_size]],
                          system_prompt, device)
            for i in range(0, len(pairs), batch_size)]


# --------------------------------------------------------------------------- #
# Knockout and scoring
# --------------------------------------------------------------------------- #

class HeadKnockout:
    """Pre-hook on one o_proj that zeros head j's slice at masked (response) positions."""

    def __init__(self, o_proj, head_idx, head_dim):
        self.slice = slice(head_idx * head_dim, (head_idx + 1) * head_dim)
        self.mask = None   # (b, s) bool, set per batch
        self._handle = o_proj.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        x = inputs[0].clone()
        x[..., self.slice] = x[..., self.slice].masked_fill(self.mask.unsqueeze(-1), 0.0)
        return (x,)

    def remove(self):
        self._handle.remove()


@torch.no_grad()
def mean_logprob(model, enc, resp):
    """Teacher-forced mean log-prob of response tokens; token t is scored by logit t-1."""
    logits = model(**enc).logits[:, :-1].float()
    lp = torch.log_softmax(logits, -1).gather(-1, enc["input_ids"][:, 1:, None]).squeeze(-1)
    m = resp[:, 1:].float()
    return ((lp * m).sum(1) / m.sum(1)).tolist()


def score(model, batches, knocker=None):
    out = []
    for enc, resp in batches:
        if knocker is not None:
            knocker.mask = resp
        out.extend(mean_logprob(model, enc, resp))
    return out


# --------------------------------------------------------------------------- #
# Stats and plots
# --------------------------------------------------------------------------- #

def spearman(x, y):
    """Spearman rho with a normal-approximation two-sided p (no scipy here; no ties expected)."""
    rx, ry = (np.argsort(np.argsort(v)).astype(float) for v in (x, y))
    rho = float(np.corrcoef(rx, ry)[0, 1])
    p = math.erfc(abs(rho) * math.sqrt(len(x) - 1) / math.sqrt(2))
    return rho, p


def plot_heatmap(heads, key, n_layers, n_heads, label, title, out_path, top_n=64):
    """Layer × head grid of one metric; untested heads grey, top-n by ascending value circled."""
    grid = np.full((n_layers, n_heads), np.nan)
    for r in heads:
        grid[r["layer"] - 1, r["head"] - 1] = r[key]
    cmap = plt.cm.RdBu.copy()
    cmap.set_bad(color="#dddddd")
    vals = grid[~np.isnan(grid)]
    norm = mcolors.TwoSlopeNorm(vmin=min(vals.min(), -1e-6), vcenter=0.0, vmax=max(vals.max(), 1e-6))

    fig, ax = plt.subplots(figsize=(max(8, n_heads * 0.35 + 2), max(6, n_layers * 0.35 + 2)))
    im = ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(label, fontsize=9)
    for r in sorted(heads, key=lambda r: r[key])[:top_n]:
        ax.plot(r["head"] - 1, r["layer"] - 1, "o", mfc="none", mec="white", mew=1.2, ms=7, zorder=5)
    ax.set_xlabel("Head index")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([str(i + 1) for i in range(n_heads)], fontsize=8)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([str(i + 1) for i in range(n_layers)], fontsize=8)
    ax.set_title(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(args):
    rep_dir = Path(args.rep_dir)
    with open(rep_dir / "metadata.json") as f:
        metadata = json.load(f)
    head_results = Path(args.head_results or rep_dir / "probes/plots/head_accuracy_results.json")
    out_dir = Path(args.output_dir or rep_dir / "causal")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "head_causal_checkpoint.json"

    pairs = load_split_pairs(metadata, args.pairs_file or metadata["input_file"],
                             args.split, args.val_ratio, args.seed, args.max_pairs)
    with open(head_results) as f:
        top_heads = json.load(f)[:args.top_k]
    print(f"{rep_dir}\n  cwe={metadata['cwe_id']} split={args.split} pairs={len(pairs)} "
          f"prompts={len({p['src_id'] for p in pairs})} top_k={len(top_heads)}")

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    o_projs = [m.o_proj for m in model.modules() if hasattr(m, "o_proj")]
    assert len(o_projs) == n_layers, f"found {len(o_projs)} o_proj, expected {n_layers}"
    sys_prompt = metadata.get("system_prompt")

    batches = {s: build_batches(tokenizer, pairs, s, sys_prompt, args.batch_size, device)
               for s in ("safe", "vuln")}

    # resume
    base_delta, results = None, []
    if ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text())
        base_delta, results = ckpt["baseline_delta"], ckpt["head_results"]
        print(f"  resuming: {len(results)}/{len(top_heads)} heads done")
    done = {(r["layer"], r["head"]) for r in results}

    def save_ckpt():
        ckpt_path.write_text(json.dumps({"baseline_delta": base_delta, "head_results": results}))

    if base_delta is None:
        base = [s - v for s, v in zip(score(model, batches["safe"]), score(model, batches["vuln"]))]
        base_delta = float(np.mean(base))
        print(f"  baseline delta = {base_delta:+.4f}  (>0: prefers safe)")
        save_ckpt()

    for rank, h in enumerate(top_heads):
        if (h["layer"], h["head"]) in done:
            continue
        knocker = HeadKnockout(o_projs[h["layer"] - 1], h["head"] - 1, head_dim)
        try:
            deltas = [s - v for s, v in zip(score(model, batches["safe"], knocker),
                                            score(model, batches["vuln"], knocker))]
        finally:
            knocker.remove()
        mean_d = float(np.mean(deltas))
        results.append({
            "layer": h["layer"], "head": h["head"], "probe_rank": rank,
            "probe_accuracy": h["val_accuracy"], "mean_delta": mean_d,
            "std_delta": float(np.std(deltas)), "delta_vs_baseline": mean_d - base_delta,
            "n_pairs": len(deltas), "per_pair_deltas": deltas,
        })
        save_ckpt()
        print(f"  [{rank + 1}/{len(top_heads)}] L{h['layer']:02d}H{h['head']:02d} "
              f"probe={h['val_accuracy']:.3f} Delta={mean_d - base_delta:+.5f}")

    order = sorted(results, key=lambda r: r["mean_delta"])
    for i, r in enumerate(order):
        r["causal_rank"] = i
    rho, p = spearman([r["probe_rank"] for r in results], [r["mean_delta"] for r in results])

    out = {
        "cwe_id": metadata["cwe_id"], "model": args.model, "intervention": "knockout",
        "split": args.split, "n_pairs": len(pairs), "pair_ids": [x["id"] for x in pairs],
        "top_k": len(top_heads), "baseline_delta": base_delta,
        "spearman_probe_rank_vs_delta": {"rho": rho, "p_normal_approx": p},
        "heads": order,
    }
    (out_dir / "head_causal_results.json").write_text(json.dumps(out, indent=2))
    ckpt_path.unlink()

    plot_heatmap(results, "mean_delta", n_layers, n_heads,
                 "delta = L(safe | ko) - L(vuln | ko)", "Knockout delta per head (grey = not tested)",
                 out_dir / "head_causal_delta.png")
    plot_heatmap(results, "delta_vs_baseline", n_layers, n_heads,
                 "Delta = delta_ko - delta_base (<0: safe-promoting)",
                 "Knockout Delta vs baseline per head (grey = not tested)",
                 out_dir / "head_causal_delta_vs_base.png")

    print(f"\n  baseline delta = {base_delta:+.4f}")
    print(f"  Spearman(probe rank, Delta) = {rho:+.3f} (p~{p:.2f}); "
          f"most causal head at probe rank {order[0]['probe_rank']}")
    print(f"  {'head':<8}{'causal':>7}{'probe':>7}{'acc':>7}{'Delta':>10}{'std':>9}")
    for r in order[:args.show]:
        print(f"  L{r['layer']:02d}H{r['head']:02d}  {r['causal_rank']:>7}{r['probe_rank']:>7}"
              f"{r['probe_accuracy']:>7.3f}{r['delta_vs_baseline']:>+10.5f}{r['std_delta']:>9.4f}")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep_dir", required=True, help="output dir of extract_representations.py")
    ap.add_argument("--pairs_file", default=None, help="default: metadata input_file")
    ap.add_argument("--head_results", default=None,
                    help="default: <rep_dir>/probes/plots/head_accuracy_results.json")
    ap.add_argument("--output_dir", default=None, help="default: <rep_dir>/causal")
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--split", choices=["val", "train", "all"], default="val")
    ap.add_argument("--val_ratio", type=float, default=0.2, help="must match train_probe.py")
    ap.add_argument("--seed", type=int, default=42, help="must match train_probe.py")
    ap.add_argument("--top_k", type=int, default=256, help="paper: top-256 probe-ranked heads")
    ap.add_argument("--max_pairs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--show", type=int, default=20, help="rows in the printed summary")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(ap.parse_args())
