"""Compute steering vectors from extracted representations (paper §3.4).

Port of DuoSteer steering/extract_steering_vector.py. Same methods, normalization
and .pt contents:
  mean_diff  sv_unit = (mean(safe) - mean(vuln)) / ||.||
  probe      sv_unit = -w / ||w||   (probe labels safe=0 / vuln=1, so -w points to safe)
  norm=sigma sv = sv_unit * sigma_proj, sigma_proj = std of <x_i, sv_unit> over safe ∪ vuln
             (alpha=1 shifts the projection by one std; this is what the official code
             does, the paper's Eq. 3 text says per-component sigma)
  norm=raw   sv = raw difference (mean_diff) or raw -w (probe)

Difference from the original: vectors and sigma use the probe's train split
(paper: "calculated over the intra-prompt training pairs"); the original uses
every pair in the directory. --split all restores that.

Output <rep_dir>/vectors/{layer,head}/{mean_diff,probe}/
  steering_vector_layer_XX.pt / steering_vector_head_LL_HH.pt
    {"steering_vector", "sv_unit", "raw_norm", "proj_sigma", "norm", "method",
     ["safe_mean", "vuln_mean"]}
  metadata.json

Usage:
  python steering/extract_steering_vector.py --rep_dir <extract output dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_probe import make_split_indices


def proj_sigma(safe, vuln, sv_unit):
    return (torch.cat([safe, vuln]).float() @ sv_unit.float()).std().item()


def mean_diff_vector(safe, vuln, norm):
    safe_mean, vuln_mean = safe.mean(0), vuln.mean(0)
    raw = safe_mean - vuln_mean
    raw_norm = raw.norm().item()
    sv_unit = raw / raw_norm
    sigma = proj_sigma(safe, vuln, sv_unit)
    return {"steering_vector": sv_unit * sigma if norm == "sigma" else raw,
            "sv_unit": sv_unit, "safe_mean": safe_mean, "vuln_mean": vuln_mean,
            "raw_norm": raw_norm, "proj_sigma": sigma, "norm": norm, "method": "mean_diff"}


def probe_vector(w_safe, safe, vuln, norm):
    raw_norm = w_safe.norm().item()
    sv_unit = w_safe / raw_norm
    sigma = proj_sigma(safe, vuln, sv_unit)
    return {"steering_vector": sv_unit * sigma if norm == "sigma" else w_safe,
            "sv_unit": sv_unit, "raw_norm": raw_norm, "proj_sigma": sigma,
            "norm": norm, "method": "probe"}


def load_rep(path, idx):
    d = torch.load(path, map_location="cpu", weights_only=True)
    return d["safe"][idx].float(), d["vuln"][idx].float()


def load_probe_w(path):
    """-w of a probe checkpoint (points toward safe)."""
    return -torch.load(path, map_location="cpu", weights_only=True)["linear.weight"][0].float()


def main(args):
    rep_dir = Path(args.rep_dir)
    meta = json.loads((rep_dir / "metadata.json").read_text())
    train_idx, val_idx = make_split_indices(meta, args.val_ratio, args.seed)
    idx = torch.tensor({"train": train_idx, "val": val_idx,
                        "all": sorted(train_idx + val_idx)}[args.split])
    ckpt_dir = rep_dir / "probes/checkpoints"
    out_root = Path(args.output_dir or rep_dir / "vectors")
    methods = ["mean_diff", "probe"] if args.method == "both" else [args.method]
    modes = ["layer", "head"] if args.mode == "both" else [args.mode]
    print(f"{rep_dir}\n  split={args.split} ({len(idx)} pairs) methods={methods} "
          f"modes={modes} norm={args.norm}")

    L, H = meta["n_layers"], meta["n_heads"]
    for mode in modes:
        units = [(l, None) for l in range(1, L + 1)] if mode == "layer" else \
                [(l, h) for l in range(1, L + 1) for h in range(1, H + 1)]
        for method in methods:
            out_dir = out_root / mode / method
            out_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for l, h in units:
                stem = f"layer_{l:02d}" if h is None else f"head_layer_{l:02d}_head_{h:02d}"
                safe, vuln = load_rep(rep_dir / f"{stem}.pt", idx)
                if method == "mean_diff":
                    vec = mean_diff_vector(safe, vuln, args.norm)
                else:
                    ckpt = ckpt_dir / f"{stem}_best.pt"
                    if not ckpt.exists():
                        raise SystemExit(f"missing probe checkpoint {ckpt}; run train_probe.py --mode {mode}")
                    vec = probe_vector(load_probe_w(ckpt), safe, vuln, args.norm)
                fname = (f"steering_vector_layer_{l:02d}.pt" if h is None
                         else f"steering_vector_head_{l:02d}_{h:02d}.pt")
                torch.save(vec, out_dir / fname)
                rows.append({"layer_idx": l, **({} if h is None else {"head_idx": h}),
                             "file": fname, "dim": vec["steering_vector"].shape[0],
                             "raw_norm": vec["raw_norm"], "proj_sigma": vec["proj_sigma"]})
            (out_dir / "metadata.json").write_text(json.dumps({
                "model": meta["model"], "cwe_id": meta["cwe_id"], "rep_dir": str(rep_dir),
                "split": args.split, "n_pairs": len(idx), "mode": mode, "method": method,
                "norm": args.norm, "vectors": rows}, indent=2))
            print(f"  {mode}/{method}: {len(rows)} vectors -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep_dir", required=True, help="output dir of extract_representations.py")
    ap.add_argument("--output_dir", default=None, help="default: <rep_dir>/vectors")
    ap.add_argument("--mode", choices=["layer", "head", "both"], default="both")
    ap.add_argument("--method", choices=["mean_diff", "probe", "both"], default="both",
                    help="probe needs train_probe.py checkpoints")
    ap.add_argument("--norm", choices=["sigma", "raw"], default="sigma")
    ap.add_argument("--split", choices=["train", "val", "all"], default="train")
    ap.add_argument("--val_ratio", type=float, default=0.2, help="must match train_probe.py")
    ap.add_argument("--seed", type=int, default=42, help="must match train_probe.py")
    main(ap.parse_args())
