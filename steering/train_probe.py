"""Train linear probes on extracted representations.

Port of DuoSteer localization/train_probe.py. One nn.Linear(dim, 1) per layer or head,
BCEWithLogits + Adam (unregularized logistic regression), safe=0 / vuln=1, best val
checkpoint kept. plots/{layer,head}_accuracy_results.json keep the original format.

Differences from the original:
  1. Split is always by src_id (instruction); missing src_id is an error. The original
     falls back to pair id, which leaks prompts on DuoSteer's released pairs.
  2. The leakage check compares src_id (the original compares unique pair ids).
  3. --epochs defaults to 200 (paper Appendix B.1; original default 100).
  4. torch is seeded, so reruns give identical probes.

Usage:
  python steering/train_probe.py --rep_dir <extract output dir> --mode layer
  python steering/train_probe.py --rep_dir <extract output dir> --mode head
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam


class LinearProbe(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def make_split_indices(metadata, val_ratio, seed):
    """Shuffle and split at the src_id level; a prompt's pairs stay on one side."""
    groups = defaultdict(list)
    for p in metadata["pairs"]:
        if not p.get("src_id"):
            raise SystemExit("metadata has no src_id; a pair-level split would leak prompts")
        groups[p["src_id"]].append(p["index"])
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_val = max(1, round(len(keys) * val_ratio))
    val_idx = sorted(i for k in keys[:n_val] for i in groups[k])
    train_idx = sorted(i for k in keys[n_val:] for i in groups[k])
    return train_idx, val_idx


def build_tensors(pt_data, indices, device):
    """{"safe", "vuln"} -> X (2n, dim), y (2n,) with safe=0, vuln=1."""
    idx = torch.tensor(indices, dtype=torch.long)
    X = torch.cat([pt_data["safe"][idx].float(), pt_data["vuln"][idx].float()]).to(device)
    y = torch.cat([torch.zeros(len(idx)), torch.ones(len(idx))]).to(device)
    return X, y


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train_one_probe(X_train, y_train, X_val, y_val, epochs, lr, batch_size, device, ckpt_path):
    probe = LinearProbe(X_train.shape[1]).to(device)
    opt = Adam(probe.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()
    val_accs, best = [], 0.0
    for _ in range(epochs):
        probe.train()
        perm = torch.randperm(len(X_train), device=device)
        for s in range(0, len(X_train), batch_size):
            b = perm[s:s + batch_size]
            opt.zero_grad()
            crit(probe(X_train[b]), y_train[b]).backward()
            opt.step()
        probe.eval()
        with torch.no_grad():
            acc = ((probe(X_val) > 0) == y_val.bool()).float().mean().item()
        val_accs.append(acc)
        if acc > best:
            best = acc
            torch.save(probe.state_dict(), ckpt_path)
    return val_accs, best


# --------------------------------------------------------------------------- #
# Plots (same three figures as the original)
# --------------------------------------------------------------------------- #

def plot_val_curves(curves, out_path, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("plasma", len(curves))
    for i, (label, c) in enumerate(curves.items()):
        ax.plot(c, color=cmap(i), linewidth=0.9, alpha=0.8, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    if len(curves) <= 20:
        ax.legend(fontsize=7, ncol=2, loc="lower right")
    else:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, len(curves) - 1))
        plt.colorbar(sm, ax=ax, label="Layer index")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_best_per_layer(best_accs, labels, out_path):
    fig, ax = plt.subplots(figsize=(max(8, len(best_accs) * 0.35), 5))
    x = np.arange(len(best_accs))
    norm = plt.Normalize(max(0.0, min(best_accs) - 0.05), min(1.0, max(best_accs) + 0.05))
    cmap = plt.get_cmap("Blues")
    ax.bar(x, best_accs, color=[cmap(norm(a)) for a in best_accs], edgecolor="none")
    ax.plot(x, best_accs, "o-", color="#333333", linewidth=1.2, markersize=3, alpha=0.8)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="Chance (0.5)")
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, label="Best Validation Accuracy")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Best Validation Accuracy")
    ax.set_title("Probe accuracy across transformer layers")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45 if len(x) > 16 else 0)
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_head_heatmap(acc, out_path):
    """y = layer (layer 1 at bottom), x = rank within layer (0 = best head)."""
    n_layers, n_heads = acc.shape
    sorted_acc = -np.sort(-acc, axis=1)
    fig, ax = plt.subplots(figsize=(max(10, n_heads * 0.45), max(6, n_layers * 0.35)))
    im = ax.imshow(sorted_acc[::-1], cmap="Blues", aspect="auto",
                   vmin=max(0.4, sorted_acc.min() - 0.02),
                   vmax=min(1.0, sorted_acc.max() + 0.02))
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02).set_label("Best Validation Accuracy")
    ax.set_xlabel("Head rank within layer  (0 = best)")
    ax.set_ylabel("Layer")
    ax.set_title("Probe accuracy per layer × head\n(heads sorted high → low within each layer)")
    ax.set_xticks(np.arange(n_heads))
    ax.set_xticklabels([str(r) for r in range(n_heads)], fontsize=7)
    ax.set_yticks(np.arange(n_layers))
    ax.set_yticklabels([str(n_layers - l) for l in range(n_layers)], fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #

def run_layer_mode(rep_dir, train_idx, val_idx, args, device, ckpt_dir, plot_dir):
    files = sorted(rep_dir.glob("layer_*.pt"))
    if not files:
        raise SystemExit(f"no layer_*.pt in {rep_dir}")
    curves, best_accs = {}, []
    for f in files:
        l = int(f.stem.split("_")[1])
        data = torch.load(f, map_location="cpu")
        X_tr, y_tr = build_tensors(data, train_idx, device)
        X_va, y_va = build_tensors(data, val_idx, device)
        c, best = train_one_probe(X_tr, y_tr, X_va, y_va, args.epochs, args.lr,
                                  args.batch_size, device, ckpt_dir / f"layer_{l:02d}_best.pt")
        curves[f"layer {l:02d}"] = c
        best_accs.append((l, best))
        print(f"  layer {l:02d}  best_val_acc={best:.4f}")

    results = sorted([{"layer": l, "val_accuracy": round(a, 6)} for l, a in best_accs],
                     key=lambda r: r["val_accuracy"], reverse=True)
    with open(plot_dir / "layer_accuracy_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  best layer: {results[0]}")
    plot_val_curves(curves, plot_dir / "val_curves.png", "Validation accuracy over epochs (layer probes)")
    plot_best_per_layer([a for _, a in best_accs], [str(l) for l, _ in best_accs],
                        plot_dir / "best_per_layer.png")


def run_head_mode(rep_dir, train_idx, val_idx, metadata, args, device, ckpt_dir, plot_dir):
    n_layers, n_heads = metadata["n_layers"], metadata["n_heads"]
    acc = np.zeros((n_layers, n_heads))
    for l in range(n_layers):
        for h in range(n_heads):
            name = f"head_layer_{l + 1:02d}_head_{h + 1:02d}"
            data = torch.load(rep_dir / f"{name}.pt", map_location="cpu")
            X_tr, y_tr = build_tensors(data, train_idx, device)
            X_va, y_va = build_tensors(data, val_idx, device)
            _, acc[l, h] = train_one_probe(X_tr, y_tr, X_va, y_va, args.epochs, args.lr,
                                           args.batch_size, device, ckpt_dir / f"{name}_best.pt")
            print(f"  [{l * n_heads + h + 1}/{n_layers * n_heads}] L{l + 1:02d}H{h + 1:02d} "
                  f"best_val_acc={acc[l, h]:.4f}", end="\r")
    print()

    results = []
    for l in range(n_layers):
        rank = {int(h): r for r, h in enumerate(np.argsort(acc[l])[::-1])}
        for h in range(n_heads):
            results.append({"layer": l + 1, "head": h + 1,
                            "val_accuracy": round(float(acc[l, h]), 6), "rank_in_layer": rank[h]})
    results.sort(key=lambda r: r["val_accuracy"], reverse=True)
    with open(plot_dir / "head_accuracy_results.json", "w") as f:
        json.dump(results, f, indent=2)
    plot_head_heatmap(acc, plot_dir / "head_heatmap.png")
    b = results[0]
    print(f"  best head: L{b['layer']}H{b['head']} acc={b['val_accuracy']:.4f}; "
          f"mean={acc.mean():.4f}; heads>=0.65: {int((acc >= 0.65).sum())}")


def main(args):
    rep_dir = Path(args.rep_dir)
    with open(rep_dir / "metadata.json") as f:
        metadata = json.load(f)

    train_idx, val_idx = make_split_indices(metadata, args.val_ratio, args.seed)
    src = {p["index"]: p["src_id"] for p in metadata["pairs"]}
    train_src, val_src = {src[i] for i in train_idx}, {src[i] for i in val_idx}
    assert not (train_src & val_src), "prompt leakage between train and val"
    print(f"{rep_dir}\n  model={metadata['model']} cwe={metadata['cwe_id']} mode={args.mode}")
    print(f"  train {len(train_idx)} pairs / {len(train_src)} prompts; "
          f"val {len(val_idx)} pairs / {len(val_src)} prompts")

    out_dir = Path(args.output_dir) if args.output_dir else rep_dir / "probes"
    ckpt_dir, plot_dir = out_dir / "checkpoints", out_dir / "plots"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if args.mode == "layer":
        run_layer_mode(rep_dir, train_idx, val_idx, args, device, ckpt_dir, plot_dir)
    else:
        run_head_mode(rep_dir, train_idx, val_idx, metadata, args, device, ckpt_dir, plot_dir)
    print(f"-> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep_dir", required=True, help="output dir of extract_representations.py")
    ap.add_argument("--mode", choices=["layer", "head"], required=True)
    ap.add_argument("--output_dir", default=None, help="default: <rep_dir>/probes")
    ap.add_argument("--epochs", type=int, default=200, help="paper Appendix B.1")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(ap.parse_args())
