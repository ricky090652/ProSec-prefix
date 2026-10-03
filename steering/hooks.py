"""Steering hooks shared by steering/steer_generate.py and the eval scripts.

SteerLayer / SteerHead are the ports of DuoSteer steering/steer_eval.py: alpha * v is
added at every position (prompt included). Each hook can be switched off without
being removed, so one model serves both the steered and the unsteered side.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import torch


class SteerLayer:
    """Adds alpha * v to the output hidden state of one decoder layer."""

    def __init__(self, layer, sv, alpha):
        self.offset = alpha * sv
        self.active = True
        self._handle = layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if not self.active:
            return None
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden + self.offset.to(hidden.device, hidden.dtype)
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    def remove(self):
        self._handle.remove()


class SteerHead:
    """Adds alpha * v to one head's slice of the o_proj input."""

    def __init__(self, o_proj, head_idx, head_dim, sv, alpha):
        self.slice = slice(head_idx * head_dim, (head_idx + 1) * head_dim)
        self.offset = alpha * sv
        self.active = True
        self._handle = o_proj.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        if not self.active:
            return None
        x = inputs[0].clone()
        x[..., self.slice] += self.offset.to(x.device, x.dtype)
        return (x,)

    def remove(self):
        self._handle.remove()


def rank_heads(path):
    """Probe json -> val_accuracy desc; causal json -> delta_vs_baseline asc (as in DuoSteer)."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and "heads" in data:
        return sorted(data["heads"], key=lambda r: r.get("delta_vs_baseline", r["mean_delta"])), "causal"
    return sorted(data, key=lambda r: r["val_accuracy"], reverse=True), "probe"


def load_sv(path):
    return torch.load(path, map_location="cpu", weights_only=True)["steering_vector"].float()


def decoder_layers(model):
    return [m for m in model.modules() if hasattr(m, "self_attn") and hasattr(m, "mlp")]


def o_projs(model):
    return [m.o_proj for m in model.modules() if hasattr(m, "o_proj")]


def layer_hooks(model, vector_dir, method, layer, alpha):
    sv = load_sv(Path(vector_dir) / "layer" / method / f"steering_vector_layer_{layer:02d}.pt")
    return [SteerLayer(decoder_layers(model)[layer - 1], sv, alpha)]


def head_hooks(model, vector_dir, method, targets, alpha):
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    ops = o_projs(model)
    return [SteerHead(ops[t["layer"] - 1], t["head"] - 1, head_dim,
                      load_sv(Path(vector_dir) / "head" / method /
                              f"steering_vector_head_{t['layer']:02d}_{t['head']:02d}.pt"), alpha)
            for t in targets]


# --------------------------------------------------------------------------- #
# CLI glue for eval/run_humaneval.py and eval/run_multipl_e.py
# --------------------------------------------------------------------------- #

def add_steer_args(ap):
    g = ap.add_argument_group("steering (ON = steered model, OFF = same model unsteered)")
    g.add_argument("--steer_setting", choices=["layer", "head"], default=None)
    g.add_argument("--steer_vector_dir", default=None, help="extract_steering_vector.py output")
    g.add_argument("--steer_method", choices=["mean_diff", "probe"], default="mean_diff")
    g.add_argument("--steer_head_results", default=None,
                   help="head_causal_results.json (CausalMD) or head_accuracy_results.json (ProbeMD)")
    g.add_argument("--steer_top_k", type=int, default=32)
    g.add_argument("--steer_layer", type=int, default=None)
    g.add_argument("--steer_alpha", type=float, default=None)


def attach_from_args(model, args):
    """Register hooks per the --steer_* args. Returns (hooks, label)."""
    if not args.steer_setting:
        return [], None
    if args.steer_vector_dir is None or args.steer_alpha is None:
        raise SystemExit("--steer_setting needs --steer_vector_dir and --steer_alpha")
    if args.steer_setting == "layer":
        if args.steer_layer is None:
            raise SystemExit("--steer_setting layer needs --steer_layer")
        hooks = layer_hooks(model, args.steer_vector_dir, args.steer_method, args.steer_layer, args.steer_alpha)
        label = f"layer_L{args.steer_layer:02d}_{args.steer_method}_a{args.steer_alpha:g}"
    else:
        if args.steer_head_results is None:
            raise SystemExit("--steer_setting head needs --steer_head_results")
        ranked, rtype = rank_heads(args.steer_head_results)
        hooks = head_hooks(model, args.steer_vector_dir, args.steer_method,
                           ranked[:args.steer_top_k], args.steer_alpha)
        label = f"head_{rtype}_top{args.steer_top_k}_{args.steer_method}_a{args.steer_alpha:g}"
    print(f"steering: {label} ({len(hooks)} hooks)")
    return hooks, label


@contextmanager
def hooks_off(hooks):
    for h in hooks:
        h.active = False
    try:
        yield
    finally:
        for h in hooks:
            h.active = True
