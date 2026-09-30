"""Extract Phi-3 layer / head representations from intra pairs.

Port of DuoSteer localization/extract_representations.py; output layout is the same.

Representations (paper §3.1), mean over response tokens:
  layer  h̄^(l)   = output_hidden_states[1:]
  head   z̄^(l,j) = input of o_proj at layer l, split into H slices

Differences from the original:
  1. Prompt is the ProSec instruction as-is (no DuoSteer template, no --prompt_mode).
  2. Tokenization matches this repo: prompt as in eval/gen_for_icd.py, response as in
     TRL tokenize_row. The response mask covers the code only, not the template's
     trailing <|end|><|endoftext|>.
  3. position_ids are passed under left padding. RoPE is relative, so this only
     removes numerical noise (9e-7 on a tiny Phi-3).
  4. --token_agg defaults to response_mean (paper setting).
  5. Aggregation is done per layer on device, then split into heads.
  6. --cwe_id all pools every CWE.

Output {output_dir}/{model_slug}/{pairs file stem}/{cwe_tag}/{token_agg}/
  layer_XX.pt               {"safe": (n, d),   "vuln": (n, d)}
  head_layer_XX_head_YY.pt  {"safe": (n, d_h), "vuln": (n, d_h)}
  metadata.json
Indices are 1-based.

Usage:
  python steering/extract_representations.py \
      --input_file data/steering/intra.jsonl --cwe_id cwe-502 --bf16
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def cwe_key(cwe):
    return str(cwe).lower().removeprefix("cwe-").lstrip("0") or "0"


def load_pairs(path, cwe_id, max_pairs):
    target = None if cwe_id == "all" else cwe_key(cwe_id)
    pairs = []
    with open(path) as f:
        for line in f:
            p = json.loads(line)
            if target is None or cwe_key(p["cwe_id"]) == target:
                pairs.append(p)
                if max_pairs and len(pairs) >= max_pairs:
                    break
    return pairs


# --------------------------------------------------------------------------- #
# Tokenization
# --------------------------------------------------------------------------- #

def build_messages(system_prompt, instruction):
    """Same as eval/gen_for_icd.py. Mainline DPO arms use no system prompt."""
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": instruction})
    return msgs


def tokenize_side(tokenizer, prompt, code, system_prompt):
    """Return (prompt_ids, resp_ids)."""
    text = tokenizer.apply_chat_template(
        build_messages(system_prompt, prompt), tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(text)["input_ids"]
    resp_ids = tokenizer(code, add_special_tokens=False)["input_ids"]
    return prompt_ids, resp_ids


def check_alignment(tokenizer, pair, system_prompt):
    """Print the prompt/response boundary and compare with full chat-template tokenization."""
    prompt_ids, resp_ids = tokenize_side(tokenizer, pair["prompt"], pair["safe_code"], system_prompt)
    ours = prompt_ids + resp_ids
    full = tokenizer.apply_chat_template(
        build_messages(system_prompt, pair["prompt"])
        + [{"role": "assistant", "content": pair["safe_code"]}], tokenize=True)
    same = full[:len(ours)] == ours
    print("  tokenization check (first pair, safe):")
    print(f"    prompt tail : {tokenizer.convert_ids_to_tokens(prompt_ids[-3:])}")
    print(f"    response    : {tokenizer.convert_ids_to_tokens(resp_ids[:6])}")
    print(f"    prompt={len(prompt_ids)} response={len(resp_ids)}")
    print(f"    matches chat template: {same}; "
          f"unmasked template tail: {tokenizer.convert_ids_to_tokens(full[len(ours):])}")


def prepare_batch(tokenizer, sides, system_prompt, device):
    """Left-pad a batch. Returns (encoding incl. position_ids, response_mask)."""
    seqs = [tokenize_side(tokenizer, s["prompt"], s["code"], system_prompt) for s in sides]
    max_len = max(len(p) + len(r) for p, r in seqs)
    pad_id = tokenizer.pad_token_id
    ids, attn, resp = [], [], []
    for p, r in seqs:
        if not r:
            raise ValueError("empty response")
        pad = max_len - len(p) - len(r)
        ids.append([pad_id] * pad + p + r)
        attn.append([0] * pad + [1] * (len(p) + len(r)))
        resp.append([False] * (pad + len(p)) + [True] * len(r))
    attn = torch.tensor(attn, device=device)
    return {
        "input_ids": torch.tensor(ids, device=device),
        "attention_mask": attn,
        "position_ids": (attn.cumsum(-1) - 1).clamp(min=0),
    }, torch.tensor(resp, device=device)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

class HeadOutputCapture:
    """Forward pre-hook on every o_proj; captures its (b, s, H*d_h) input.

    Phi3Attention reshapes attn_output as transpose(1, 2).reshape(b, s, H*d_h),
    so the last dim is head-major.
    """

    def __init__(self, model):
        self.captures = []
        self._hooks = [m.o_proj.register_forward_pre_hook(self._hook)
                       for m in model.modules() if hasattr(m, "o_proj")]

    def _hook(self, module, inputs):
        self.captures.append(inputs[0].detach())

    def clear(self):
        self.captures.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()


def aggregate(t, resp_mask, token_agg):
    """(b, s, d) -> (b, d) over response tokens, in float32."""
    if token_agg == "response_last":
        return t[:, -1].float()   # left padding: last position is the last response token
    m = resp_mask.unsqueeze(-1).float()
    return (t.float() * m).sum(1) / m.sum(1)


@torch.no_grad()
def extract_batch(model, tokenizer, sides, args, capture, n_heads, head_dim, device):
    enc, resp_mask = prepare_batch(tokenizer, sides, args.system_prompt, device)
    if capture is not None:
        capture.clear()
    out = model(**enc, output_hidden_states=True)
    res = {}
    if args.mode in ("layer", "both"):
        res["layer"] = torch.stack(
            [aggregate(hs, resp_mask, args.token_agg) for hs in out.hidden_states[1:]],
            dim=1).cpu()                                           # (b, L, d)
    if capture is not None:
        res["head"] = torch.stack(
            [aggregate(c, resp_mask, args.token_agg) for c in capture.captures],
            dim=1).view(len(sides), -1, n_heads, head_dim).cpu()   # (b, L, H, d_h)
    return res


def slugify(name):
    return re.sub(r"[^\w\-]", "_", name)


def main(args):
    input_file = Path(args.input_file)
    pairs = load_pairs(input_file, args.cwe_id, args.max_pairs)
    print(f"{input_file.name}: cwe={args.cwe_id}, {len(pairs)} pairs")
    if not pairs:
        return

    cwe_tag = "all" if args.cwe_id == "all" else f"cwe-{cwe_key(args.cwe_id)}"
    out_dir = Path(args.output_dir) / slugify(args.model) / input_file.stem / cwe_tag / args.token_agg
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    cfg = model.config
    n_layers, n_heads = cfg.num_hidden_layers, cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_heads
    capture = HeadOutputCapture(model) if args.mode in ("head", "both") else None
    if capture is not None and len(capture._hooks) != n_layers:
        raise SystemExit(f"found {len(capture._hooks)} o_proj modules, expected {n_layers}")
    print(f"model {args.model}: L={n_layers} H={n_heads} d_h={head_dim} dtype={dtype}")
    check_alignment(tokenizer, pairs[0], args.system_prompt)

    acc = {"safe": {"layer": [], "head": []}, "vuln": {"layer": [], "head": []}}
    n_batches = (len(pairs) + args.batch_size - 1) // args.batch_size
    for bi, i in enumerate(range(0, len(pairs), args.batch_size), 1):
        batch = pairs[i:i + args.batch_size]
        for side in ("safe", "vuln"):
            sides = [{"prompt": p["prompt"], "code": p[f"{side}_code"]} for p in batch]
            res = extract_batch(model, tokenizer, sides, args, capture, n_heads, head_dim, device)
            for k, v in res.items():
                acc[side][k].append(v.to(dtype))
        print(f"  batch {bi}/{n_batches}", end="\r")
    print()

    # .clone() so each file stores only its slice, not the whole buffer
    if acc["safe"]["layer"]:
        safe, vuln = (torch.cat(acc[s]["layer"]) for s in ("safe", "vuln"))
        for l in range(n_layers):
            torch.save({"safe": safe[:, l].clone(), "vuln": vuln[:, l].clone()},
                       out_dir / f"layer_{l + 1:02d}.pt")
        print(f"  layer: {n_layers} files, shape=({len(pairs)}, {cfg.hidden_size})")
    if acc["safe"]["head"]:
        safe, vuln = (torch.cat(acc[s]["head"]) for s in ("safe", "vuln"))
        for l in range(n_layers):
            for h in range(n_heads):
                torch.save({"safe": safe[:, l, h].clone(), "vuln": vuln[:, l, h].clone()},
                           out_dir / f"head_layer_{l + 1:02d}_head_{h + 1:02d}.pt")
        print(f"  head: {n_layers * n_heads} files, shape=({len(pairs)}, {head_dim})")

    metadata = {
        "model": args.model, "cwe_id": args.cwe_id, "input_file": str(input_file),
        "mode": args.mode, "token_agg": args.token_agg, "dtype": str(dtype),
        "system_prompt": args.system_prompt,
        "n_pairs": len(pairs), "n_layers": n_layers, "n_heads": n_heads,
        "head_dim": head_dim, "hidden_dim": cfg.hidden_size,
        "pairs": [{"index": i, "id": p["id"], "src_id": p["src_id"],
                   "cwe_id": p["cwe_id"], "lang": p["lang"]} for i, p in enumerate(pairs)],
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    if capture is not None:
        capture.remove()
    size = sum(f.stat().st_size for f in out_dir.glob("*.pt")) / 1e6
    print(f"-> {out_dir} ({size:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True, help="output of steering/build_pairs.py")
    ap.add_argument("--cwe_id", default="all", help="all, or one CWE e.g. cwe-502 / 502")
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--output_dir", default="data/representations")
    ap.add_argument("--mode", choices=["layer", "head", "both"], default="both")
    ap.add_argument("--token_agg", choices=["response_mean", "response_last"], default="response_mean")
    ap.add_argument("--system_prompt", default=None,
                    help="must match the adapters being compared; mainline DPO arms use none")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_pairs", type=int, default=None, help="first N pairs only (testing)")
    main(ap.parse_args())
