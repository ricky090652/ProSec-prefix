"""Generate code under activation steering, in the format ICD scoring expects.

Port of DuoSteer steering/steer_eval.py. Settings (paper Table 2):
  baseline  no hooks
  layer     h_l  <- h_l  + alpha * v_l    output of decoder layer l    (LayerMD / LayerPD)
  head      z_lj <- z_lj + alpha * v_lj   pre-o_proj slice, top-k heads (ProbeMD / CausalMD)
Head ranking from --head_results, as in the original: probe json -> val_accuracy desc;
causal json -> delta_vs_baseline asc (most safe-promoting first). As in the original,
alpha * v is added at every position, prompt included.

Differences from the original:
  1. Prompts and decoding follow eval/gen_for_icd.py, so outputs feed
     normalize_responses.py -> detect_all.py -> score_detected.py unchanged:
     num_gen samples, temperature 0.8, top_p 0.95, bf16, per-prompt seed
     seed*100003 + i with i the index in the full prompt list. The baseline reproduces
     the shared OFF run, so steered and OFF samples are paired.
     (Original: greedy by default; paper Appendix H.2 shows DuoSteer holds under sampling.)
  2. Prompt sources: --instruct_json (PurpleLlama eval set) or --pairs_file
     (held-out D_sec, one entry per prompt) for config selection.
  3. head_suppress is not ported (not part of the paper's Table 2).
  4. File names include the head ranking source, so ProbeMD and CausalMD share a dir.

Usage:
  python steering/steer_generate.py --instruct_json $IJ --safecoder_only --cwe 502 \
      --setting head --head_results $REP/causal/head_causal_results.json \
      --vector_dir $REP/vectors --top_k_list 16 32 --alpha_list 1 3 --out_dir outputs/steer/cwe-502
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hooks import head_hooks, layer_hooks, rank_heads  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from gen_for_icd import SAFECODER_CWES, build_messages, cwe_num  # noqa: E402


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #

def load_prompts(args):
    """[(i, {"lang", "cwe", "prompt"})]; i is the 1-based index in the full list (seed key)."""
    if args.instruct_json:
        langs = {s.strip() for s in args.langs.split(",") if s.strip()}
        data = [x for x in json.load(open(args.instruct_json)) if x.get("language") in langs]
        if args.safecoder_only:
            data = [x for x in data if cwe_num(x.get("cwe_identifier")) in SAFECODER_CWES]
        rows = [{"lang": x["language"], "cwe": x.get("cwe_identifier", ""),
                 "prompt": x["test_case_prompt"]} for x in data]
    else:
        seen = {}
        for line in open(args.pairs_file):
            p = json.loads(line)
            seen.setdefault(p["src_id"], {"lang": p["lang"], "cwe": f"CWE-{p['cwe_id']}",
                                          "prompt": p["prompt"]})
        rows = list(seen.values())
    if args.limit:
        rows = rows[:args.limit]
    items = list(enumerate(rows, 1))
    if args.cwe:
        items = [(i, x) for i, x in items if cwe_num(x["cwe"]) == cwe_num(args.cwe)]
    return items


# --------------------------------------------------------------------------- #
# Generation (same procedure as eval/gen_for_icd.py)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def gen(model, tokenizer, prompt, seed, args, device):
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    text = tokenizer.apply_chat_template(build_messages(args.system_prompt, prompt),
                                         tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, return_tensors="pt").to(device)
    n_in = enc.input_ids.shape[1]
    out = model.generate(**enc, do_sample=True, num_return_sequences=args.num_gen,
                         temperature=args.temperature, top_p=args.top_p,
                         max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id)
    texts, trunc = [], []
    for o in out:
        new = o[n_in:]
        texts.append(tokenizer.decode(new, skip_special_tokens=True))
        trunc.append(bool((new != tokenizer.eos_token_id).all().item()) and len(new) >= args.max_new_tokens)
    return texts, trunc


def run_one(name, info, hooks, items, model, tokenizer, args, device, out_dir):
    path = out_dir / f"{name}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["i"] for l in open(path) if l.strip()}
    todo = [(i, x) for i, x in items if i not in done]
    (out_dir / f"{name}.info.json").write_text(json.dumps(info, indent=2))
    print(f"\n== {name}: {len(todo)} prompts to go ({len(done)} done)")
    try:
        with open(path, "a") as f:
            for n, (i, x) in enumerate(todo, 1):
                seed = None if args.seed is None else args.seed * 100003 + i
                resp, trunc = gen(model, tokenizer, x["prompt"], seed, args, device)
                f.write(json.dumps({**x, "responses": resp, "truncated": trunc,
                                    "max_new_tokens": args.max_new_tokens, "i": i},
                                   ensure_ascii=False) + "\n")
                f.flush()
                if n % 10 == 0:
                    print(f"  {n}/{len(todo)}")
    finally:
        for h in hooks:
            h.remove()
    print(f"  -> {path}")


def main(args):
    if args.seed is not None and args.seed < 0:
        args.seed = None
    items = load_prompts(args)
    if not items:
        raise SystemExit("no prompts after filtering")
    print(f"{len(items)} prompts (cwe={args.cwe or 'all'})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map={"": 0} if device == "cuda" else None).eval()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    common = {"num_gen": args.num_gen, "temperature": args.temperature, "top_p": args.top_p,
              "seed": args.seed, "max_new_tokens": args.max_new_tokens, "cwe": args.cwe,
              "source": args.instruct_json or args.pairs_file}

    for setting in args.setting:
        if setting == "baseline":
            run_one("baseline", {"setting": "baseline", **common}, [],
                    items, model, tokenizer, args, device, out_dir)

        elif setting == "layer":
            for l in args.layer_list:
                for a in args.alpha_list:
                    hooks = layer_hooks(model, args.vector_dir, args.method, l, a)
                    run_one(f"layer_L{l:02d}_{args.method}_a{a:g}",
                            {"setting": "layer", "layer": l, "alpha": a, "method": args.method, **common},
                            hooks, items, model, tokenizer, args, device, out_dir)

        elif setting == "head":
            ranked, rtype = rank_heads(args.head_results)
            for k in args.top_k_list:
                targets = ranked[:k]
                for a in args.alpha_list:
                    hooks = head_hooks(model, args.vector_dir, args.method, targets, a)
                    run_one(f"head_{rtype}_top{k}_{args.method}_a{a:g}",
                            {"setting": "head", "ranking": rtype, "top_k": k, "alpha": a,
                             "method": args.method, "heads": [f"L{t['layer']:02d}H{t['head']:02d}" for t in targets],
                             **common},
                            hooks, items, model, tokenizer, args, device, out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--instruct_json", help="PurpleLlama datasets/instruct/instruct.json")
    src.add_argument("--pairs_file", help="steering/build_pairs.py output, e.g. heldout.jsonl")
    ap.add_argument("--langs", default="c,cpp,java,javascript,python")
    ap.add_argument("--safecoder_only", action="store_true", help="the 693-prompt subset")
    ap.add_argument("--cwe", default=None, help="keep one CWE, e.g. 502")
    ap.add_argument("--limit", type=int, default=None, help="first N prompts before the CWE filter")
    ap.add_argument("--setting", nargs="+", choices=["baseline", "layer", "head"], required=True)
    ap.add_argument("--vector_dir", default=None, help="extract_steering_vector.py output")
    ap.add_argument("--method", choices=["mean_diff", "probe"], default="mean_diff")
    ap.add_argument("--head_results", default=None,
                    help="head_accuracy_results.json (ProbeMD) or head_causal_results.json (CausalMD)")
    ap.add_argument("--layer_list", type=int, nargs="+", default=[])
    ap.add_argument("--top_k_list", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--alpha_list", type=float, nargs="+", default=[1, 2, 3, 5, 10])
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--num_gen", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42, help="-1 for unseeded")
    ap.add_argument("--system_prompt", default=None, help="mainline DPO arms use none")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    if "layer" in args.setting and not (args.layer_list and args.vector_dir):
        ap.error("layer setting needs --layer_list and --vector_dir")
    if "head" in args.setting and not (args.head_results and args.vector_dir):
        ap.error("head setting needs --head_results and --vector_dir")
    main(args)
