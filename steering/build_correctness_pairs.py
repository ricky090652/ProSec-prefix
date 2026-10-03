"""Correctness pairs for DuoSteer's second vector (paper §3.5), built on MBPP.

The paper pairs safe-and-correct with safe-but-incorrect code of the same task, so the
correctness direction carries no safety signal. Here:
  gen    Base model (no adapter, no steering, no security instruction) samples N
         solutions per MBPP task in python (MBPP full) and js / cpp (MultiPL-E mbpp-*);
         each sample is judged by the task's unit tests. MBPP is disjoint from the
         HumanEval / MultiPL-E humaneval-* functional evaluation.
  ICD    ProSec's detect_all.py on samples.jsonl (safety filter; the paper uses CodeQL).
  pair   Drop truncated / malformed-fence / duplicate / flagged / unscanned samples, then
         pair a passing and a failing sample of the same task (intra-prompt). A response
         without a code block is taken as code, as the eval scripts do; one that has ```
         but no well-formed block is dropped.

Output follows steering/build_pairs.py, so extract_representations.py, train_probe.py,
head_causal_analysis.py and extract_steering_vector.py run unchanged. safe_code holds
the passing side and vuln_code the failing side, so the vector points toward correct.

Differences from the paper:
  1. No safety steering during generation. The paper steers only to enlarge the pool of
     safe samples (§3.5); MBPP tasks are not security tasks, so nearly all are safe.
  2. Correctness from unit tests instead of the GPT-4.1 judge.
  3. One pooled vector over three languages instead of one per CWE.

Prompts are the eval instructions (eval/run_humaneval.py, eval/run_multipl_e.py);
sampling follows eval/gen_for_icd.py (temperature 0.8, top-p 0.95, per-task seed).
src_id is the MBPP task id, shared across languages, so all languages of a task land on
the same side of the probe / knockout split.

Usage:
  # pilot: 50 random tasks per language, then look at the mixed pass/fail rate
  python steering/build_correctness_pairs.py gen --limit 50 --out_dir data/correctness_pilot
  python steering/build_correctness_pairs.py stats --samples data/correctness_pilot/samples.jsonl

  # full run -> ICD -> pairs
  python steering/build_correctness_pairs.py gen --out_dir data/correctness
  (cd $ICD && python prosec_scripts/detect_all.py --fin $REPO/data/correctness/samples.jsonl)
  python steering/build_correctness_pairs.py pair --samples data/correctness/samples.jsonl \
      --detected data/correctness/samples.jsonl.detected.jsonl

Runs model-generated code in subprocesses, as the eval scripts do.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
from run_multipl_e import assemble, build_messages, run_program  # noqa: E402

# key -> (lang name used by detect_all.py and the pairs file, fence tag, display name)
LANGS = {
    "python": ("python", "python", "Python"),
    "js": ("javascript", "js", "JavaScript"),
    "cpp": ("cpp", "cpp", "C++"),
}
FENCE_RE = re.compile(r"```(?:[a-zA-Z+#]*)\s*\n(.*?)```", re.DOTALL)  # as eval/run_multipl_e.py


# --------------------------------------------------------------------------- #
# Tasks and prompts
# --------------------------------------------------------------------------- #

def python_stub(ex):
    """HumanEval-style stub (signature + docstring) for the function the tests call."""
    code = ex["code"].replace("\r\n", "\n").replace("\t", "    ")
    for name, params in re.findall(r"^def\s+(\w+)\s*\(([^)]*)\)", code, re.M):
        if re.search(rf"\b{re.escape(name)}\s*\(", ex["test_list"][0]):
            text = ex["text"].strip().replace('"""', "'''")
            return f'def {name}({params}):\n    """{text}"""\n'
    return None


def load_tasks(lang, limit, seed):
    from datasets import load_dataset
    tasks = []
    if lang == "python":
        ds = load_dataset("mbpp", "full")
        for split in ds:
            for ex in ds[split]:
                stub = python_stub(ex)
                if stub is None:
                    continue
                tests = "\n".join([ex["test_setup_code"], *ex["test_list"]])
                tasks.append({"task_id": int(ex["task_id"]), "stub": stub, "tests": tests})
    else:
        for ex in load_dataset("nuprl/MultiPL-E", f"mbpp-{lang}", split="test"):
            tasks.append({"task_id": int(ex["name"].split("_")[1]),
                          "stub": ex["prompt"], "tests": ex["tests"]})
    tasks.sort(key=lambda t: t["task_id"])
    if limit and limit < len(tasks):
        tasks = sorted(random.Random(seed).sample(tasks, limit), key=lambda t: t["task_id"])
    return tasks


def instruction(lang, stub):
    if lang == "python":  # eval/run_humaneval.py
        return ("Complete the following Python function. "
                "Return the complete function in a single ```python code block.\n\n"
                f"```python\n{stub}\n```")
    name = LANGS[lang][2]  # eval/run_multipl_e.py
    return (f"Complete the following {name} function. Return ONLY the "
            f"complete function in a single ```{lang} code block, keeping the exact "
            f"given signature and name.\n\n```{lang}\n{stub}\n```")


# --------------------------------------------------------------------------- #
# Generation and tests
# --------------------------------------------------------------------------- #

def generate(model, tokenizer, prompt, seed, args):
    """Same sampling as steering/steer_generate.py and eval/gen_for_icd.py."""
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    text = tokenizer.apply_chat_template(build_messages(None, prompt),
                                         tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, return_tensors="pt").to(model.device)
    n_in = enc.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(**enc, do_sample=True, num_return_sequences=args.num_gen,
                             temperature=args.temperature, top_p=args.top_p,
                             max_new_tokens=args.max_new_tokens,
                             pad_token_id=tokenizer.pad_token_id)
    texts, trunc = [], []
    for o in out:
        new = o[n_in:]
        texts.append(tokenizer.decode(new, skip_special_tokens=True))
        trunc.append(bool((new != tokenizer.eos_token_id).all().item())
                     and len(new) >= args.max_new_tokens)
    return texts, trunc


def run_python(code, tests, workdir, timeout):
    path = os.path.join(workdir, "prog.py")
    with open(path, "w") as f:
        f.write(code + "\n\n" + tests + "\n")
    try:
        return subprocess.run([sys.executable, path], capture_output=True, cwd=workdir,
                              timeout=timeout).returncode == 0
    except Exception:
        return False


def run_tests(lang, codes, tests, timeout):
    def one(code):
        with tempfile.TemporaryDirectory() as wd:
            if lang == "python":
                return run_python(code, tests, wd, timeout)
            return run_program(lang, assemble(lang, code, tests), wd, None, timeout)
    with ThreadPoolExecutor(len(codes)) as pool:
        return list(pool.map(one, codes))


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def cmd_gen(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "samples.jsonl"
    done = {(r["lang"], r["task_id"]) for r in read_jsonl(path)} if path.exists() else set()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else None).eval()

    with open(path, "a") as f:
        for lang in args.langs.split(","):
            name, fence, _ = LANGS[lang]
            tasks = load_tasks(lang, args.limit, args.seed)
            todo = [t for t in tasks if (name, t["task_id"]) not in done]
            print(f"\n== {lang}: {len(todo)} tasks to go ({len(tasks) - len(todo)} done)")
            for n, t in enumerate(todo, 1):
                prompt = instruction(lang, t["stub"])
                raw, trunc = generate(model, tokenizer, prompt,
                                      args.seed * 100003 + t["task_id"], args)
                matches = [FENCE_RE.search(x) for x in raw]
                codes = [m.group(1) if m else x for m, x in zip(matches, raw)]
                passed = run_tests(lang, codes, t["tests"], args.test_timeout)
                f.write(json.dumps({
                    "lang": name, "cwe": "", "task_id": t["task_id"],
                    "src_id": f"mbpp-{t['task_id']}", "prompt": prompt,
                    # detect_all.py scans the fenced blocks of "responses"
                    "responses": [f"```{fence}\n{c}\n```" for c in codes],
                    "codes": codes, "passed": passed, "truncated": trunc,
                    "fenced": [m is not None for m in matches], "raw": raw,
                }, ensure_ascii=False) + "\n")
                f.flush()
                if n % 10 == 0:
                    print(f"  {n}/{len(todo)}")
    print(f"-> {path}")
    summarize(read_jsonl(path))


# --------------------------------------------------------------------------- #
# Stats and pairing
# --------------------------------------------------------------------------- #

def malformed(raw, fenced):
    """Has ``` but no well-formed block: the code cannot be told apart from the prose."""
    return not fenced and "```" in raw


def summarize(rows):
    """Per language: pass rate and how many tasks have both passing and failing samples."""
    print(f"\n{'lang':<11}{'tasks':>6}{'samples':>9}{'pass%':>7}{'mixed':>7}{'mixed%':>8}"
          f"{'trunc':>7}{'nofence':>8}{'malformed':>10}")
    by = defaultdict(list)
    for r in rows:
        by[r["lang"]].append(r)
    for lang, rs in by.items():
        n = sum(len(r["passed"]) for r in rs)
        p = sum(sum(r["passed"]) for r in rs)
        mixed = sum(0 < sum(r["passed"]) < len(r["passed"]) for r in rs)
        tr = sum(sum(r["truncated"]) for r in rs)
        nf = sum(len(r["fenced"]) - sum(r["fenced"]) for r in rs)
        mf = sum(sum(malformed(x, fe) for x, fe in zip(r["raw"], r["fenced"])) for r in rs)
        print(f"{lang:<11}{len(rs):>6}{n:>9}{100 * p / n:>7.1f}{mixed:>7}"
              f"{100 * mixed / len(rs):>8.1f}{tr:>7}{nf:>8}{mf:>10}")


def load_detection(path):
    """(lang, prompt, stripped code) of every scanned block and of the flagged ones."""
    scanned, flagged = set(), set()
    for e in read_jsonl(path):
        key = (e["lang"], e["prompt"], e["code"].strip())
        scanned.add(key)
        if e["detection_results"]:
            flagged.add(key)
    return scanned, flagged


def cmd_stats(args):
    summarize(read_jsonl(args.samples))


def cmd_pair(args):
    if not args.detected and not args.no_detect:
        raise SystemExit("give --detected <samples.jsonl.detected.jsonl>, or --no_detect")
    rows = read_jsonl(args.samples)
    scanned, flagged = load_detection(args.detected) if args.detected else (None, None)
    rng = random.Random(args.seed)
    drop = Counter()
    by_lang = defaultdict(list)
    n_tasks = Counter()

    for r in rows:
        good, bad, seen = [], [], set()
        for code, ok, tr, fe, raw in zip(r["codes"], r["passed"], r["truncated"],
                                         r["fenced"], r["raw"]):
            c = code.strip()
            reason = ("truncated" if tr else "malformed" if malformed(raw, fe) else "empty" if not c
                      else "duplicate" if c in seen else None)
            if reason is None and scanned is not None:
                key = (r["lang"], r["prompt"], c)
                reason = "unscanned" if key not in scanned else "flagged" if key in flagged else None
            if reason:
                drop[reason] += 1
                continue
            seen.add(c)
            (good if ok else bad).append(code)
        if not good or not bad:
            continue
        n_tasks[r["lang"]] += 1
        rng.shuffle(good)
        rng.shuffle(bad)
        for k in range(min(args.max_pairs_per_task, max(len(good), len(bad)))):
            by_lang[r["lang"]].append({
                "src_id": r["src_id"], "cwe_id": "none", "lang": r["lang"],
                "task_id": r["task_id"], "prompt": r["prompt"],
                "safe_code": good[k % len(good)], "vuln_code": bad[k % len(bad)],
            })

    counts = {lang: len(v) for lang, v in by_lang.items()}
    target = args.per_lang or (None if args.no_balance else min(counts.values(), default=0))
    pairs = []
    for lang in sorted(by_lang):
        v = by_lang[lang]
        if target and len(v) > target:
            v = rng.sample(v, target)
        pairs.extend(sorted(v, key=lambda p: (p["task_id"], p["safe_code"], p["vuln_code"])))

    out = Path(args.out or Path(args.samples).with_name("pairs.jsonl"))
    with open(out, "w") as f:
        for i, p in enumerate(pairs):
            f.write(json.dumps({"id": f"corr-{i}", **p}, ensure_ascii=False) + "\n")

    print(f"dropped samples: {dict(drop)}")
    print(f"{'lang':<11}{'tasks':>6}{'pairs':>7}{'kept':>6}")
    kept = Counter(p["lang"] for p in pairs)
    for lang in sorted(by_lang):
        print(f"{lang:<11}{n_tasks[lang]:>6}{counts[lang]:>7}{kept[lang]:>6}")
    print(f"total {len(pairs)} pairs / {len({p['src_id'] for p in pairs})} tasks -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="sample solutions and run unit tests")
    g.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    g.add_argument("--langs", default="python,js,cpp")
    g.add_argument("--limit", type=int, default=None, help="random tasks per language (pilot)")
    g.add_argument("--num_gen", type=int, default=10)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top_p", type=float, default=0.95)
    g.add_argument("--max_new_tokens", type=int, default=2048)
    g.add_argument("--test_timeout", type=int, default=20)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--out_dir", default="data/correctness")
    g.set_defaults(func=cmd_gen)

    s = sub.add_parser("stats", help="pass rate and mixed-task rate of a samples file")
    s.add_argument("--samples", required=True)
    s.set_defaults(func=cmd_stats)

    p = sub.add_parser("pair", help="filter samples and build pass-vs-fail pairs")
    p.add_argument("--samples", required=True)
    p.add_argument("--detected", default=None, help="detect_all.py output for --samples")
    p.add_argument("--no_detect", action="store_true", help="skip the ICD filter (pilot only)")
    p.add_argument("--max_pairs_per_task", type=int, default=2)
    p.add_argument("--per_lang", type=int, default=None,
                   help="pairs per language; default = the smallest language")
    p.add_argument("--no_balance", action="store_true", help="keep every pair")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="default: pairs.jsonl next to --samples")
    p.set_defaults(func=cmd_pair)

    args = ap.parse_args()
    for lang in getattr(args, "langs", "python").split(","):
        if lang not in LANGS:
            raise SystemExit(f"unknown language {lang!r} (choices: {', '.join(LANGS)})")
    args.func(args)


if __name__ == "__main__":
    main()
