"""HumanEval-Multi (multilingual) pass@1 via MultiPL-E, prefix ON vs OFF.

Matches the paper's multilingual utility eval for the non-Python languages.
Python is the original HumanEval (run_humaneval.py); MultiPL-E has no humaneval-py.

Protocol (standard MultiPL-E, completion-style):
  program = prompt + completion(truncated at stop_tokens) + tests  -> compile & run
  exit code 0 = pass.

Toolchains required (install via conda if no sudo):
  js   -> node
  cpp  -> g++
  java -> javac/java + org.javatuples jar (pass via --javatuples_jar)

Usage (start with js to validate the harness):
  python eval/run_multipl_e.py --adapter ./outputs/phi3-prefix-dpo-full --langs js
  python eval/run_multipl_e.py --adapter ./outputs/phi3-prefix-dpo-full \
      --langs js,cpp,java --javatuples_jar /path/to/javatuples-1.2.jar
"""
import argparse
import json
import os
import re
import subprocess
import tempfile

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

LANG_CFG = {"js": "humaneval-js", "cpp": "humaneval-cpp", "java": "humaneval-java"}


def truncate_at_stops(text, stops):
    cut = len(text)
    for s in stops:
        p = text.find(s)
        if p != -1:
            cut = min(cut, p)
    return text[:cut]


def _run(cmd, cwd, timeout):
    try:
        r = subprocess.run(cmd, capture_output=True, cwd=cwd, timeout=timeout)
        return r.returncode == 0
    except Exception:
        return False


def run_program(lang, program, workdir, javatuples_jar, timeout=20):
    if lang == "js":
        open(os.path.join(workdir, "prog.js"), "w").write(program)
        return _run(["node", "prog.js"], workdir, timeout)
    if lang == "cpp":
        open(os.path.join(workdir, "prog.cpp"), "w").write(program)
        if not _run(["g++", "-std=c++17", "-O0", "-o", "prog", "prog.cpp"], workdir, timeout):
            return False
        return _run(["./prog"], workdir, timeout)
    if lang == "java":
        m = re.search(r"(?:public\s+)?class\s+(\w+)", program)
        cls = m.group(1) if m else "Problem"
        open(os.path.join(workdir, cls + ".java"), "w").write(program)
        cp = "." + (os.pathsep + javatuples_jar if javatuples_jar else "")
        if not _run(["javac", "-cp", cp, cls + ".java"], workdir, timeout):
            return False
        return _run(["java", "-ea", "-cp", cp, cls], workdir, timeout)
    raise ValueError(lang)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--langs", default="js,cpp,java")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--javatuples_jar", default=None, help="org.javatuples jar (java only)")
    ap.add_argument("--out", default="./outputs/multipl_e_result.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"載入 base：{args.model} + prefix：{args.adapter}")
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map={"": 0} if device == "cuda" else None,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    @torch.no_grad()
    def complete(prompt, stops):
        # completion-style: feed the raw stub, continue it
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        out = model.generate(ids, do_sample=False, max_new_tokens=args.max_new_tokens,
                             pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        return truncate_at_stops(text, stops)

    result = {}
    for lang in [s.strip() for s in args.langs.split(",") if s.strip()]:
        ds = load_dataset("nuprl/MultiPL-E", LANG_CFG[lang], split="test")
        if args.limit:
            ds = ds.select(range(args.limit))
        print(f"\n=== {lang} ({LANG_CFG[lang]}), n={len(ds)} ===")

        def eval_pass(use_prefix):
            passed = 0
            for ex in ds:
                if use_prefix:
                    comp = complete(ex["prompt"], ex["stop_tokens"])
                else:
                    with model.disable_adapter():
                        comp = complete(ex["prompt"], ex["stop_tokens"])
                program = ex["prompt"] + comp + "\n" + ex["tests"]
                with tempfile.TemporaryDirectory() as wd:
                    if run_program(lang, program, wd, args.javatuples_jar):
                        passed += 1
            return passed

        on = eval_pass(True)
        off = eval_pass(False)
        n = len(ds)
        result[lang] = {
            "n": n,
            "off_pass@1": round(100.0 * off / n, 2),
            "on_pass@1": round(100.0 * on / n, 2),
            "delta": round(100.0 * (on - off) / n, 2),
        }
        print(f"  {lang}: OFF {result[lang]['off_pass@1']}%  ON {result[lang]['on_pass@1']}%  "
              f"Δ {result[lang]['delta']:+}")

    json.dump(result, open(args.out, "w"), indent=2)
    print("\n==== MultiPL-E pass@1 ====")
    print(f"{'lang':<8}{'OFF':>8}{'ON':>8}{'Δ':>8}")
    for lang, r in result.items():
        print(f"{lang:<8}{r['off_pass@1']:>7}%{r['on_pass@1']:>7}%{r['delta']:>+7}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
