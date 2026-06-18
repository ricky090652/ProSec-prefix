"""HumanEval-Multi (multilingual) pass@1 via MultiPL-E, prefix ON vs OFF.

IMPORTANT: our prefix is trained with the Phi-3 chat template, so it only works in
INSTRUCT mode. Raw completion (MultiPL-E's default) puts the prefix out-of-distribution
(observed: ON pass@1 = 0% while OFF = 70%). We therefore generate in instruct mode
(ask the model to implement the function, extract the code block) and assemble the
program with MultiPL-E's tests per language. This matches how the prefix is actually
used and how our Python HumanEval was run.

Python is the original HumanEval (run_humaneval.py); MultiPL-E has no humaneval-py.

Toolchains (install prebuilt to $HOME, no sudo): js -> node, cpp -> g++.
(java needs JDK + javatuples jar + class handling; best-effort, off by default.)

Usage (validate with js first):
  python eval/run_multipl_e.py --adapter ./outputs/phi3-prefix-dpo-full --langs js --limit 20
  python eval/run_multipl_e.py --adapter ./outputs/phi3-prefix-dpo-full --langs js,cpp
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
LANG_NAME = {"js": "JavaScript", "cpp": "C++", "java": "Java"}


def extract_code(text):
    m = re.search(r"```(?:[a-zA-Z+#]*)\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


def assemble(lang, code, tests):
    """Build a runnable program from instruct-generated full code + MultiPL-E tests."""
    if lang == "js":
        return code + "\n" + tests
    if lang == "cpp":
        mi = code.find("int main")          # drop any model-added main()
        if mi != -1:
            code = code[:mi]
        t = tests.lstrip()
        if t.startswith("}"):               # tests assume completion closes the fn
            t = t[1:]
        pre = "#include<assert.h>\n#include<bits/stdc++.h>\nusing namespace std;\n"
        return pre + code + "\n" + t
    if lang == "java":                      # best-effort
        t = tests.lstrip()
        if t.startswith("}"):
            t = t[1:]
        return code + "\n" + t
    raise ValueError(lang)


def _run(cmd, cwd, timeout):
    try:
        return subprocess.run(cmd, capture_output=True, cwd=cwd, timeout=timeout).returncode == 0
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
    ap.add_argument("--langs", default="js,cpp")
    ap.add_argument("--max_new_tokens", type=int, default=640)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--javatuples_jar", default=None)
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
    def gen_code(stub, lang):
        instr = (f"Complete the following {LANG_NAME[lang]} function. Return ONLY the "
                 f"complete function in a single ```{lang} code block, keeping the exact "
                 f"given signature and name.\n\n```{lang}\n{stub}\n```")
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": instr}], tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        out = model.generate(ids, do_sample=False, max_new_tokens=args.max_new_tokens,
                             pad_token_id=tokenizer.pad_token_id)
        return extract_code(tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True))

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
                    code = gen_code(ex["prompt"], lang)
                else:
                    with model.disable_adapter():
                        code = gen_code(ex["prompt"], lang)
                program = assemble(lang, code, ex["tests"])
                with tempfile.TemporaryDirectory() as wd:
                    if run_program(lang, program, wd, args.javatuples_jar):
                        passed += 1
            return passed

        on, off, n = eval_pass(True), eval_pass(False), len(ds)
        result[lang] = {"n": n,
                        "off_pass@1": round(100.0 * off / n, 2),
                        "on_pass@1": round(100.0 * on / n, 2),
                        "delta": round(100.0 * (on - off) / n, 2)}
        print(f"  {lang}: OFF {result[lang]['off_pass@1']}%  ON {result[lang]['on_pass@1']}%  "
              f"Δ {result[lang]['delta']:+}")

    json.dump(result, open(args.out, "w"), indent=2)
    print("\n==== MultiPL-E pass@1 (instruct mode) ====")
    print(f"{'lang':<8}{'OFF':>8}{'ON':>8}{'Δ':>8}")
    for lang, r in result.items():
        print(f"{lang:<8}{r['off_pass@1']:>7}%{r['on_pass@1']:>7}%{r['delta']:>+7}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
