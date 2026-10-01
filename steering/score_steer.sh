#!/usr/bin/env bash
# Normalize + ICD-detect every steer_generate.py output in a directory, then summarize.
# Same steps as stage 2 of scripts/run_full_eval.sh. A file is redone when its
# generation is newer than its .norm.jsonl (e.g. a resumed run).
#
# Usage (from the repo root):
#   ICD=<ProSec/PurpleLlama> bash steering/score_steer.sh <out_dir> [shared OFF .detected.jsonl]
set -euo pipefail
: "${ICD:?export ICD=<ProSec/PurpleLlama path>}"
DIR="$(realpath "$1")"
OFF="${2:-}"
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"   # weggli

for f in "$DIR"/*.jsonl; do
  case "$f" in *.norm.jsonl|*.detected.jsonl) continue ;; esac
  n="${f%.jsonl}.norm.jsonl"
  if [ -s "$n" ] && [ "$n" -nt "$f" ]; then continue; fi
  echo "=== $(basename "$f")"
  rm -f "$n.detected.jsonl"
  python eval/normalize_responses.py --in_file "$f" --out "$n"
  ( cd "$ICD" && python prosec_scripts/detect_all.py --fin "$n" )
done

python steering/summarize_steer.py --dir "$DIR" ${OFF:+--off "$OFF"}
