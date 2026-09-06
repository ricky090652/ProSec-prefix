#!/usr/bin/env bash
# 論文對齊的全量評測：693 題 × 10 樣本 × 5 語言，四臂一次跑完。
#
# 與便宜篩選（100 題 × 5、只有 c/cpp）的差別是樣本量 13.9 倍，
# 標準誤從約 2.24 pt 降到約 0.6 pt —— 這才是能拿去跟論文 Table 1 並列的數字。
#
# **OFF 只生成一次。** OFF 是關掉 adapter 的 base model，與 adapter 無關；
# 實測四臂的 OFF 逐項相同（漏洞率都是 49.00、退化統計逐項一致），
# 而逐題 seed 只取決於 --seed 與題目順序，所以分開生成的配對關係完全相同。
# 這讓生成量從 4×13,860 降到 6,930 + 4×6,930，省下約 16 小時。
#
# 每個階段都會跳過已完成的輸出，可以中斷後重跑。
#
# 用法：
#   bash scripts/run_full_eval.sh              # 全部（gen → detect → score）
#   STAGE=detect bash scripts/run_full_eval.sh # 只跑靜態分析與計分
#   ARMS="lora prefix" bash scripts/run_full_eval.sh   # 只跑部分臂
set -euo pipefail

: "${REPO:?請先 export REPO=<本 repo 的絕對路徑>}"
: "${ICD:?請先 export ICD=<ProSec/PurpleLlama 的絕對路徑>}"
IJ="$ICD/CybersecurityBenchmarks/datasets/instruct/instruct.json"
[ -f "$IJ" ] || { echo "找不到 $IJ —— 檢查 \$ICD"; exit 1; }

ARMS="${ARMS:-lora lora_qkv prefix prefix_nvt8}"
LANGS="${LANGS:-c,cpp,java,javascript,python}"
NUM_GEN="${NUM_GEN:-10}"
MAX_NEW="${MAX_NEW:-2048}"
OUT="${OUT:-outputs}"
ADAPTER_ROOT="${ADAPTER_ROOT:-outputs/dpo-arms}"
STAGE="${STAGE:-all}"
SHARED="$OUT/full_shared"          # 共用的 OFF

cd "$REPO"
mkdir -p "$OUT"

# 已存在且非空就跳過，讓中斷後可以接著跑
done_file() { [ -s "$1" ]; }

# ── 階段 1：生成 ────────────────────────────────────────────────────────
if [ "$STAGE" = "all" ] || [ "$STAGE" = "gen" ]; then
  if done_file "$SHARED.off.jsonl"; then
    echo "=== OFF 已存在，跳過（要重生請刪 $SHARED.off.jsonl）==="
  else
    echo "=== 生成共用的 OFF（$LANGS，${NUM_GEN} samples）==="
    # adapter 只是載體，這一趟它是關掉的
    first_arm="${ARMS%% *}"
    python eval/gen_for_icd.py --adapter "$ADAPTER_ROOT/$first_arm" \
      --instruct_json "$IJ" --safecoder_only --langs "$LANGS" \
      --num_gen "$NUM_GEN" --max_new_tokens "$MAX_NEW" --sides off \
      --out_prefix "$SHARED" 2>&1 | tee "$OUT/full_shared_gen.log"
  fi

  for ARM in $ARMS; do
    if done_file "$OUT/full_$ARM.on.jsonl"; then
      echo "=== $ARM 的 ON 已存在，跳過 ==="
      continue
    fi
    echo "=== 生成 $ARM 的 ON  $(date +%H:%M:%S) ==="
    python eval/gen_for_icd.py --adapter "$ADAPTER_ROOT/$ARM" \
      --instruct_json "$IJ" --safecoder_only --langs "$LANGS" \
      --num_gen "$NUM_GEN" --max_new_tokens "$MAX_NEW" --sides on \
      --out_prefix "$OUT/full_$ARM" 2>&1 | tee "$OUT/full_${ARM}_gen.log"
  done
fi

# ── 階段 2：補 code fence + 靜態分析 ────────────────────────────────────
if [ "$STAGE" = "all" ] || [ "$STAGE" = "detect" ]; then
  echo "=== 補 code fence ==="
  norm() {   # $1=輸入 jsonl（不含 .jsonl）
    done_file "$1.norm.jsonl" && { echo "  已有 $1.norm.jsonl"; return; }
    python eval/normalize_responses.py --in_file "$1.jsonl" --out "$1.norm.jsonl"
  }
  norm "$SHARED.off"
  for ARM in $ARMS; do norm "$OUT/full_$ARM.on"; done

  echo "=== 靜態分析（在 \$ICD 底下執行）==="
  [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"   # weggli
  detect() {
    done_file "$1.detected.jsonl" && { echo "  已有 $(basename "$1").detected.jsonl"; return; }
    ( cd "$ICD" && python prosec_scripts/detect_all.py --fin "$1" )
  }
  detect "$REPO/$SHARED.off.norm.jsonl"
  for ARM in $ARMS; do detect "$REPO/$OUT/full_$ARM.on.norm.jsonl"; done
fi

# ── 階段 3：計分 + 退化守門 ─────────────────────────────────────────────
if [ "$STAGE" = "all" ] || [ "$STAGE" = "score" ]; then
  echo
  for ARM in $ARMS; do
    echo "############################## $ARM ##############################"
    python eval/score_detected.py \
      --on  "$OUT/full_$ARM.on.norm.jsonl.detected.jsonl" \
      --off "$SHARED.off.norm.jsonl.detected.jsonl"
    python eval/check_degeneration.py \
      --on "$OUT/full_$ARM.on.jsonl" --off "$SHARED.off.jsonl" --per_lang
  done 2>&1 | tee "$OUT/security_full.txt"
  echo
  echo "完整結果 → $OUT/security_full.txt"
  echo "看【依語言】表，對 5 語言取平均，才是與論文 Table 1 對應的數字。"
fi
