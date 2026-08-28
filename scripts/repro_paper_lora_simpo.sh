#!/usr/bin/env bash
# ⚠️ 2026-08-28：這支腳本已**不是主線**，保留為 SimPO 失敗的可重現紀錄。
#    照論文原始設定跑出來的結果是 step ~350 起平滑失控、chosen 從 -0.45 掉到 -19，
#    模型完全毀掉（EXPERIMENTS.md S2-repro）。主線已改用 DPO：scripts/run_dpo_arms.sh。
#    要重現那次崩塌、或將來要測 masked SimPO（S5）時才用這支。
#
# ProSec 論文原始設定的復現：Phi-3-mini + LoRA + SimPO + §3.3 influence selection。
#
# 超參數全部來自論文，不是我們調出來的：
#   Appendix C Table 6  SimPO: lr=5e-6, beta=1.5, gamma=0.5, steps=1500 (Phi-3-mini)
#   Appendix C          LoRA r=8, alpha=16；total batch size 64
#   §5 / §6             warm-up: 在 D_sec 上跑 1000 steps，每 100 步存 checkpoint
#   §5                  D_norm 選擇：每個 D_sec 取 top-2，再丟掉全域最低 20%
#
# 論文沒寫、我們必須自己決定的部分列在 docs/REPRO_PAPER.md，改動時請同步更新。
#
# 用法：
#   bash scripts/repro_paper_lora_simpo.sh            # 跑完整四階段
#   STAGE=3 bash scripts/repro_paper_lora_simpo.sh    # 只從第 3 階段開始
#   STAGE=4 FINAL_TRAIN_FILE=data/train_pref.jsonl bash scripts/repro_paper_lora_simpo.sh
#                                                     # 跑法 A：跳過 selection，直接訓練
set -euo pipefail

MODEL=${MODEL:-microsoft/Phi-3-mini-4k-instruct}
TRAIN_FILE=${TRAIN_FILE:-data/train_pref.jsonl}
OUT=${OUT:-outputs/repro-paper}
STAGE=${STAGE:-1}
# 階段 4 吃的資料。預設是階段 3 篩出來的；跑法 A（見 docs/REPRO_PAPER.md §4）
# 直接指到原始的 train_pref.jsonl，跳過階段 1–3。
FINAL_TRAIN_FILE=${FINAL_TRAIN_FILE:-$OUT/train_pref_selected.jsonl}

# --- 論文超參數（不要在這裡調；要調就是另一個實驗，不叫復現） ---
LR=5e-6
BETA=1.5
GAMMA=0.5
LORA_R=8
LORA_ALPHA=16
MAIN_STEPS=1500
WARMUP_STEPS=1000
SAVE_STEPS=100
# ProSec 程式庫裡有三個互不相同的 system prompt（產生資料的、influence score 的、
# 訓練用的未公開），無從判斷論文訓練/評測用哪一個，且只給一臂加會破壞 prefix vs LoRA
# 的對照與既有的 base OFF 基線。預設不用；要重現最初那次崩塌才設。
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"

# --- 硬體相關：只要乘積 = 64（論文的 total batch size）就等價 ---
BATCH=${BATCH:-4}
ACCUM=${ACCUM:-16}
MAX_LENGTH=${MAX_LENGTH:-2048}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}

mkdir -p "$OUT"
echo "effective batch = $((BATCH * ACCUM))（論文 64）"
[ $((BATCH * ACCUM)) -eq 64 ] || echo "⚠️  不等於 64，這不是論文設定"

common_args=(
  --model "$MODEL"
  --peft_method lora --lora_r $LORA_R --lora_alpha $LORA_ALPHA
  --objective simpo --beta $BETA --gamma $GAMMA --cpo_alpha 0
  --lr $LR
  --batch_size "$BATCH" --grad_accum "$ACCUM"
  --max_length "$MAX_LENGTH" --max_prompt_length "$MAX_PROMPT_LENGTH"
  --max_grad_norm 1.0            # LLaMA-Factory 預設；不是我們主線的 0.3
  --lr_scheduler_type cosine --warmup_ratio 0.1   # SimPO 官方 recipe
  --bf16 --gradient_checkpointing
)
[ -n "$SYSTEM_PROMPT" ] && common_args+=(--system_prompt "$SYSTEM_PROMPT")

# ── 階段 1：暖身訓練（只用 D_sec，1000 步，每 100 步存檔）────────────────
if [ "$STAGE" -le 1 ]; then
  echo "=== [1/4] warm-up on D_sec：$WARMUP_STEPS steps ==="
  # 一定要 tee：train_prefix.py 設 report_to=[]，訓練指標只印到 stdout，
  # 不存檔的話事後沒東西可以餵給 eval/compare_lr_sweep.py
  python train_prefix.py "${common_args[@]}" \
    --train_file "$TRAIN_FILE" --subset dsec \
    --output_dir "$OUT/warmup-dsec" \
    --max_steps $WARMUP_STEPS --save_steps $SAVE_STEPS --save_total_limit 20 \
    2>&1 | tee "$OUT/warmup-dsec.log"
fi

# ── 階段 2：在 10 個 checkpoint 上量 training dynamics ────────────────────
if [ "$STAGE" -le 2 ]; then
  echo "=== [2/4] collect training dynamics ==="
  python data/collect_training_dynamics.py \
    --train_file "$TRAIN_FILE" --base_model "$MODEL" \
    --ckpt_root "$OUT/warmup-dsec" --out_dir "$OUT/dynamics" \
    --ckpt_every $SAVE_STEPS --sides both \
    --max_length "$MAX_LENGTH" --system_prompt "$SYSTEM_PROMPT" --bf16   # 空字串=不用
fi

# ── 階段 3：influence selection，產出最終訓練集 ──────────────────────────
if [ "$STAGE" -le 3 ]; then
  echo "=== [3/4] D_norm influence selection（top-2 + 丟最低 20%）==="
  python data/select_dnorm.py \
    --train_file "$TRAIN_FILE" --dynamics_dir "$OUT/dynamics" \
    --out "$OUT/train_pref_selected.jsonl" \
    --stats_out "$OUT/influence_scores.jsonl" \
    --corr_method kendall --f on --g neg_if \
    --top_n 2 --downsample_ratio 0.8
fi

# ── 階段 4：正式對齊訓練（1500 步）─────────────────────────────────────
if [ "$STAGE" -le 4 ]; then
  echo "=== [4/4] alignment training：$MAIN_STEPS steps ==="
  python train_prefix.py "${common_args[@]}" \
    --train_file "$FINAL_TRAIN_FILE" \
    --output_dir "$OUT/lora-simpo" \
    --max_steps $MAIN_STEPS --save_steps 300 --save_total_limit 6 \
    2>&1 | tee "$OUT/lora-simpo.log"
fi

cat <<EOF

完成。adapter 在 $OUT/lora-simpo，訓練 log 在 $OUT/lora-simpo.log。

--- 先判讀訓練健康度，再決定要不要花幾小時跑評測 ---

  python eval/compare_lr_sweep.py $OUT/lora-simpo.log

  看三件事（判準見下方註解與 EXPERIMENTS.md S0/S1）：
    1. margin 成長率為正 —— 絕對值無意義，SimPO 與 DPO 的 reward 尺度差一個量級
    2. margin 是「靠拉高 chosen」長出來的，不是「兩側一起崩」
    3. grad_norm 全程 < 50
  rewards/accuracies 低於 0.5 **不是**失敗訊號：這份資料起點就是 ~0.40
  （y_f 是在看過 y_v 後生成的，只給 x_v 時 y_v 天生機率較高）。

以下是完整評測流程（eval 腳本走 PeftModel.from_pretrained + disable_adapter，
LoRA 與 prefix 通用，不需要改）。ICD=\$ProSec/PurpleLlama、REPO=本 repo 的絕對路徑。

--- 安全評測（4 步，缺一不可）---

1) 生成 ON/OFF 兩份
  python eval/gen_for_icd.py --adapter $OUT/lora-simpo \\
      --instruct_json \$ICD/CybersecurityBenchmarks/datasets/instruct/instruct.json \\
      --safecoder_only --langs c,cpp,java,javascript,python --num_gen 10 \\
      --out_prefix outputs/icd_repro

2) 補 code fence —— **不做的話 ~84% 程式碼會被丟掉、漏洞率被低估約 6 倍**
  python eval/normalize_responses.py --in_file outputs/icd_repro.off.jsonl --out outputs/icd_repro.off.norm.jsonl
  python eval/normalize_responses.py --in_file outputs/icd_repro.on.jsonl  --out outputs/icd_repro.on.norm.jsonl

3) 靜態分析（ProSec 的程式，**必須在 PurpleLlama/ 目錄下跑**，它用相對路徑 import）
  cd \$ICD && source \$REPO/.venv/bin/activate && source ~/.cargo/env   # semgrep + weggli
  python prosec_scripts/detect_all.py --fin \$REPO/outputs/icd_repro.off.norm.jsonl
  python prosec_scripts/detect_all.py --fin \$REPO/outputs/icd_repro.on.norm.jsonl

4) 計分（看「依語言」表，對 5 語言取平均）
  cd \$REPO
  python eval/score_detected.py \\
      --on  outputs/icd_repro.on.norm.jsonl.detected.jsonl \\
      --off outputs/icd_repro.off.norm.jsonl.detected.jsonl | tee outputs/security_repro.txt

--- 退化守門（EXPERIMENTS.md 全域規則：每組 security 數字都要附）---

  python eval/check_degeneration.py --on outputs/icd_repro.on.jsonl \\
      --off outputs/icd_repro.off.jsonl --per_lang \\
      --json_out outputs/degeneration_repro.json
  # 三條件：ON 程式碼長度 >= OFF 的 90%、語法完整率 Δ >= -3pt、stub 率 Δ <= +2pt
  # 沒過就代表 security 數字被退化污染，不能單獨回報

--- 功能性評測 ---

  python eval/run_humaneval.py --adapter $OUT/lora-simpo \\
      --out outputs/humaneval_repro.json
  python eval/run_multipl_e.py --adapter $OUT/lora-simpo --langs js,cpp \\
      --out outputs/multipl_e_repro.json
EOF
