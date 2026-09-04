#!/usr/bin/env bash
# SVEN 尺度的 prefix vs LoRA 對照（P-sven）。
#
# 為什麼需要：先前跑的 nvt=16 / 64 **都在 SVEN 的設計範圍之外**。
# SVEN（He & Vechev 2023，本工作 prefix 的參考方法）的 n_prefix_token
# 隨模型大小縮放，見 sven/scripts/train.py:50-57：
#     codegen-350M → 5      codegen-2B → 8      codegen-6B → 12
# Phi-3-mini 3.8B 內插約 9~10，而我們跑的 nvt=64 是 SVEN 上限（12）的 5.3 倍。
#
# 而我們量到的趨勢是「utility 損害隨 nvt 上升」（nvt=16 −10.98 → nvt=64 −32.93），
# 所以 SVEN 尺度很可能溫和得多。不測這一段，就等於沒測 SVEN 的方法。
#
# 參數量又剛好可以精確配對（prefix = nvt × 32層 × 2 × 3072）：
#     nvt=8  →  1,572,864  ==  LoRA r=4 on qkv_proj
#     nvt=16 →  3,145,728  ==  LoRA r=8 on qkv_proj      （已跑）
#     nvt=64 → 12,582,912  ==  LoRA r=8 all-linear       （已跑）
# 跑完會有三個容量點，可以畫出完整的曲線而不只是兩個端點。
#
# 另一項對齊 SVEN 的改動：`--prefix_dropout 0.1`。
# SVEN 對每層的 prefix key/value 加 dropout（sven/model.py:20,30-31，預設 --dropout 0.1），
# PEFT 的 PrefixEncoder 沒有這一層，train_prefix.py 用 forward hook 補上。
#
# ⚠️ 變數混淆提醒：nvt=16 那一輪**沒有** dropout，這一輪長度與 dropout 同時改。
# 先跑 SVEN 的完整配置（主要問題是「SVEN 的設定行不行」），若明顯改善再跑
# ARMS=prefix8nodrop 做歸因，才知道是長度還是 dropout 的功勞。
#
# 用法：
#   bash scripts/run_sven_scale.sh                       # 只跑 prefix nvt=8（約 2.5 小時）
#   ARMS=prefix8nodrop bash scripts/run_sven_scale.sh    # 歸因 ablation（結果好再跑）
#   ARMS=lorar4 bash scripts/run_sven_scale.sh           # 參數配對用（多半不需要）
set -euo pipefail

MODEL="${MODEL:-microsoft/Phi-3-mini-4k-instruct}"
TRAIN_FILE="${TRAIN_FILE:-data/train_pref.jsonl}"
OUT="${OUT:-outputs/dpo-arms}"
# 預設只跑 prefix nvt=8。LoRA r=4 是「參數量精確配對」用的，但這一輪不需要：
# LoRA 的趨勢是參數越少越好（all-linear 12.6M −4.27 → qkv_proj 3.1M +1.22），
# 所以 r=4(1.57M) 大概率 >= +1.22。拿 prefix nvt=8(1.57M) 去比現有的
# lora_qkv(3.1M) 已經對 prefix 寬容——它用一半參數對上 LoRA。
# 只有在 prefix nvt=8 反而贏過 lora_qkv 時，才需要補跑 r=4 排除參數量效應。
ARMS="${ARMS:-prefix8}"

# --- 與 run_dpo_arms.sh / run_param_matched.sh 逐項相同 ---
BETA="${BETA:-0.05}"
STEPS="${STEPS:-800}"
BATCH="${BATCH:-1}"
ACCUM="${ACCUM:-64}"
MAX_LENGTH=2048
MAX_PROMPT_LENGTH=1024
MAX_GRAD_NORM=0.3
SEED=42

# lr 沿用各自那一臂已掃出的值。縮小容量後最佳 lr 可能移動，但先固定才能把
# 「prefix 長度」單獨隔離出來——一次只改一個變數。
PREFIX_LR="${PREFIX_LR:-5e-5}"
LORA_LR="${LORA_LR:-5e-6}"
# SVEN 的 --dropout 預設值
PREFIX_DROPOUT="${PREFIX_DROPOUT:-0.1}"

mkdir -p "$OUT"
common=(
  --model "$MODEL" --train_file "$TRAIN_FILE"
  --objective dpo --beta $BETA
  --max_steps $STEPS --batch_size "$BATCH" --grad_accum "$ACCUM"
  --max_length $MAX_LENGTH --max_prompt_length $MAX_PROMPT_LENGTH
  --max_grad_norm $MAX_GRAD_NORM
  --seed $SEED --bf16
  --save_steps 100 --save_total_limit 10
)

for arm in $ARMS; do
  case "$arm" in
    prefix8)
      echo "=== P-sven: prefix nvt=8 + dropout $PREFIX_DROPOUT（1,572,864 參數）==="
      python train_prefix.py "${common[@]}" \
        --peft_method prefix --num_virtual_tokens 8 --prefix_init_scale 0 \
        --prefix_dropout "$PREFIX_DROPOUT" \
        --lr "$PREFIX_LR" --output_dir "$OUT/prefix_nvt8" \
        2>&1 | tee "$OUT/prefix_nvt8.log"
      ;;
    lorar4)
      echo "=== P-sven: LoRA r=4 on qkv_proj（1,572,864 參數，配對用）==="
      python train_prefix.py "${common[@]}" \
        --peft_method lora --lora_r 4 --lora_alpha 8 \
        --lora_target_modules qkv_proj \
        --lr "$LORA_LR" --output_dir "$OUT/lora_r4" \
        2>&1 | tee "$OUT/lora_r4.log"
      ;;
    prefix8nodrop)
      # 歸因用：nvt=8 但**不加** dropout。nvt=16 沒有 dropout、nvt=8 有，
      # 一次動了兩個變數；若 nvt=8 明顯改善，跑這組才分得出是長度還是 dropout 的功勞。
      echo "=== 歸因 ablation: prefix nvt=8，無 dropout ==="
      python train_prefix.py "${common[@]}" \
        --peft_method prefix --num_virtual_tokens 8 --prefix_init_scale 0 \
        --lr "$PREFIX_LR" --output_dir "$OUT/prefix_nvt8_nodrop" \
        2>&1 | tee "$OUT/prefix_nvt8_nodrop.log"
      ;;
    *) echo "未知的 arm：$arm（可用：prefix8 prefix8nodrop lorar4）"; exit 1 ;;
  esac
done

cat <<EOF

完成。判讀（跟既有的兩個 prefix 尺寸並排）：

  python eval/compare_lr_sweep.py $OUT/prefix_nvt8.log \\
      $OUT/prefix.log $OUT/prefix_nvt64.log

最想看的是 prefix nvt=8 的 \`rewards/chosen early\`。nvt=16 是 −0.770、nvt=64 是 −4.441，
若 nvt=8 明顯更接近 0，就確認了「初始擾動隨 prefix 長度放大」這個機制，
而 SVEN 選 5~12 正是為了把它壓住。

接著跑評測（**兩者都要 --max_new_tokens 2048**）：

  python eval/run_humaneval.py --adapter $OUT/prefix_nvt8 --max_new_tokens 2048 \\
      --out outputs/humaneval_prefix_nvt8.json \\
      --dump outputs/humaneval_prefix_nvt8_detail.jsonl

utility 若從 nvt=16 的 −10.98 明顯回升，就代表先前的結論是
「我們把 prefix 開得太大」而不是「prefix 這個方法不行」——論文的措辭要跟著改。
EOF
