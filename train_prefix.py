"""用 ProSec 偏好資料訓練一組 adapter（PEFT PrefixTuning 或 LoRA），支援 DPO 與 SimPO。

檔名沿用 train_prefix.py 以免動到 RUNBOOK / eval 腳本的既有指令；
`--peft_method lora` 才是論文原本的做法（ProSec Appendix C：LoRA r=8, alpha=16）。

設計對應關係（沿用舊 SVEN 概念，但實作全移到現代棧）：
  - SVEN 的「sec prefix」      → PEFT PrefixTuning adapter（只訓練 prefix 參數）
  - SVEN 的 prefix 零初始化    → PrefixTuningConfig(init_weights="zero")（僅 prefix 適用）
  - ProSec 論文原本的 PEFT     → LoraConfig(r=8, lora_alpha=16)
  - ProSec 的 SimPO            → TRL CPOTrainer(loss_type="simpo", cpo_alpha=0)
  - 舊版沿用的 DPO             → TRL DPOTrainer（保留為 ablation 用）

資料格式（每行 jsonl）：{"prompt", "chosen", "rejected", ...}
  由 data/convert_prosec_to_pref.py 從 ProSec 最終資料產生。

範例：
  # 主線：DPO（beta 預設 0.1）。SimPO 在這份資料上會失控，見 EXPERIMENTS.md S2-repro
  python train_prefix.py \
      --model microsoft/Phi-3-mini-4k-instruct \
      --train_file data/train_pref.jsonl \
      --output_dir outputs/phi3-prefix-dpo \
      --objective dpo --lr 1e-4 \
      --num_virtual_tokens 16 --epochs 1 \
      --batch_size 1 --grad_accum 16 --bf16

  # LoRA 對照臂（E1）：除了 --peft_method 之外設定與上面完全相同
  python train_prefix.py --peft_method lora --lora_r 8 --lora_alpha 16 \
      --objective dpo --lr 5e-6 ...

  # ablation：SimPO（論文預設，已知會失控，保留供對照）
  python train_prefix.py --objective simpo --beta 1.5 --gamma 0.5 --cpo_alpha 0 ...
"""
import argparse
import dataclasses
import inspect

import torch
# peft 0.19.1 會直接讀 torch.distributed.tensor.DTensor，但有些 torch build 不會
# 自動載入這個 submodule，於是 LoRA 掛到 nn.Linear 上時噴 AttributeError。
# 手動 import 一次把它掛上去（PrefixTuning 走不到這條路徑，所以之前沒踩到）。
try:
    import torch.distributed.tensor  # noqa: F401
except Exception:
    pass
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PrefixTuningConfig, TaskType
from trl import CPOConfig, CPOTrainer, DPOConfig, DPOTrainer

# beta 在兩個目標函數之間語意完全不同，不可互換：
#   DPO   beta=0.1  控制與 reference model 的 KL 約束強度
#   SimPO beta=1.5  縮放 length-normalized reward 的溫度
# 把 DPO 的 0.1 套到 SimPO 上，訓練訊號會弱到幾乎不存在，所以預設值依 objective 決定。
DEFAULT_BETA = {"dpo": 0.1, "simpo": 1.5}

# PEFT 的 TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING 沒有 phi3 條目，
# 不指定 target_modules 會直接報錯，所以自己列。Phi-3 把 q/k/v 融成 qkv_proj、
# gate/up 融成 gate_up_proj，模組名與 llama 系列不同。
# 論文只寫了 r=8 / alpha=16（Appendix C），沒寫 target modules；他們用的
# SeCAlign-llama-factory 未公開，而 LLaMA-Factory 的預設是 all-linear，故取 all-linear。
LORA_TARGETS = {
    "phi3": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj"],
}

# 舊版 PEFT 沒有原生零初始化，需退回手動縮放。
_HAS_INIT_WEIGHTS = any(f.name == "init_weights"
                        for f in dataclasses.fields(PrefixTuningConfig))



def attach_prefix_dropout(model, p):
    """對 prefix 的 key/value 加 dropout，對齊 SVEN。

    SVEN 在 `sven/model.py` 建了 `nn.Dropout(config.prefix_dropout)`（預設 0.1），
    套在每層的 `prefix_params[key_idx]` / `[val_idx]` 上；PEFT 的 PrefixEncoder
    沒有這一層。PEFT 的 `get_prompt()` 是拿 `prompt_encoder(prompt_tokens)` 的輸出
    去 reshape 成 past_key_values，所以在那個輸出掛 hook 位置等價。

    用 functional dropout 並讀 `mod.training`，這樣 model.eval() 時會自動關閉；
    自建一個 nn.Dropout 模組的話它的 training flag 不會跟著切換。
    """
    encoders = []
    for m in model.modules():
        pe = getattr(m, "prompt_encoder", None)
        if isinstance(pe, torch.nn.ModuleDict):
            encoders.extend(pe.values())
    if not encoders:
        raise SystemExit("找不到 prompt_encoder —— --prefix_dropout 只適用於 "
                         "--peft_method prefix")
    for enc in encoders:
        enc.register_forward_hook(
            lambda mod, inp, out, _p=p:
                torch.nn.functional.dropout(out, _p, training=mod.training))
    return len(encoders)

def build_dataset(train_file, tokenizer, use_chat_template, system_prompt=None,
                  subset="all"):
    ds = load_dataset("json", data_files=train_file, split="train")

    # subset：§3.3 的暖身訓練只用 D_sec（論文「train on Dsec for 1k steps」）
    if subset != "all":
        if "benign" not in ds.column_names:
            raise SystemExit(
                "--subset 需要資料含 benign 欄位；請用新版 "
                "data/convert_prosec_to_pref.py 重新產生 train_pref.jsonl")
        want = (subset == "dnorm")
        ds = ds.filter(lambda ex: bool(ex["benign"]) == want)

    def fmt(ex):
        if use_chat_template:
            msgs = []
            # ProSec 的 encode_oneturn_phi3 有帶 system prompt，復現時要一致，
            # 否則 policy 看到的 prompt 分布與論文不同
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": ex["prompt"]})
            ex["prompt"] = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
        return ex

    keep = {"prompt", "chosen", "rejected"}
    ds = ds.map(fmt, remove_columns=[c for c in ds.column_names if c not in keep])
    return ds


def build_trainer(args, beta, model, peft_cfg, tokenizer, train_ds):
    """依 objective 選 trainer。兩條路徑共用 dataset、peft config 與 logging。"""
    common = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        beta=beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        bf16=args.bf16,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if args.gradient_checkpointing else None),
        seed=args.seed,
        logging_steps=5,
        save_strategy="steps" if args.save_steps else "epoch",
        save_steps=args.save_steps or 500,
        save_total_limit=args.save_total_limit,
        report_to=[],
        remove_unused_columns=False,
    )

    if args.objective == "simpo":
        # CPOTrainer 是 reference-free，不需要 ref_model，也不需要 disable_adapter 前向。
        cfg = CPOConfig(loss_type="simpo", simpo_gamma=args.gamma,
                        cpo_alpha=args.cpo_alpha, **common)
        cls, kwargs = CPOTrainer, {}
    else:
        # 給了 peft_config + ref_model=None → TRL 用 disable_adapter 當 reference。
        cfg = DPOConfig(loss_type=args.loss_type, **common)
        cls, kwargs = DPOTrainer, {"ref_model": None}

    kwargs.update(model=model, args=cfg, train_dataset=train_ds, peft_config=peft_cfg)
    # TRL 版本相容：新版用 processing_class，舊版用 tokenizer
    sig = inspect.signature(cls.__init__).parameters
    kwargs["processing_class" if "processing_class" in sig else "tokenizer"] = tokenizer
    return cls(**kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--train_file", default="data/sample_pref.jsonl")
    ap.add_argument("--output_dir", default="outputs/phi3-prefix-dpo")
    # === PEFT 方法 ===
    ap.add_argument("--peft_method", choices=["prefix", "lora"], default="prefix",
                    help="prefix=本研究主線；lora=ProSec 論文原本的做法（Appendix C）")
    ap.add_argument("--num_virtual_tokens", type=int, default=16,
                    help="僅 --peft_method prefix 時使用")
    ap.add_argument("--lora_r", type=int, default=8, help="論文 Appendix C：r=8")
    ap.add_argument("--lora_alpha", type=int, default=16, help="論文 Appendix C：alpha=16")
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--prefix_dropout", type=float, default=0.0,
                    help="對 prefix 的 key/value 加 dropout，對齊 SVEN（其 --dropout 預設 0.1）。"
                         "PEFT 的 PrefixEncoder 沒有這一層，我們用 forward hook 補上。"
                         "僅 --peft_method prefix 有效")
    ap.add_argument("--lora_target_modules", default="auto",
                    help="逗號分隔的模組名；auto=依 model_type 取 all-linear（見 LORA_TARGETS）")
    # === 目標函數 ===
    ap.add_argument("--objective", choices=["simpo", "dpo"], default="dpo",
                    help="dpo=主線（TRL DPOTrainer，有 reference model 當錨，穩定）。"
                         "simpo=ProSec 論文預設，但在這份資料上會失控——"
                         "低 ρ 尾巴的長度歸一化 margin 目標不可達、又沒有 reference 錨，"
                         "prefix 與 LoRA 兩次都把 chosen/rejected 一起壓垮，見 EXPERIMENTS.md S2-repro。"
                         "論文 Table 8 本身也驗證過 DPO 在這份資料上有效（Vul 40.76→34.65）")
    ap.add_argument("--beta", type=float, default=None,
                    help=f"不給則依 objective 自動選：{DEFAULT_BETA}。兩者語意不同，不可互換")
    ap.add_argument("--gamma", type=float, default=0.5,
                    help="SimPO 的 target reward margin（論文設定 0.5）。dpo 時忽略")
    ap.add_argument("--cpo_alpha", type=float, default=0.0,
                    help="CPO 的 SFT 項權重。**純 SimPO 必須為 0**；"
                         "TRL 預設 1.0 會變成 CPO-SimPO 混合，不是論文的方法")
    ap.add_argument("--loss_type", default="sigmoid",
                    help="僅 --objective dpo 時使用：sigmoid(標準DPO) / ipo / hinge")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--max_prompt_length", type=int, default=512)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--no_chat_template", action="store_true")
    ap.add_argument("--system_prompt", default=None,
                    help="套進 chat template 的 system 訊息。論文管線用的是 "
                         "\"You are helpful coding assistant.\"（ProSec "
                         "influence_score/data_utils_refactored.py encode_oneturn_phi3）")
    ap.add_argument("--subset", choices=["all", "dsec", "dnorm"], default="all",
                    help="只取資料的一部分。§3.3 的暖身訓練用 dsec")
    ap.add_argument("--load_4bit", action="store_true", help="用 bitsandbytes 4-bit 載入以省顯存")
    # === 穩定性相關 ===
    ap.add_argument("--warmup_ratio", type=float, default=0.1,
                    help="前期 lr 暖機比例")
    ap.add_argument("--max_grad_norm", type=float, default=0.3,
                    help="梯度裁剪上限。注意：log 印的 grad_norm 是裁剪『前』的值，"
                         "實測常在 8~21，代表每一步都被夾到此上限")
    ap.add_argument("--lr_scheduler_type", default="cosine",
                    help="lr 排程：cosine / linear / constant_with_warmup")
    ap.add_argument("--gradient_checkpointing", action="store_true",
                    help="省顯存（犧牲速度）。長序列 OOM 時開")
    # === 子集 / 快速迭代 ===
    ap.add_argument("--max_samples", type=int, default=None,
                    help="只取 N 筆訓練（lr sweep 等快速迭代用；正式跑全量時不要設）")
    ap.add_argument("--max_steps", type=int, default=-1,
                    help="覆寫 epochs，只跑固定步數（smoke 用，例如 20）")
    ap.add_argument("--prefix_init_scale", type=float, default=0.0,
                    help="0=零初始化(仿SVEN，最穩，走 PEFT 原生 init_weights='zero')；"
                         "1=PEFT 預設隨機初始化(會不穩)；其他值=隨機後手動縮放")
    ap.add_argument("--save_steps", type=int, default=None,
                    help="每 N 步存一個 checkpoint。長訓練必開——"
                         "偏好學習後期可能由『拉高 chosen』翻轉成『壓垮兩側』，"
                         "要有中途點才能回頭挑")
    ap.add_argument("--save_total_limit", type=int, default=None,
                    help="最多保留幾個 checkpoint（省磁碟）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # PrefixTuning 會把 prefix 塞進 past_key_values；gradient checkpointing 重算時
    # 會再串接一次，key 長度變成 (L+nvt)+L，attention mask 對不上直接爆
    # （RuntimeError: The expanded size of the tensor ... at non-singleton dimension 3）。
    # LoRA 沒有這個問題，因為它不碰 past_key_values。
    if args.gradient_checkpointing and args.peft_method == "prefix":
        print("⚠️  PrefixTuning 與 gradient checkpointing 不相容（prefix 的 "
              "past_key_values 在重算時會被重複串接），已自動關閉。"
              "顯存不夠請改小 --batch_size 或 --max_length")
        args.gradient_checkpointing = False

    beta = args.beta if args.beta is not None else DEFAULT_BETA[args.objective]
    if args.objective == "simpo" and args.cpo_alpha > 0:
        print(f"⚠️  cpo_alpha={args.cpo_alpha} > 0 → 這是 CPO-SimPO 混合，不是純 SimPO。"
              "要對齊 ProSec 論文請用 --cpo_alpha 0")
    if args.beta is not None and args.objective == "simpo" and args.beta < 0.5:
        print(f"⚠️  SimPO 的 beta={args.beta} 偏低（論文用 1.5）。"
              "DPO 的 beta 尺度不可直接套用到 SimPO")

    print(f"objective={args.objective}  beta={beta}"
          + (f"  gamma={args.gamma}  cpo_alpha={args.cpo_alpha}"
             if args.objective == "simpo" else f"  loss_type={args.loss_type}")
          + f"  lr={args.lr}  epochs={args.epochs}"
          + f"  effective_batch={args.batch_size * args.grad_accum}"
          + (f"  max_steps={args.max_steps}" if args.max_steps > 0 else ""))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"載入 base model：{args.model}")
    model_kwargs = dict(
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.enable_input_require_grads()  # PEFT + gradient checkpointing 需要

    zero_init = args.peft_method == "prefix" and args.prefix_init_scale == 0.0
    if args.peft_method == "lora":
        if args.lora_target_modules == "auto":
            mtype = model.config.model_type
            if mtype not in LORA_TARGETS:
                raise SystemExit(
                    f"model_type={mtype} 沒有內建 LoRA target modules，"
                    "請用 --lora_target_modules 明確指定")
            targets = LORA_TARGETS[mtype]
        else:
            targets = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        peft_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, target_modules=targets, bias="none",
        )
        print(f"PEFT=LoRA r={args.lora_r} alpha={args.lora_alpha} targets={targets}")
    else:
        # 只訓練一組 prefix —— 取代 SVEN 手刻的 prefix_params + hf/ modeling
        peft_kwargs = {}
        if zero_init and _HAS_INIT_WEIGHTS:
            peft_kwargs["init_weights"] = "zero"
        peft_cfg = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=args.num_virtual_tokens,
            **peft_kwargs,
        )
        print(f"PEFT=PrefixTuning nvt={args.num_virtual_tokens}")

    train_ds = build_dataset(args.train_file, tokenizer, not args.no_chat_template,
                             system_prompt=args.system_prompt, subset=args.subset)
    if args.max_samples is not None and args.max_samples < len(train_ds):
        train_ds = train_ds.shuffle(seed=args.seed).select(range(args.max_samples))
        print(f"取隨機子集：{args.max_samples} 筆")
    print(f"訓練資料：{len(train_ds)} 筆偏好對（subset={args.subset}）")

    trainer = build_trainer(args, beta, model, peft_cfg, tokenizer, train_ds)

    # === prefix 初始值 ===
    # PEFT 預設隨機初始化，會讓 policy 從第 0 步就大幅偏離 base，造成 reward 量級爆炸。
    # 零初始化 → policy 起步 ≈ base（仿 SVEN）。優先用 PEFT 原生 init_weights，
    # 舊版 PEFT 或需要中間值（如 0.01）時才退回手動縮放。
    if args.prefix_dropout > 0:
        n = attach_prefix_dropout(trainer.model, args.prefix_dropout)
        print(f"prefix dropout={args.prefix_dropout}（掛在 {n} 個 prompt_encoder 上，對齊 SVEN）")

    if args.peft_method == "lora":
        # LoRA 的 B 矩陣預設就是零，policy 起步已經 == base，不需要額外處理
        pass
    elif zero_init and _HAS_INIT_WEIGHTS:
        print("prefix 初始化：PEFT 原生 init_weights='zero'")
    elif args.prefix_init_scale != 1.0:
        n_scaled = 0
        with torch.no_grad():
            for n, p in trainer.model.named_parameters():
                if p.requires_grad:  # 只有 prefix 參數可訓練
                    p.mul_(args.prefix_init_scale)
                    n_scaled += p.numel()
        print(f"prefix 初始值手動縮放 ×{args.prefix_init_scale}（{n_scaled} 個參數）")
    else:
        print("prefix 初始化：PEFT 預設隨機（不穩，僅供 ablation）")

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"完成。{args.peft_method} adapter 已存到 {args.output_dir}")


if __name__ == "__main__":
    main()
