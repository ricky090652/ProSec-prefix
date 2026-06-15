"""用 ProSec 偏好資料訓練一組 prefix（PEFT PrefixTuning），透過 TRL 做 DPO。

設計對應關係（沿用舊 SVEN 概念，但實作全移到現代棧）：
  - SVEN 的「sec prefix」      → PEFT PrefixTuning adapter（只訓練 prefix 參數）
  - SVEN 的「無 prefix 參考前向」→ TRL 對 PEFT 模型自動用 disable_adapter 當 reference
  - SVEN 的 DPO loss           → TRL DPOTrainer（beta 可調）
  - SVEN 的 diff-level token mask → 不需要（ProSec 是整段 chosen/rejected 偏好對）

資料格式（每行 jsonl）：{"prompt", "chosen", "rejected", ...}
  由 data/convert_prosec_to_pref.py 從 ProSec 最終資料產生。

範例：
  python train_prefix.py \
      --model codellama/CodeLlama-7b-Instruct-hf \
      --train_file data/train_pref.jsonl \
      --output_dir outputs/codellama7b-prefix-dpo \
      --num_virtual_tokens 16 --beta 0.1 --lr 5e-5 \
      --epochs 1 --batch_size 1 --grad_accum 16 --bf16

  # 想跑 SimPO 而非 DPO：加 --loss_type simpo（見下方說明）
"""
import argparse
import inspect

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PrefixTuningConfig, TaskType
from trl import DPOConfig, DPOTrainer


def build_dataset(train_file, tokenizer, use_chat_template):
    ds = load_dataset("json", data_files=train_file, split="train")

    def fmt(ex):
        if use_chat_template:
            ex["prompt"] = tokenizer.apply_chat_template(
                [{"role": "user", "content": ex["prompt"]}],
                tokenize=False, add_generation_prompt=True,
            )
        return ex

    keep = {"prompt", "chosen", "rejected"}
    ds = ds.map(fmt, remove_columns=[c for c in ds.column_names if c not in keep])
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="codellama/CodeLlama-7b-Instruct-hf")
    ap.add_argument("--train_file", default="data/sample_pref.jsonl")
    ap.add_argument("--output_dir", default="outputs/codellama7b-prefix-dpo")
    ap.add_argument("--num_virtual_tokens", type=int, default=16)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--max_prompt_length", type=int, default=512)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--no_chat_template", action="store_true")
    ap.add_argument("--load_4bit", action="store_true", help="用 bitsandbytes 4-bit 載入以省顯存")
    ap.add_argument("--loss_type", default="sigmoid",
                    help="TRL DPO loss_type：sigmoid(標準DPO) / ipo / 等。SimPO 請見 README。")
    args = ap.parse_args()

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

    # 只訓練一組 prefix —— 取代 SVEN 手刻的 prefix_params + hf/ modeling
    peft_cfg = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=args.num_virtual_tokens,
    )

    train_ds = build_dataset(args.train_file, tokenizer, not args.no_chat_template)
    print(f"訓練資料：{len(train_ds)} 筆偏好對")

    dpo_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        beta=args.beta,
        loss_type=args.loss_type,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        bf16=args.bf16,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
        remove_unused_columns=False,
    )

    # TRL 版本相容：新版用 processing_class，舊版用 tokenizer
    trainer_kwargs = dict(
        model=model,
        ref_model=None,          # 給了 peft_config + ref_model=None → TRL 用 disable_adapter 當 reference
        args=dpo_args,
        train_dataset=train_ds,
        peft_config=peft_cfg,
    )
    sig = inspect.signature(DPOTrainer.__init__).parameters
    if "processing_class" in sig:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = DPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"完成。prefix adapter 已存到 {args.output_dir}")


if __name__ == "__main__":
    main()
