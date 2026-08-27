"""§3.3 第 2 步：在暖身訓練的每個 checkpoint 上，量測每筆偏好對的 log-prob。

對應 ProSec `influence_score/training_dynamics_refactored.py` + `scores.py`。
論文 Eq.(3)(4) 要的是一條「隨 checkpoint 變化」的軌跡：

    r(x, y, theta) = (1/|y|) * log p_theta(y | x)      <- 長度歸一化，scores.py 的 ln_seq_logps

暖身訓練（只用 D_sec、1000 steps、每 100 步存檔）產生 theta_1..theta_10，
本腳本對「整份資料」在每個 theta_t 上算 r，輸出給 data/select_dnorm.py 算相關係數。

只需要 chosen 側就能算論文的預設設定（f=r(x_n,y_n)、g=-r(x_v,y_f) 都落在 chosen 側），
`--sides chosen` 可以省一半算力；要跑 Table 3 的其他 f/g 變體才需要 both。

用法：
  python data/collect_training_dynamics.py \
      --train_file data/train_pref.jsonl \
      --base_model microsoft/Phi-3-mini-4k-instruct \
      --ckpt_root outputs/phi3-lora-warmup-dsec \
      --out_dir outputs/dynamics \
      --system_prompt "You are helpful coding assistant." --bf16
"""
import argparse
import json
import os
import re

import torch
# peft 0.19.1 會直接讀 torch.distributed.tensor.DTensor，但有些 torch build 不會
# 自動載入這個 submodule，於是 LoRA 掛到 nn.Linear 上時噴 AttributeError。
# 手動 import 一次把它掛上去（PrefixTuning 走不到這條路徑，所以之前沒踩到）。
try:
    import torch.distributed.tensor  # noqa: F401
except Exception:
    pass
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ProSec influence_score/data_utils_refactored.py 的 encode_oneturn_phi3 用的 system prompt
PROSEC_SYSTEM_PROMPT = "You are helpful coding assistant."


def load_rows(train_file):
    rows = []
    with open(train_file) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            e.setdefault("id", i)
            rows.append(e)
    return rows


def encode(tokenizer, instruction, response, system_prompt, max_length):
    """回傳 (input_ids, labels)。labels 只在 response 上有值，其餘 -100。

    對齊 ProSec 的 encode_oneturn_*：prompt 段整段遮掉，長度歸一化的分母因此
    只算 response 的 token 數。
    """
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": instruction})
    prompt_ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
    resp_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    resp_ids = resp_ids + [tokenizer.eos_token_id]

    # 超長時砍 prompt 的頭而不是砍 response 的尾——response 被截斷會直接扭曲
    # 長度歸一化的分子與分母，是 SimPO 尺度下最嚴重的污染來源。
    keep_resp = min(len(resp_ids), max_length - 1)
    resp_ids = resp_ids[:keep_resp]
    keep_prompt = max_length - len(resp_ids)
    if len(prompt_ids) > keep_prompt:
        prompt_ids = prompt_ids[-keep_prompt:]

    input_ids = prompt_ids + resp_ids
    labels = [-100] * len(prompt_ids) + resp_ids
    return input_ids, labels


@torch.no_grad()
def score_batch(model, batch, pad_id, device):
    """回傳 (sum_logp, ln_logp, n_tokens)，皆為 list。"""
    maxlen = max(len(x[0]) for x in batch)
    input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), maxlen), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
    for i, (ids, lab) in enumerate(batch):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        labels[i, : len(lab)] = torch.tensor(lab)
        attn[i, : len(ids)] = 1
    input_ids, labels, attn = input_ids.to(device), labels.to(device), attn.to(device)

    logits = model(input_ids=input_ids, attention_mask=attn).logits
    # 標準的 causal shift：位置 t 的 logits 預測位置 t+1 的 token。
    # ProSec 的 scores.py 少做了這個 shift，我們照正確定義做（見 docs/REPRO_PAPER.md）。
    logits = logits[:, :-1, :]
    tgt = labels[:, 1:]
    mask = tgt != -100
    logps = torch.log_softmax(logits.float(), dim=-1)
    tok_logp = torch.gather(logps, 2, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    tok_logp = tok_logp * mask
    total = tok_logp.sum(dim=1)
    n = mask.sum(dim=1).clamp_min(1)
    return total.tolist(), (total / n).tolist(), n.tolist()


def run_checkpoint(model, tokenizer, encoded, sides, batch_size, device, out_path):
    pad_id = tokenizer.pad_token_id
    out = {}
    for side in sides:
        # 依長度排序批次化，把 padding 浪費壓到最低；結果再照 id 存回去
        order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][side][0]))
        for s in range(0, len(order), batch_size):
            idxs = order[s : s + batch_size]
            total, ln, n = score_batch(
                model, [encoded[i][side] for i in idxs], pad_id, device)
            for j, i in enumerate(idxs):
                rec = out.setdefault(encoded[i]["id"], {})
                rec[f"{side}_logp"] = total[j]
                rec[f"{side}_ln_logp"] = ln[j]
                rec[f"{side}_len"] = n[j]
            if (s // batch_size) % 50 == 0:
                print(f"  [{side}] {s}/{len(order)}", flush=True)
    with open(out_path, "w") as f:
        for rid in sorted(out):
            f.write(json.dumps({"id": rid, **out[rid]}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--base_model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--ckpt_root", required=True,
                    help="暖身訓練的 output_dir，底下應有 checkpoint-100 ... checkpoint-1000")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt_every", type=int, default=100,
                    help="只取 id 為此值倍數的 checkpoint（論文：每 100 步）")
    ap.add_argument("--sides", choices=["chosen", "both"], default="both",
                    help="chosen=只算 chosen 側（論文預設 f/g 只需要它，快一倍）")
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--system_prompt", default=PROSEC_SYSTEM_PROMPT)
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_rows(args.train_file)
    sides = ["chosen"] if args.sides == "chosen" else ["chosen", "rejected"]
    print(f"編碼 {len(rows)} 筆 × {len(sides)} 側 ...")
    encoded = []
    for e in rows:
        item = {"id": e["id"]}
        for side in sides:
            item[side] = encode(tokenizer, e["prompt"], e[side],
                                args.system_prompt, args.max_length)
        encoded.append(item)

    ckpts = []
    for name in os.listdir(args.ckpt_root):
        m = re.fullmatch(r"checkpoint-(\d+)", name)
        if m and int(m.group(1)) % args.ckpt_every == 0:
            ckpts.append((int(m.group(1)), os.path.join(args.ckpt_root, name)))
    ckpts.sort()
    if not ckpts:
        raise SystemExit(f"{args.ckpt_root} 底下找不到 checkpoint-* —— "
                         "暖身訓練要加 --save_steps 100")
    print(f"找到 {len(ckpts)} 個 checkpoint：{[c for c, _ in ckpts]}")

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map={"": 0} if device == "cuda" else None,
    )
    base.config.use_cache = False

    for ckpt_id, ckpt_dir in ckpts:
        out_path = os.path.join(args.out_dir, f"checkpoint-{ckpt_id}.jsonl")
        if os.path.exists(out_path):
            print(f"跳過已完成的 checkpoint-{ckpt_id}")
            continue
        print(f"=== checkpoint-{ckpt_id} ===", flush=True)
        model = PeftModel.from_pretrained(base, ckpt_dir).eval()
        run_checkpoint(model, tokenizer, encoded, sides,
                       args.batch_size, device, out_path)
        # 卸掉 adapter，下一個 checkpoint 重新掛，避免疊加
        model.unload()
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"完成 → {args.out_dir}")


if __name__ == "__main__":
    main()
