"""從 intra pairs 抽 Phi-3 的內部表徵（DuoSteer Stage 2 第一步的移植版）。

原版：DuoSteer-Safe-Correct-Code-Gen/localization/extract_representations.py。
輸出格式與原版相同，下游的 probe / steering 向量腳本可以直接吃。

與原版的差異（都是為了對齊本 repo 的訓練與評測，或修掉原版的問題）：
  1. prompt 直接用 ProSec 的指令，不套 DuoSteer 的 CODE_GENERATION_PROMPT 模板；
     原版的 --prompt_mode 在這裡沒有意義，拿掉（原版 vuln 那側也沒真的吃到這個參數）。
  2. 斷詞 = 評測的 prompt 斷法 + TRL 的 completion 斷法：
       prompt_ids = tokenizer(apply_chat_template(..., add_generation_prompt=True))
                    ← 與 eval/gen_for_icd.py 相同（Phi-3 不自動加 BOS）
       resp_ids   = tokenizer(code, add_special_tokens=False)
                    ← 與 TRL DPOTrainer.tokenize_row 相同
     response mask 剛好蓋住程式碼本身。原版用 chat template 斷完整對話，
     mask 會多包進結尾的 <|end|>、<|endoftext|>。
  3. 左側 padding 時明確傳 position_ids，讓每筆的位置與不 padding 時相同。
     RoPE 只看相對位置，不傳也只差數值誤差（tiny Phi-3 實測 9e-7），這是保險不是修 bug。
  4. --token_agg 預設 response_mean（論文的設定；原版預設 response_last）。
  5. 聚合在 GPU 上一次對整層做，再切成各 head（平均的切片 = 切片的平均），
     不用原版逐 head、逐筆的 Python 迴圈。
  6. --cwe_id 可以給 all（合併所有 CWE），對應 ProSec 的全域設定。

表徵定義（論文 §3）：
  layer  h̄^(ℓ)   = output_hidden_states[1:] 在 response token 上的平均
  head   z̄^(ℓ,j) = 第 ℓ 層 o_proj 的 input 切成 H 份後，在 response token 上的平均

輸出 {output_dir}/{model_slug}/{pairs 檔名}/{cwe_tag}/{token_agg}/
  layer_XX.pt               {"safe": (n_pairs, 3072), "vuln": (n_pairs, 3072)}
  head_layer_XX_head_YY.pt  {"safe": (n_pairs, 96),   "vuln": (n_pairs, 96)}
  metadata.json
編號從 1 開始（與原版相同）。

用法：
  python steering/extract_representations.py \
      --input_file data/steering/intra.jsonl --cwe_id all --bf16
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def cwe_key(cwe):
    return str(cwe).lower().removeprefix("cwe-").lstrip("0") or "0"


def load_pairs(path, cwe_id, max_pairs):
    target = None if cwe_id == "all" else cwe_key(cwe_id)
    pairs = []
    with open(path) as f:
        for line in f:
            p = json.loads(line)
            if target is None or cwe_key(p["cwe_id"]) == target:
                pairs.append(p)
                if max_pairs and len(pairs) >= max_pairs:
                    break
    return pairs


# --------------------------------------------------------------------------- #
# 斷詞
# --------------------------------------------------------------------------- #

def build_messages(system_prompt, instruction):
    """與 eval/gen_for_icd.py 相同。主線的 DPO 臂訓練時沒有 system prompt。"""
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": instruction})
    return msgs


def tokenize_side(tokenizer, prompt, code, system_prompt):
    """回傳 (prompt_ids, resp_ids)。"""
    text = tokenizer.apply_chat_template(
        build_messages(system_prompt, prompt), tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(text)["input_ids"]
    resp_ids = tokenizer(code, add_special_tokens=False)["input_ids"]
    return prompt_ids, resp_ids


def check_alignment(tokenizer, pair, system_prompt):
    """拿第一筆印出 prompt / response 的交界，並與 chat template 斷完整對話的結果比對。

    不一致不代表錯（我們對齊的是 TRL 與評測，不是 template），但要知道差在哪。
    """
    prompt_ids, resp_ids = tokenize_side(
        tokenizer, pair["prompt"], pair["safe_code"], system_prompt)
    ours = prompt_ids + resp_ids
    msgs = build_messages(system_prompt, pair["prompt"])
    full = tokenizer.apply_chat_template(
        msgs + [{"role": "assistant", "content": pair["safe_code"]}], tokenize=True)
    tail = tokenizer.convert_ids_to_tokens(full[len(ours):])
    same = full[:len(ours)] == ours
    print("  斷詞交界（第一筆 safe）：")
    print(f"    prompt 結尾 : {tokenizer.convert_ids_to_tokens(prompt_ids[-3:])}")
    print(f"    response 開頭: {tokenizer.convert_ids_to_tokens(resp_ids[:6])}")
    print(f"    prompt={len(prompt_ids)}  response={len(resp_ids)}")
    print(f"    與 chat template 完整對話{'一致' if same else '不一致 ⚠️'}；"
          f"template 多出的結尾（不進 mask）：{tail}")


def prepare_batch(tokenizer, sides, system_prompt, device):
    """手動左側 padding。回傳 input_ids / attention_mask / position_ids / response_mask。"""
    seqs = [tokenize_side(tokenizer, s["prompt"], s["code"], system_prompt) for s in sides]
    max_len = max(len(p) + len(r) for p, r in seqs)
    pad_id = tokenizer.pad_token_id
    ids, attn, resp = [], [], []
    for p, r in seqs:
        if not r:
            raise ValueError("response 為空，無法聚合")
        pad = max_len - len(p) - len(r)
        ids.append([pad_id] * pad + p + r)
        attn.append([0] * pad + [1] * (len(p) + len(r)))
        resp.append([False] * (pad + len(p)) + [True] * len(r))
    attn = torch.tensor(attn, device=device)
    return {
        "input_ids": torch.tensor(ids, device=device),
        "attention_mask": attn,
        # 左側 padding 時預設的 position_ids 會把真實 token 往後推 pad 個位置
        "position_ids": (attn.cumsum(-1) - 1).clamp(min=0),
    }, torch.tensor(resp, device=device)


# --------------------------------------------------------------------------- #
# 抽取
# --------------------------------------------------------------------------- #

class HeadOutputCapture:
    """在每層 o_proj 掛 forward pre-hook，抓 (batch, seq, H*d_h) 的 input。

    Phi3Attention 的 attn_output 是 transpose(1, 2).reshape(b, s, H*d_h)，
    所以最後一維是 head-major：[head0 的 d_h 維 | head1 | ...]。
    """

    def __init__(self, model):
        self.captures = []
        self._hooks = [
            m.o_proj.register_forward_pre_hook(self._hook)
            for m in model.modules() if hasattr(m, "o_proj")
        ]

    def _hook(self, module, inputs):
        self.captures.append(inputs[0].detach())

    def clear(self):
        self.captures.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()


def aggregate(t, resp_mask, token_agg):
    """(b, s, d) → (b, d)，只看 response token，float32 計算。"""
    if token_agg == "response_last":
        return t[:, -1].float()   # 左側 padding：最後一格一定是 response 的最後一個 token
    m = resp_mask.unsqueeze(-1).float()
    return (t.float() * m).sum(1) / m.sum(1)


@torch.no_grad()
def extract_batch(model, tokenizer, sides, args, capture, n_heads, head_dim, device):
    enc, resp_mask = prepare_batch(tokenizer, sides, args.system_prompt, device)
    if capture is not None:
        capture.clear()
    out = model(**enc, output_hidden_states=True)
    res = {}
    if args.mode in ("layer", "both"):
        res["layer"] = torch.stack(
            [aggregate(hs, resp_mask, args.token_agg) for hs in out.hidden_states[1:]],
            dim=1).cpu()                                           # (b, L, d)
    if capture is not None:
        res["head"] = torch.stack(
            [aggregate(c, resp_mask, args.token_agg) for c in capture.captures],
            dim=1).view(len(sides), -1, n_heads, head_dim).cpu()   # (b, L, H, d_h)
    return res


def slugify(name):
    return re.sub(r"[^\w\-]", "_", name)


def main(args):
    input_file = Path(args.input_file)
    pairs = load_pairs(input_file, args.cwe_id, args.max_pairs)
    print(f"{input_file.name}：cwe={args.cwe_id} 共 {len(pairs)} pairs")
    if not pairs:
        return

    cwe_tag = "all" if args.cwe_id == "all" else f"cwe-{cwe_key(args.cwe_id)}"
    out_dir = (Path(args.output_dir) / slugify(args.model) / input_file.stem
               / cwe_tag / args.token_agg)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    cfg = model.config
    n_layers, n_heads = cfg.num_hidden_layers, cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_heads
    capture = HeadOutputCapture(model) if args.mode in ("head", "both") else None
    if capture is not None and len(capture._hooks) != n_layers:
        raise SystemExit(f"找到 {len(capture._hooks)} 個 o_proj，但模型有 {n_layers} 層")
    print(f"模型 {args.model}：{n_layers} 層 × {n_heads} heads × d_h={head_dim}，dtype={dtype}")
    check_alignment(tokenizer, pairs[0], args.system_prompt)

    acc = {"safe": {"layer": [], "head": []}, "vuln": {"layer": [], "head": []}}
    n_batches = (len(pairs) + args.batch_size - 1) // args.batch_size
    for bi, i in enumerate(range(0, len(pairs), args.batch_size), 1):
        batch = pairs[i:i + args.batch_size]
        for side in ("safe", "vuln"):
            sides = [{"prompt": p["prompt"], "code": p[f"{side}_code"]} for p in batch]
            res = extract_batch(model, tokenizer, sides, args, capture, n_heads, head_dim, device)
            for k, v in res.items():
                acc[side][k].append(v.to(dtype))
        print(f"  batch {bi}/{n_batches}", end="\r")
    print()

    if acc["safe"]["layer"]:
        safe, vuln = (torch.cat(acc[s]["layer"]) for s in ("safe", "vuln"))
        for l in range(n_layers):
            torch.save({"safe": safe[:, l].clone(), "vuln": vuln[:, l].clone()},
                       out_dir / f"layer_{l + 1:02d}.pt")
        print(f"  layer：{n_layers} 個檔，shape=({len(pairs)}, {cfg.hidden_size})")
    if acc["safe"]["head"]:
        safe, vuln = (torch.cat(acc[s]["head"]) for s in ("safe", "vuln"))
        for l in range(n_layers):
            for h in range(n_heads):
                torch.save({"safe": safe[:, l, h].clone(), "vuln": vuln[:, l, h].clone()},
                           out_dir / f"head_layer_{l + 1:02d}_head_{h + 1:02d}.pt")
        print(f"  head：{n_layers * n_heads} 個檔，shape=({len(pairs)}, {head_dim})")

    metadata = {
        "model": args.model, "cwe_id": args.cwe_id, "input_file": str(input_file),
        "mode": args.mode, "token_agg": args.token_agg, "dtype": str(dtype),
        "system_prompt": args.system_prompt,
        "n_pairs": len(pairs), "n_layers": n_layers, "n_heads": n_heads,
        "head_dim": head_dim, "hidden_dim": cfg.hidden_size,
        "pairs": [
            {"index": i, "id": p["id"], "src_id": p["src_id"],
             "cwe_id": p["cwe_id"], "lang": p["lang"]}
            for i, p in enumerate(pairs)
        ],
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    if capture is not None:
        capture.remove()
    size = sum(f.stat().st_size for f in out_dir.glob("*.pt")) / 1e6
    print(f"→ {out_dir}（{size:.1f} MB）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--input_file", required=True, help="steering/build_pairs.py 的輸出")
    ap.add_argument("--cwe_id", default="all", help="all 或單一 CWE，例如 cwe-338 / 338")
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--output_dir", default="data/representations")
    ap.add_argument("--mode", choices=["layer", "head", "both"], default="both")
    ap.add_argument("--token_agg", choices=["response_mean", "response_last"],
                    default="response_mean")
    ap.add_argument("--system_prompt", default=None,
                    help="必須與要比較的 adapter 訓練時一致；主線 DPO 臂沒有 system prompt")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_pairs", type=int, default=None, help="只取前 N 筆（測試用）")
    main(ap.parse_args())
