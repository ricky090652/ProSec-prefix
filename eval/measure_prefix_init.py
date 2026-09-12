"""量測零初始化 prefix 在 step 0 就造成多大的偏移。

為什麼要量：LoRA 的 B=0 是**精確的 no-op**（加上去的是 0）。但 prefix 的零初始化
不是——peft 把 `embedding.weight` 整個設 0，也就是 K 和 V 都是零向量
（`peft/tuners/prefix_tuning/model.py:79`；SVEN 的 `sven/model.py:18` 同樣是
`torch.zeros`）。V=0 確實不貢獻輸出，但 **K=0 讓 prefix 位置的 attention score
變成 0，在 softmax 裡拿到 exp(0)=1 的權重**，於是它們吸走一部分 attention mass。

代數上很乾淨：設真實 token 的分數為 s_j、Z = Σ_j exp(s_j)，則加了 nvt 個零 key 後

    w_j = exp(s_j) / (Z + nvt) = p_j · Z/(Z+nvt)
    attention 輸出 = α · (base 的 attention 輸出)，  α = Z/(Z+nvt) < 1

所以零初始化 prefix 等於把**每個 head、每個位置**的 attention 輸出乘上一個小於 1
的係數，而且 nvt 越大 α 越小。這解釋了「prefix 越長 utility 掉越多」。

⚠️ 這是 prefix-tuning 這個方法的性質，不是實作瑕疵——SVEN 也承受同一件事
（它選 nvt=5~12 正是把它壓住）。所以量到的數字是**機制證據**，不是要撤回的理由。

量三件事：
  1. absorbed  prefix 位置吸走的 attention mass（= 1 − α），直接把 prefix 模型的
     attention 權重前 nvt 欄加總，不需要推導
  2. KL        base 與 prefix 在下一個 token 分布上的 KL
  3. hidden    最後一層 hidden state 的相對變化

用法（在有 GPU 的機器上，幾分鐘）：
  python eval/measure_prefix_init.py --nvt 8 16 64 --json_out outputs/prefix_init.json
"""
import argparse
import json

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--nvt", type=int, nargs="+", default=[8, 16, 64])
    ap.add_argument("--n_prompts", type=int, default=20,
                    help="取幾題 HumanEval 的 prompt 當輸入（真實的評測輸入）")
    ap.add_argument("--lora_targets", default="qkv_proj,o_proj,gate_up_proj,down_proj",
                    help="LoRA 對照掛哪些模組（Phi-3 的 all-linear）")
    ap.add_argument("--no_lora_control", action="store_true",
                    help="跳過 LoRA 零初始化的對照量測")
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PrefixTuningConfig, TaskType, get_peft_model

    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        from datasets import load_dataset
        prompts = [r["prompt"] for r in
                   load_dataset("openai_humaneval", split="test")][: args.n_prompts]
    except Exception as e:
        print(f"讀不到 HumanEval（{e}），改用內建 prompt")
        prompts = ["def read_file(path):\n    ", "import sqlite3\n\ndef q(name):\n    "]

    def load():
        m = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="eager")   # output_attentions 需要 eager
        m.eval()
        return m

    enc = [tok(p, return_tensors="pt", add_special_tokens=False) for p in prompts]

    print("計算 base 的參考輸出 …")
    base = load()
    dev = next(base.parameters()).device
    base_ref = []
    with torch.no_grad():
        for e in enc:
            o = base(input_ids=e.input_ids.to(dev),
                     attention_mask=e.attention_mask.to(dev),
                     output_hidden_states=True)
            base_ref.append((o.logits[0, -1].float().log_softmax(-1).cpu(),
                             o.hidden_states[-1][0, -1].float().cpu()))
    del base
    torch.cuda.empty_cache()

    rows = []
    for nvt in args.nvt:
        print(f"\n=== nvt={nvt} ===")
        m = load()
        m = get_peft_model(m, PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM, num_virtual_tokens=nvt,
            init_weights="zero"))
        dev = next(m.parameters()).device
        absorbed_last, kls, hdev, prof = [], [], [], {}
        with torch.no_grad():
            for k, (e, (blp, bh)) in enumerate(zip(enc, base_ref)):
                o = m(input_ids=e.input_ids.to(dev),
                      attention_mask=e.attention_mask.to(dev),
                      output_attentions=True, output_hidden_states=True)
                # attn: [B, H, q_len, nvt + seq] —— 前 nvt 欄就是 prefix 吸走的部分
                per_layer = [a[0, :, :, :nvt].sum(-1).float().cpu() for a in o.attentions]
                A = torch.stack(per_layer)                 # [L, H, q_len]
                absorbed_last.append(A[:, :, -1].mean().item())
                if k == 0:                                  # 看它隨 query 位置怎麼變
                    q = A.shape[-1]
                    for pos in sorted({0, min(7, q-1), min(31, q-1), q-1}):
                        prof[pos + 1] = A[:, :, pos].mean().item()
                plp = o.logits[0, -1].float().log_softmax(-1).cpu()
                kls.append(torch.nn.functional.kl_div(plp, blp, log_target=True,
                                                      reduction="sum").item())
                h = o.hidden_states[-1][0, -1].float().cpu()
                hdev.append(((h - bh).norm() / bh.norm()).item())
        avg = lambda v: sum(v) / len(v)
        r = {"nvt": nvt, "absorbed": avg(absorbed_last), "kl": avg(kls),
             "hidden_rel": avg(hdev), "by_context_len": prof}
        rows.append(r)
        print(f"  吸走的 attention mass（最後位置、跨層平均）：{r['absorbed']:.1%}")
        print(f"  KL(base‖prefix) 下一個 token：{r['kl']:.3f}")
        print(f"  最後一層 hidden 相對變化：{r['hidden_rel']:.1%}")
        print("  依 context 長度（第一題）：" +
              "  ".join(f"{c} tok → {v:.1%}" for c, v in sorted(prof.items())))
        del m
        torch.cuda.empty_cache()

    # LoRA 零初始化的對照。硬寫 0 是不誠實的——實際量一次。它應該精確等於 base
    # （B=0 → ΔW=0），所以這一列同時也是整支腳本的 sanity check：量出來不是 ~0
    # 就代表量測管線本身有問題，prefix 的數字也不能信。
    lora_row = None
    if not args.no_lora_control:
        print("\n=== LoRA 對照（零初始化，應精確等於 base）===")
        from peft import LoraConfig
        m = get_peft_model(load(), LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
            target_modules=[t.strip() for t in args.lora_targets.split(",") if t.strip()]))
        dev = next(m.parameters()).device
        kls, hdev = [], []
        with torch.no_grad():
            for e, (blp, bh) in zip(enc, base_ref):
                o = m(input_ids=e.input_ids.to(dev),
                      attention_mask=e.attention_mask.to(dev), output_hidden_states=True)
                plp = o.logits[0, -1].float().log_softmax(-1).cpu()
                kls.append(torch.nn.functional.kl_div(plp, blp, log_target=True,
                                                      reduction="sum").item())
                h = o.hidden_states[-1][0, -1].float().cpu()
                hdev.append(((h - bh).norm() / bh.norm()).item())
        lora_row = {"kl": sum(kls) / len(kls), "hidden_rel": sum(hdev) / len(hdev)}
        print(f"  KL={lora_row['kl']:.2e}   hidden 相對變化={lora_row['hidden_rel']:.2e}")
        del m
        torch.cuda.empty_cache()

    print("\n" + "=" * 68)
    print(f"{'nvt':>5}{'吸走 attention':>16}{'KL':>10}{'hidden 變化':>14}")
    for r in rows:
        print(f"{r['nvt']:>5}{r['absorbed']:>15.1%}{r['kl']:>10.3f}{r['hidden_rel']:>13.1%}")
    if lora_row:
        print(f"{'LoRA':>5}{'—':>16}{lora_row['kl']:>10.1e}{lora_row['hidden_rel']:>13.1e}"
              "   ← 實測的 no-op 對照（無 prefix 位置，故無吸收量）")
    print("\n註：實際評測時 context 是完整的指令（上百個 token），"
          "所以要看長 context 那一欄，不是 1 token 的極端值。")

    if args.json_out:
        json.dump({"prefix": rows, "lora_control": lora_row},
                  open(args.json_out, "w"), indent=2)
        print(f"\n→ {args.json_out}")


if __name__ == "__main__":
    main()
