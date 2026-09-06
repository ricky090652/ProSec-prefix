"""核對各臂的 adapter 設定與訓練參數是否如預期、彼此是否只差一個變數。

在花數十小時跑全量評測之前，先確認每個 adapter 真的是它宣稱的東西。
會檢查的事：

  1. adapter_config.json 的 peft_type / r / alpha / target_modules / num_virtual_tokens
  2. 由設定推算的可訓練參數量（並與 safetensors 實際存的張量比對）
  3. training_args.bin 裡的 lr / beta / max_steps / batch / grad_accum / seed / max_length
  4. 跨臂一致性：除了 PEFT 方法與各自的 lr，其餘應完全相同

用法：
  python scripts/verify_arms.py outputs/dpo-arms/lora outputs/dpo-arms/lora_qkv \\
      outputs/dpo-arms/prefix outputs/dpo-arms/prefix_nvt8
"""
import argparse
import json
import os

# Phi-3-mini-4k-instruct
H, L, INTER, TOTAL = 3072, 32, 8192, 3_821_079_552


def expected_params(cfg):
    """由 adapter_config 推算可訓練參數量。"""
    t = cfg.get("peft_type")
    if t == "PREFIX_TUNING":
        nvt = cfg["num_virtual_tokens"]
        if cfg.get("prefix_projection"):
            return None          # 走 MLP 重參數化，公式不同
        return nvt * L * 2 * H
    if t == "LORA":
        r = cfg["r"]
        dims = {"qkv_proj": (H, 3 * H), "o_proj": (H, H),
                "gate_up_proj": (H, 2 * INTER), "down_proj": (INTER, H),
                "q_proj": (H, H), "k_proj": (H, H), "v_proj": (H, H)}
        tgt = cfg.get("target_modules") or []
        if isinstance(tgt, str):
            tgt = [tgt]
        if any(m not in dims for m in tgt):
            return None
        return sum(r * (dims[m][0] + dims[m][1]) for m in tgt) * L
    return None


def actual_params(path):
    """讀 safetensors / bin 的實際張量元素數。"""
    st = os.path.join(path, "adapter_model.safetensors")
    if os.path.exists(st):
        try:
            from safetensors import safe_open
            n = 0
            with safe_open(st, framework="pt") as f:
                for k in f.keys():
                    sh = f.get_slice(k).get_shape()
                    p = 1
                    for d in sh:
                        p *= d
                    n += p
            return n
        except Exception as e:
            return f"讀取失敗：{e}"
    return None


def training_args(path):
    """從 checkpoint 的 training_args.bin 取出關鍵超參數。"""
    for root in (path, *(os.path.join(path, d) for d in sorted(os.listdir(path))
                         if d.startswith("checkpoint-"))):
        f = os.path.join(root, "training_args.bin")
        if not os.path.exists(f):
            continue
        try:
            import torch
            a = torch.load(f, weights_only=False)
        except Exception:
            continue
        return {k: getattr(a, k, None) for k in
                ("learning_rate", "beta", "max_steps", "per_device_train_batch_size",
                 "gradient_accumulation_steps", "seed", "max_length",
                 "max_prompt_length", "max_grad_norm", "lr_scheduler_type",
                 "warmup_ratio", "loss_type")}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    args = ap.parse_args()

    rows = []
    for path in args.arms:
        cfg_file = os.path.join(path, "adapter_config.json")
        if not os.path.exists(cfg_file):
            print(f"⚠️  {path}：找不到 adapter_config.json，跳過")
            continue
        cfg = json.load(open(cfg_file))
        exp, act = expected_params(cfg), actual_params(path)
        rows.append({"path": path, "cfg": cfg, "expected": exp, "actual": act,
                     "targs": training_args(path)})

    print("=" * 100)
    print("【adapter 設定】")
    print("=" * 100)
    hdr = f"{'臂':<28}{'PEFT':<9}{'結構':<34}{'推算參數':>12}{'佔模型':>9}  實測"
    print(hdr)
    for r in rows:
        c = r["cfg"]
        if c["peft_type"] == "LORA":
            tgt = c.get("target_modules")
            tgt = sorted(tgt) if isinstance(tgt, (list, set)) else [tgt]
            struct = f"r={c['r']} a={c['lora_alpha']} {','.join(tgt)}"
        else:
            struct = (f"nvt={c['num_virtual_tokens']} "
                      f"proj={c.get('prefix_projection')}")
        exp = r["expected"]
        e = f"{exp:>12,}" if exp else f"{'?':>12}"
        pct = f"{exp/TOTAL:>8.4%}" if exp else f"{'?':>9}"
        act = r["actual"]
        ok = "✅ 相符" if (exp and isinstance(act, int) and act == exp) else \
             (f"⚠️ {act:,}" if isinstance(act, int) else f"⚠️ {act}")
        print(f"{os.path.basename(r['path']):<28}{c['peft_type']:<9}{struct:<34}{e}{pct}  {ok}")

    print("\n" + "=" * 100)
    print("【訓練超參數】除了 lr 之外，同一列的值應該全部相同")
    print("=" * 100)
    keys = ["learning_rate", "beta", "max_steps", "per_device_train_batch_size",
            "gradient_accumulation_steps", "max_length", "max_prompt_length",
            "max_grad_norm", "lr_scheduler_type", "warmup_ratio", "seed"]
    names = [os.path.basename(r["path"]) for r in rows]
    w = max(16, max((len(n) for n in names), default=16) + 2)
    print(f"{'':<30}" + "".join(f"{n:>{w}}" for n in names))
    ALLOWED_DIFF = {"learning_rate"}
    problems = []
    for k in keys:
        vals = [r["targs"].get(k) for r in rows]
        show = ["—" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))
                for v in vals]
        uniq = {s for s in show if s != "—"}
        flag = ""
        if len(uniq) > 1 and k not in ALLOWED_DIFF:
            flag = "  ❌ 不一致"
            problems.append(k)
        elif len(uniq) > 1:
            flag = "  (允許不同)"
        print(f"{k:<30}" + "".join(f"{s:>{w}}" for s in show) + flag)

    eff = [(r["targs"].get("per_device_train_batch_size") or 0) *
           (r["targs"].get("gradient_accumulation_steps") or 0) for r in rows]
    print(f"{'→ effective batch':<30}" + "".join(f"{e:>{w}}" for e in eff)
          + ("  ❌ 不一致" if len(set(eff)) > 1 else ""))

    n_with_args = sum(1 for r in rows if r["targs"])
    print("\n" + "=" * 100)
    if n_with_args == 0:
        print("⚠️  完全讀不到任何 training_args.bin —— **無法驗證超參數是否一致**。")
        print("   （adapter 目錄或其 checkpoint-* 底下應該要有這個檔）")
    elif n_with_args < len(rows):
        missing = [os.path.basename(r["path"]) for r in rows if not r["targs"]]
        print(f"⚠️  這些臂讀不到 training_args.bin，未納入一致性檢查：{', '.join(missing)}")
    if problems:
        print(f"❌ 以下超參數跨臂不一致，對照不成立：{', '.join(problems)}")
    elif n_with_args >= 2:
        print(f"✅ 除 lr 之外所有訓練超參數一致（{n_with_args} 臂），"
              "只差 PEFT 方法 —— 對照成立")
    bad = [r for r in rows if r["expected"] and isinstance(r["actual"], int)
           and r["actual"] != r["expected"]]
    if bad:
        print("❌ 有臂的實際參數量與設定推算不符：" +
              ", ".join(os.path.basename(r["path"]) for r in bad))
    else:
        print("✅ 所有臂的實際參數量與設定推算相符")


if __name__ == "__main__":
    main()
