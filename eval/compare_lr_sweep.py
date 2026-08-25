"""解析 train_prefix.py 的訓練 log，比較不同設定（主要用於 S0 的 lr sweep）。

為什麼需要：train_prefix.py 設了 report_to=[]，沒有接 wandb/tensorboard，訓練指標只會
以 dict 形式印到 stdout（logging_steps=5）。跑 sweep 時把 stdout tee 成檔案，再用本腳本
把多個 log 拉成一張對照表。

判讀重點（§9.1）：健康的偏好學習應該讓 margin 由「兩側共同貢獻」——rewards/chosen 上升、
rewards/rejected 下降。若只有 rejected 單邊下降而 chosen 平坦甚至下滑，代表模型只學會
「壓低某類輸出的機率」，沒學會「提高安全寫法的機率」；這正是 lr 過低導致 underfit 的
典型徵狀（也可能是目標函數本身的限制，所以才要掃 lr 把兩者分開）。

用法：
  python eval/compare_lr_sweep.py outputs/lr_sweep/*.log

  # 想順便輸出曲線資料做圖
  python eval/compare_lr_sweep.py outputs/lr_sweep/*.log --csv_dir outputs/lr_sweep/curves
"""
import argparse
import ast
import os
import re

# HF Trainer 把每次 log 的 dict 直接 print 出來，且不含巢狀大括號，
# 所以用「不含大括號的內容」抓整段最安全（tqdm 進度條可能與它同行）。
LOG_RE = re.compile(r"\{'(?:loss|train_runtime)'[^{}]*\}")

TRACK = ["loss", "grad_norm", "rewards/chosen", "rewards/rejected",
         "rewards/margins", "rewards/accuracies", "learning_rate"]


def parse_log(path):
    """回傳 (records, summary)。records 是每個 logging step 的 dict。"""
    text = open(path, errors="replace").read()
    records, summary = [], {}
    for m in LOG_RE.findall(text):
        try:
            d = ast.literal_eval(m)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(d, dict):
            continue
        if "train_runtime" in d:
            summary = d
        elif "loss" in d:
            records.append(d)
    return records, summary


def window(records, key, frac=0.2, tail=False):
    """取前段或後段窗口的平均。窗口至少 1 筆。"""
    vals = [r[key] for r in records if key in r and isinstance(r[key], (int, float))]
    if not vals:
        return None
    k = max(1, int(len(vals) * frac))
    seg = vals[-k:] if tail else vals[:k]
    return sum(seg) / len(seg)


def summarize(path):
    records, summary = parse_log(path)
    if not records:
        return None
    out = {
        "name": os.path.basename(path).replace(".log", ""),
        "n_logs": len(records),
        "runtime": summary.get("train_runtime"),
        "peak_lr": max((r.get("learning_rate", 0) for r in records), default=0),
        "nan": any(isinstance(r.get("loss"), float) and r["loss"] != r["loss"]
                   for r in records),
    }
    for key in TRACK:
        out[f"{key}.early"] = window(records, key, tail=False)
        out[f"{key}.late"] = window(records, key, tail=True)
    gn = [r["grad_norm"] for r in records
          if isinstance(r.get("grad_norm"), (int, float))]
    out["grad_max"] = max(gn) if gn else None
    out["records"] = records
    return out


def fmt(v, spec="{:.4f}"):
    return "—" if v is None else spec.format(v)


def delta(s, key):
    e, l = s.get(f"{key}.early"), s.get(f"{key}.late")
    return None if (e is None or l is None) else l - e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="train_prefix.py 的 stdout log 檔")
    ap.add_argument("--csv_dir", default=None, help="把每個 run 的曲線另存成 csv")
    args = ap.parse_args()

    runs = [s for s in (summarize(p) for p in args.logs) if s]
    runs.sort(key=lambda r: r["peak_lr"])   # 依 lr 由小到大排，方便看趨勢
    if not runs:
        print("沒有解析到任何訓練紀錄。確認 log 是 train_prefix.py 的 stdout。")
        return

    W = 16
    print("=" * (18 + W * len(runs)))
    print("【訓練動態對照】early = 前 20% 的平均，late = 後 20% 的平均")
    print("=" * (18 + W * len(runs)))
    hdr = f"{'':<18}" + "".join(f"{r['name']:>{W}}" for r in runs)
    print(hdr)
    print("-" * len(hdr))

    def row(label, fn, spec="{:.4f}"):
        print(f"{label:<18}" + "".join(f"{fmt(fn(r), spec):>{W}}" for r in runs))

    row("peak lr", lambda r: r["peak_lr"], "{:.1e}")
    row("logged steps", lambda r: r["n_logs"], "{:.0f}")
    print("-" * len(hdr))
    row("loss early", lambda r: r["loss.early"])
    row("loss late", lambda r: r["loss.late"])
    row("loss Δ", lambda r: delta(r, "loss"), "{:+.4f}")
    print("-" * len(hdr))
    row("chosen early", lambda r: r["rewards/chosen.early"])
    row("chosen late", lambda r: r["rewards/chosen.late"])
    row("chosen Δ  ↑好", lambda r: delta(r, "rewards/chosen"), "{:+.4f}")
    print("-" * len(hdr))
    row("rejected early", lambda r: r["rewards/rejected.early"])
    row("rejected late", lambda r: r["rewards/rejected.late"])
    row("rejected Δ ↓好", lambda r: delta(r, "rewards/rejected"), "{:+.4f}")
    print("-" * len(hdr))
    row("margins late", lambda r: r["rewards/margins.late"])
    row("accuracies late", lambda r: r["rewards/accuracies.late"])
    print("-" * len(hdr))
    row("grad_norm late", lambda r: r["grad_norm.late"], "{:.2f}")
    row("grad_norm max", lambda r: r["grad_max"], "{:.2f}")
    row("runtime (s)", lambda r: r["runtime"], "{:.0f}")

    print("\n" + "=" * 66)
    print("【判讀】")
    print("=" * 66)
    for r in runs:
        dc, dr = delta(r, "rewards/chosen"), delta(r, "rewards/rejected")
        notes = []
        if r["nan"]:
            notes.append("❌ loss 出現 NaN，此設定不可用")
        if r["grad_max"] is not None and r["grad_max"] > 50:
            notes.append(f"⚠️  grad_norm 峰值 {r['grad_max']:.0f} > 50，步長過大")
        if dc is not None and dr is not None:
            if dc > 0 and dr < 0:
                notes.append("✅ margin 由雙邊貢獻（chosen 上升、rejected 下降）")
            elif dc <= 0 and dr < 0:
                notes.append("⚠️  只有 rejected 單邊下降，chosen 未上升 → underfit 徵狀")
            elif dc > 0 and dr >= 0:
                notes.append("⚠️  chosen 上升但 rejected 未下降")
            else:
                notes.append("❌ 兩側都往錯誤方向移動")
        acc = r["rewards/accuracies.late"]
        if acc is not None:
            notes.append(f"{'✅' if acc > 0.75 else '⚠️ '} accuracies 收斂到 {acc:.3f}"
                         f"{'' if acc > 0.75 else '（目標 > 0.75）'}")
        print(f"\n{r['name']}（lr={r['peak_lr']:.1e}）")
        for n in notes:
            print(f"  {n}")

    print("\n" + "-" * 66)
    print("若四組都是「只有 rejected 單邊下降」→ 不是 lr 問題，走 §8.3 停損點：")
    print("  先試 num_virtual_tokens 16 → 40 → 64，再考慮 prefix_projection。")
    print("挑出最好的 2 組後，務必再跑一次小規模安全評測確認有反映到真實生成上。")

    if args.csv_dir:
        os.makedirs(args.csv_dir, exist_ok=True)
        for r in runs:
            path = os.path.join(args.csv_dir, f"{r['name']}.csv")
            keys = ["step"] + TRACK
            with open(path, "w") as f:
                f.write(",".join(keys) + "\n")
                for i, rec in enumerate(r["records"], 1):
                    f.write(",".join(str(rec.get(k, "")) if k != "step" else str(i)
                                     for k in keys) + "\n")
        print(f"\n曲線 csv 已存到 {args.csv_dir}/")


if __name__ == "__main__":
    main()
