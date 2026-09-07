"""把訓練曲線畫成圖，多個 run 疊在同一張比較。

吃的輸入與 eval/compare_lr_sweep.py 相同（共用它的 parser）：
train_prefix.py 的 stdout log，或 checkpoint 裡的 trainer_state.json。

畫六格：loss / rewards 的 chosen、rejected、margins / accuracies / grad_norm。
grad_norm 用對數軸——實測範圍從個位數到 92,320，線性軸會把所有細節壓成一條線。

用法：
  python eval/plot_training.py outputs/dpo-arms/*.log --out outputs/training_curves.png

  # 只畫 prefix 的三個尺寸，並輸出資料
  python eval/plot_training.py outputs/dpo-arms/prefix{,_nvt8,_nvt64}.log \\
      --out outputs/prefix_sizes.png --csv_dir outputs/curves
"""
import argparse
import csv
import importlib.util
import os
import sys

_spec = importlib.util.spec_from_file_location(
    "_cls", os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_lr_sweep.py"))
_cls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cls)
parse_log = _cls.parse_log

# (metric key, 標題, 是否對數軸, 越大越好?)
# 標題一律用英文：matplotlib 的預設字體沒有 CJK 字形，中文會變成方塊，
# 而且這張圖是要直接放進論文的。
PANELS = [
    ("loss",                "loss",                            False),
    ("rewards/margins",     "rewards/margins  (higher better)", False),
    ("rewards/accuracies",  "rewards/accuracies (higher better)", False),
    ("rewards/chosen",      "rewards/chosen  (higher better)", False),
    ("rewards/rejected",    "rewards/rejected  (lower better)", False),
    ("grad_norm",           "grad_norm (pre-clip, log scale)",  True),
]


def label_of(path):
    b = os.path.basename(path)
    for suf in (".log", ".json"):
        if b.endswith(suf):
            b = b[: -len(suf)]
    # checkpoint-800/trainer_state.json → 用上兩層目錄名
    if b == "trainer_state":
        parts = os.path.normpath(path).split(os.sep)
        b = parts[-3] if len(parts) >= 3 else b
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="train_prefix.py 的 log 或 trainer_state.json")
    ap.add_argument("--out", default="outputs/training_curves.png")
    ap.add_argument("--csv_dir", default=None, help="另外把每個 run 的曲線存成 csv")
    ap.add_argument("--smooth", type=int, default=1,
                    help="移動平均的視窗（記錄點數）。1=不平滑")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")           # 無頭環境，不需要 display
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("需要 matplotlib：pip install matplotlib")

    runs = []
    for path in args.logs:
        records, _ = parse_log(path)
        if not records:
            print(f"⚠️  {path} 解析不到訓練記錄，略過")
            continue
        runs.append((label_of(path), records))
    if not runs:
        sys.exit("沒有可畫的資料")

    def series(records, key):
        xs, ys = [], []
        for r in records:
            if key in r and isinstance(r[key], (int, float)):
                xs.append(r.get("step", len(xs) + 1))
                ys.append(r[key])
        if args.smooth > 1 and len(ys) >= args.smooth:
            w = args.smooth
            ys = [sum(ys[max(0, i - w + 1): i + 1]) / len(ys[max(0, i - w + 1): i + 1])
                  for i in range(len(ys))]
        return xs, ys

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (key, title, logscale) in zip(axes.ravel(), PANELS):
        drew = False
        for name, records in runs:
            xs, ys = series(records, key)
            if not xs:
                continue
            if logscale:
                # 對數軸不能有 <= 0；grad_norm 恆正，但保險起見過濾
                pts = [(x, y) for x, y in zip(xs, ys) if y > 0]
                if not pts:
                    continue
                xs, ys = zip(*pts)
            ax.plot(xs, ys, label=name, linewidth=1.4)
            drew = True
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.3)
        if logscale:
            ax.set_yscale("log")
        if key == "rewards/accuracies":
            # 0.5 = 隨機。低於它代表排序比丟銅板還差
            ax.axhline(0.5, color="grey", linestyle=":", linewidth=1)
        if key.startswith("rewards/") and key != "rewards/accuracies":
            # 0 = 與 base model 相同
            ax.axhline(0.0, color="grey", linestyle=":", linewidth=1)
        if not drew:
            ax.text(0.5, 0.5, f"(no {key})", ha="center", va="center",
                    transform=ax.transAxes, color="grey")
    axes.ravel()[0].legend(fontsize=9)
    fig.suptitle("Training dynamics", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"圖已存到 {args.out}（{len(runs)} 個 run）")

    if args.csv_dir:
        os.makedirs(args.csv_dir, exist_ok=True)
        keys = [k for k, *_ in PANELS]
        for name, records in runs:
            f = os.path.join(args.csv_dir, f"{name}.csv")
            with open(f, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["step"] + keys)
                for r in records:
                    w.writerow([r.get("step", "")] + [r.get(k, "") for k in keys])
            print(f"  csv → {f}")


if __name__ == "__main__":
    main()
