"""§3.3 第 3 步：用 training dynamics 的相關係數挑 D_norm（influence selection）。

對應 ProSec `influence_score/sample_refactored.py` 的 `td_sample_plus()`。

論文 Eq.(5)(6)(7)：
    f(x_n, y_n, y_f, theta) =   r(x_n, y_n, theta)
    g(x_v, y_f, theta)      = - r(x_v, y_f, theta)
    Inf = corr(m_f, m_g)                      <- Kendall Tau，跨 T 個 checkpoint

然後：每個 D_sec 取相關係數最高的 top-2 個 D_norm 候選，
再把全域相關係數最低的 20% 丟掉（論文 §5：top-2 + discard least 20%）。

**設定來源**：論文本文沒寫死 corr 種類與方向，但作者上傳的最終資料集叫
`prosecalign/top2-cds-0.8-kendall-on-neg_if-corr-max-2`，照 sample_refactored.py
的命名規則 `{prefix}-cds-{ratio}-{corr_method}-{obj0}-{obj1}-corr-{flag}-{top_n}`
可以反推出：kendall / f=on / g=neg_if / flag=max / top_n=2 / downsample=0.8。
本腳本的預設值即為此組，`--f/--g` 可切到 Table 3 的其他變體。

配對關係（見 data/convert_prosec_to_pref.py 的說明）：
    D_norm 的 rejected == 它所屬 D_sec 的 chosen == 同一份 y_f 字串

用法：
  python data/select_dnorm.py \
      --train_file data/train_pref.jsonl \
      --dynamics_dir outputs/dynamics \
      --out data/train_pref_selected.jsonl
"""
import argparse
import json
import math
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd


def kendall_tau_b(X, Y):
    """逐列算 Kendall tau-b，回傳長度 N 的 array（分母為 0 時給 NaN）。

    自己實作而不是走 pandas 的 method='kendall'，因為那條路徑要 scipy；
    checkpoint 數 T 只有 10，O(T^2) 全展開比多一個相依划算。
    定義與 scipy.stats.kendalltau 相同（含平手修正）。
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    T = X.shape[1]
    iu = np.triu_indices(T, k=1)          # 只取 i<j 的配對
    dx = X[:, iu[0]] - X[:, iu[1]]
    dy = Y[:, iu[0]] - Y[:, iu[1]]
    num = (np.sign(dx) * np.sign(dy)).sum(axis=1)          # C - D
    n0 = T * (T - 1) / 2
    n1 = (dx == 0).sum(axis=1)            # X 的平手配對數
    n2 = (dy == 0).sum(axis=1)
    den = np.sqrt((n0 - n1) * (n0 - n2))
    with np.errstate(invalid="ignore", divide="ignore"):
        tau = np.where(den > 0, num / den, np.nan)
    return tau


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


def load_dynamics(dynamics_dir):
    """回傳 {checkpoint_id: {row_id: stats}}，依 checkpoint_id 排序。"""
    dyn = {}
    for name in sorted(os.listdir(dynamics_dir)):
        m = re.fullmatch(r"checkpoint-(\d+)\.jsonl", name)
        if not m:
            continue
        per_row = {}
        with open(os.path.join(dynamics_dir, name)) as f:
            for line in f:
                e = json.loads(line)
                per_row[e["id"]] = e
        dyn[int(m.group(1))] = per_row
    return dict(sorted(dyn.items()))


def traj(dyn, row_id, key):
    """把某一列在所有 checkpoint 上的某個量測值串成軌跡。缺任一點就回 None。"""
    out = []
    for per_row in dyn.values():
        rec = per_row.get(row_id)
        if rec is None or key not in rec:
            return None
        out.append(rec[key])
    return out


def build_measures(rows, dyn, f_kind, g_kind):
    """回傳 (dnorm_rows, parent_ids, m_f, m_g)，四者等長且一一對應。"""
    dsec = [e for e in rows if not e.get("benign", False)]
    dnorm = [e for e in rows if e.get("benign", False)]

    # y_f -> D_sec row。y_f 重複時沿用 ProSec 的作法：保留第一筆
    yf2sec = {}
    dup = 0
    for e in dsec:
        if e["chosen"] in yf2sec:
            dup += 1
            continue
        yf2sec[e["chosen"]] = e
    print(f"D_sec {len(dsec)} 筆（y_f 重複 {dup}）· D_norm {len(dnorm)} 筆")

    kept, parents, mf, mg = [], [], [], []
    missing_parent = missing_traj = 0
    n_ckpt = len(dyn)
    for e in dnorm:
        parent = yf2sec.get(e["rejected"])
        if parent is None:
            missing_parent += 1
            continue
        # f 的候選：on = r(x_n, y_n)；onof = r(x_n,y_n) - r(x_n,y_f)（Table 3 第二列）
        on = traj(dyn, e["id"], "chosen_ln_logp")
        if f_kind == "on":
            f_vec = on
        else:
            of = traj(dyn, e["id"], "rejected_ln_logp")
            f_vec = None if (on is None or of is None) else \
                [a - b for a, b in zip(on, of)]
        # g 的候選：neg_if = -r(x_v, y_f)；ofif = r(x_n,y_f) - r(x_v,y_f)；
        #           decrease = 單調遞減序列（Table 3 第三列）
        if g_kind == "decrease":
            g_vec = list(range(n_ckpt, 0, -1))
        else:
            iff = traj(dyn, parent["id"], "chosen_ln_logp")
            if g_kind == "neg_if":
                g_vec = None if iff is None else [-v for v in iff]
            else:
                of = traj(dyn, e["id"], "rejected_ln_logp")
                g_vec = None if (of is None or iff is None) else \
                    [a - b for a, b in zip(of, iff)]
        if f_vec is None or g_vec is None:
            missing_traj += 1
            continue
        kept.append(e)
        parents.append(parent["id"])
        mf.append(f_vec)
        mg.append(g_vec)

    if missing_parent:
        print(f"⚠️  {missing_parent} 筆 D_norm 找不到對應的 D_sec（y_f 對不上），已跳過")
    if missing_traj:
        print(f"⚠️  {missing_traj} 筆缺 checkpoint 量測值，已跳過")
    return kept, parents, mf, mg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--dynamics_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corr_method", default="kendall",
                    choices=["kendall", "spearman", "pearson"])
    ap.add_argument("--f", dest="f_kind", default="on", choices=["on", "onof"],
                    help="on=r(x_n,y_n)（論文最終設定）；onof=r(x_n,y_n)-r(x_n,y_f)")
    ap.add_argument("--g", dest="g_kind", default="neg_if",
                    choices=["neg_if", "ofif", "decrease"],
                    help="neg_if=-r(x_v,y_f)（論文最終設定）")
    ap.add_argument("--top_n", type=int, default=2,
                    help="每個 D_sec 取相關係數最高的幾個 D_norm（論文：2）")
    ap.add_argument("--downsample_ratio", type=float, default=0.8,
                    help="依相關係數排序後保留的比例（論文：丟掉最低的 20%%）")
    ap.add_argument("--stats_out", default=None, help="把每筆的 influence score 存成 jsonl")
    args = ap.parse_args()

    rows = load_rows(args.train_file)
    dyn = load_dynamics(args.dynamics_dir)
    if not dyn:
        raise SystemExit(f"{args.dynamics_dir} 沒有 checkpoint-*.jsonl")
    print(f"checkpoints：{list(dyn)}")
    if len(dyn) < 3:
        print("⚠️  checkpoint 數 < 3，rank correlation 幾乎沒有鑑別力")

    kept, parents, mf, mg = build_measures(rows, dyn, args.f_kind, args.g_kind)
    if not kept:
        raise SystemExit("沒有任何可計算的 D_norm，請檢查 benign 欄位與 y_f 配對")

    # 逐列相關係數（ProSec 用 pandas corrwith(axis=1)，這裡等價）
    if args.corr_method == "kendall":
        corr = pd.Series(kendall_tau_b(mf, mg))
    else:
        corr = pd.DataFrame(mf).corrwith(
            pd.DataFrame(mg), axis=1, method=args.corr_method)
    n_nan = int(corr.isna().sum())
    print(f"influence score：{len(corr)} 筆，NaN {n_nan}（填 0，同 ProSec 作法）")
    corr = corr.fillna(0.0)

    # 第一層：每個 D_sec 取 top-N
    by_parent = defaultdict(list)
    for local, pid in enumerate(parents):
        by_parent[pid].append(local)
    selected = []
    for pid, locals_ in by_parent.items():
        s = corr[locals_].nlargest(args.top_n)
        selected.extend(s.index.tolist())
    print(f"top-{args.top_n} 後：{len(selected)} 筆 D_norm"
          f"（來自 {len(by_parent)} 個 D_sec）")
    group_sizes = pd.Series([len(v) for v in by_parent.values()]).value_counts().sort_index()
    print(f"每個 D_sec 的候選數分布：{group_sizes.to_dict()}")
    if group_sizes.index.max() <= args.top_n:
        print(f"⚠️  沒有任何 D_sec 的候選數超過 top_n={args.top_n} —— "
              "top-N 這一層對這份資料是 no-op，實際只有 downsample 在起作用。"
              "見 docs/REPRO_PAPER.md「候選池已被壓平」")

    # 第二層：全域依相關係數排序，丟掉最低的 (1 - ratio)
    selected.sort(key=lambda local: corr[local], reverse=True)
    n_keep = math.ceil(len(selected) * args.downsample_ratio)
    dropped = selected[n_keep:]
    selected = selected[:n_keep]
    thr = f"{corr[selected[-1]]:.4f}" if selected else "n/a"
    print(f"downsample {args.downsample_ratio} 後：{len(selected)} 筆"
          f"（丟掉 {len(dropped)} 筆，corr 門檻 {thr}）")

    keep_ids = {kept[local]["id"] for local in selected}
    dsec_rows = [e for e in rows if not e.get("benign", False)]
    dnorm_rows = [kept[local] for local in sorted(selected)]

    with open(args.out, "w") as f:
        for e in dsec_rows + dnorm_rows:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"最終資料集：D_sec {len(dsec_rows)} + D_norm {len(dnorm_rows)} "
          f"= {len(dsec_rows) + len(dnorm_rows)} 筆 → {args.out}")
    print(f"D_norm / D_sec = {len(dnorm_rows) / max(len(dsec_rows), 1):.2f}"
          f"（論文上限 = top_n × ratio = {args.top_n * args.downsample_ratio:.2f}）")

    if args.stats_out:
        with open(args.stats_out, "w") as f:
            for local, e in enumerate(kept):
                f.write(json.dumps({
                    "id": e["id"], "parent_id": parents[local],
                    "influence": float(corr[local]),
                    "selected": e["id"] in keep_ids,
                    "lang": e.get("lang", ""), "cwe": e.get("cwe", ""),
                }) + "\n")
        print(f"influence score 明細 → {args.stats_out}")


if __name__ == "__main__":
    main()
