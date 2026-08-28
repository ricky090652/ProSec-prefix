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

  # 忘了 tee 的話，直接讀 checkpoint 裡的 trainer_state.json
  python eval/compare_lr_sweep.py outputs/repro-paper/lora-simpo/checkpoint-1500/trainer_state.json

  # 想順便輸出曲線資料做圖
  python eval/compare_lr_sweep.py outputs/lr_sweep/*.log --csv_dir outputs/lr_sweep/curves
"""
import argparse
import ast
import json
import os
import re

# HF Trainer 把每次 log 的 dict 直接 print 出來，且不含巢狀大括號，
# 所以用「不含大括號的內容」抓整段最安全（tqdm 進度條可能與它同行）。
LOG_RE = re.compile(r"\{'(?:loss|train_runtime)'[^{}]*\}")

TRACK = ["loss", "grad_norm", "rewards/chosen", "rewards/rejected",
         "rewards/margins", "rewards/accuracies", "learning_rate"]


def parse_log(path):
    """回傳 (records, summary)。records 是每個 logging step 的 dict。

    接受兩種輸入：train_prefix.py 的 stdout log，或 checkpoint 裡的
    trainer_state.json（忘了 tee 時的救援管道，內容是同一批 dict）。
    """
    if path.endswith(".json"):
        state = json.load(open(path))
        records = [r for r in state.get("log_history", []) if "loss" in r]
        summary = next((r for r in state.get("log_history", [])
                        if "train_runtime" in r), {})
        return records, summary
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
        # CPOTrainer(SimPO) 的 log 多一個 nll_loss 欄位，DPOTrainer 沒有 → 用來自動辨識
        "objective": "simpo" if any("nll_loss" in r for r in records) else "dpo",
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
    ap.add_argument("logs", nargs="+",
                    help="train_prefix.py 的 stdout log 檔，或 checkpoint 的 trainer_state.json")
    ap.add_argument("--csv_dir", default=None, help="把每個 run 的曲線另存成 csv")
    ap.add_argument("--collapse_min_abs", type=float, default=1.0,
                    help="崩潰判定的最小絕對變動量，避免小尺度 reward 被比值誤判")
    ap.add_argument("--collapse_ratio", type=float, default=2.5,
                    help="chosen reward 絕對值從 early 到 late 放大超過此倍數 → 判定崩潰")
    ap.add_argument("--flat_rel", type=float, default=0.05,
                    help="margin 相對成長低於此比例 → 視為 underfit（尺度無關）")
    ap.add_argument("--flat_abs", type=float, default=0.05,
                    help="flat 判定的絕對下限，避免 early 接近 0 時失效")
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

    print(f"{'objective':<18}" + "".join(f"{r['objective']:>{W}}" for r in runs))
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
    row("margins early", lambda r: r["rewards/margins.early"])
    row("margins late", lambda r: r["rewards/margins.late"])
    row("margins Δ ↑好", lambda r: delta(r, "rewards/margins"), "{:+.4f}")
    row("margin 成長 %", lambda r: (100 * delta(r, "rewards/margins")
                                    / abs(r["rewards/margins.early"])
                                    if r["rewards/margins.early"] else None), "{:+.1f}")
    row("accuracies early", lambda r: r["rewards/accuracies.early"])
    row("accuracies late", lambda r: r["rewards/accuracies.late"])
    print("-" * len(hdr))
    row("grad_norm late", lambda r: r["grad_norm.late"], "{:.2f}")
    row("grad_norm max", lambda r: r["grad_max"], "{:.2f}")
    row("runtime (s)", lambda r: r["runtime"], "{:.0f}")

    print("\n" + "=" * 66)
    print("【判讀】")
    print("=" * 66)
    n_underfit = n_collapse = 0
    for r in runs:
        dc, dr = delta(r, "rewards/chosen"), delta(r, "rewards/rejected")
        cl, rl = r["rewards/chosen.late"], r["rewards/rejected.late"]
        notes, state = [], None
        if r["nan"]:
            notes.append("❌ loss 出現 NaN，此設定不可用")

        # 偏好學習唯一真正要問的問題：margin 有沒有變大、排序有沒有變準。
        # 這兩項比 chosen/rejected 各自的走向更根本——兩側怎麼移動都好，
        # 只要 margin 沒擴大，就代表模型沒學到「哪一個比較好」。
        dm = delta(r, "rewards/margins")
        acc, acc_e = r["rewards/accuracies.late"], r["rewards/accuracies.early"]
        if dm is not None and dm <= 0:
            notes.append(f"❌ margin 沒有擴大（Δ {dm:+.4f}）→ 模型沒學到排序，"
                         "這不是訓練不足，是訓練訊號本身有問題")
        # accuracies < 0.5 本身不必然是壞事：若起點就 < 0.5，那是資料與 base model
        # 的性質（ProSec 的 y_f 是在「看過漏洞碼與分析器回饋」的條件下生成的，
        # 對 x_v 而言本來就偏 off-policy），要看的是它有沒有在往上走。
        if acc is not None and acc < 0.5:
            if acc_e is not None and acc_e < 0.5:
                arrow = "↑ 改善中" if acc > acc_e else "↓ 惡化"
                notes.append(f"ℹ️  accuracies {acc_e:.3f} → {acc:.3f}（{arrow}）。"
                             "起點就 < 0.5 是資料性質，非訓練失敗；看趨勢不看絕對值")
            else:
                notes.append(f"❌ accuracies 由 {acc_e:.3f} 掉到 {acc:.3f} < 0.5：訓練讓排序變差")

        # 診斷順序很重要：先判 reward 崩潰，再判 underfit。
        # 兩者的 Δ 符號可能相同（都是 chosen 下降），但成因與處置完全相反——
        # 崩潰是步長過大讓 policy 跑掉，underfit 是根本沒在學。
        #
        # 門檻一律用「相對於該 run 自身 reward 尺度」的比值，不用絕對值：
        # DPO 的 reward 是未歸一化的 log-ratio（實測起點約 -1.5），
        # SimPO 是 beta x 每 token 平均 log-prob（實測起點約 -10），兩者差一個量級。
        ce, re_ = r["rewards/chosen.early"], r["rewards/rejected.early"]
        me = r["rewards/margins.early"]
        # margin 的相對成長 = (late - early) / |early|。
        # 這是唯一能跨 objective 比較的「有沒有在學」指標：
        # chosen / rejected 各自的絕對位移會隨損失函數與資料尺度大幅改變，
        # 但「兩者的差距相對於起點擴大了幾成」在哪裡都是同一件事。
        mgrow = (dm / abs(me)) if (dm is not None and me not in (None, 0)) else None

        # 崩潰必須同時滿足「相對放大」與「絕對變動夠大」。只看比值會誤判——
        # SimPO 的 reward 起點本來就小（β x 每 token 平均 log-prob，約 -0.5），
        # 從 -0.5 掉到 -1.3 是 2.6x 但每 token 機率仍有 ~41%，完全正常。
        grew = (abs(cl) / abs(ce)) if (cl is not None and ce not in (None, 0)) else None
        if (grew is not None and grew > args.collapse_ratio
                and dc is not None and abs(dc) > args.collapse_min_abs):
            state = "collapse"
            n_collapse += 1
            notes.append(f"❌ reward 崩潰：chosen 從 {ce:.2f} 掉到 {cl:.2f}"
                         f"（放大 {grew:.1f}x 且變動 {dc:+.2f}）。"
                         "連安全碼的機率都被壓垮，不是「學得更好」")
        elif mgrow is not None and mgrow < args.flat_rel:
            state = "underfit"
            n_underfit += 1
            notes.append(f"⚠️  margin 相對成長僅 {100*mgrow:.1f}%"
                         f"（門檻 {100*args.flat_rel:.0f}%）→ underfit")
        elif dc is not None and dr is not None:
            if dc > 0 and dr < 0:
                state = "healthy"
                notes.append("✅ margin 由雙邊貢獻（chosen 上升、rejected 下降）")
            elif dc <= 0 and dr < 0:
                notes.append("⚠️  chosen 下降、rejected 下降更多 → 偏向壓低機率而非學會安全寫法")
            elif dc > 0 and dr >= 0:
                notes.append("⚠️  chosen 上升但 rejected 未下降")
            else:
                notes.append("❌ 兩側都往錯誤方向移動")

        if r["grad_max"] is not None and r["grad_max"] > 50:
            notes.append(f"⚠️  grad_norm 峰值 {r['grad_max']:.0f} > 50（注意這是裁剪前的值）")

        if acc is not None and acc >= 0.5:
            if state == "collapse":
                # accuracies 只看 margin 的正負號。兩側一起崩、只是速度不同時，
                # 它會逼近 1.0 卻毫無意義，不能拿來當「訓練成功」的證據。
                notes.append(f"⚠️  accuracies {acc:.3f} 在 reward 崩潰下不具參考價值")
            else:
                notes.append(f"{'✅' if acc > 0.75 else '⚠️ '} accuracies 收斂到 {acc:.3f}"
                             f"{'' if acc > 0.75 else '（目標 > 0.75）'}")
        print(f"\n{r['name']}（lr={r['peak_lr']:.1e}）")
        for n in notes:
            print(f"  {n}")

    print("\n" + "-" * 66)
    if n_underfit == len(runs):
        print("全部 underfit（margin 幾乎沒成長）→ 才走 §8.3 停損點：")
        print("  先試 num_virtual_tokens 16 → 40 → 64，再考慮 prefix_projection。")
    elif all(delta(r, "rewards/margins") or 0 > 0 for r in runs):
        best = max(runs, key=lambda r: delta(r, "rewards/margins") or 0)
        print(f"margin 全部在成長，最快的是 {best['name']}"
              f"（lr={best['peak_lr']:.1e}）。若成長隨 lr 單調上升且未見不穩，"
              "代表還沒到上限，可以再往上探一級。")
    elif n_collapse and n_underfit:
        print("低 lr underfit、高 lr reward 崩潰 → 可用區間夾在兩者之間。")
        print("  挑落在中間、且 margin 由雙邊貢獻的那組。")
    print("訓練曲線好看不等於生成有進步：挑出最好的 2 組後，")
    print("務必跑小規模安全評測 + check_degeneration.py 確認。")
    if any(r["objective"] != runs[0]["objective"] for r in runs):
        print("⚠️  這批 log 混了不同 objective，reward 不可跨組直接比大小，"
              "只能比「形狀」（是否雙邊貢獻）。")

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
