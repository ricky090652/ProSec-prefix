"""把撞到 max_new_tokens 的回覆從 ON/OFF 兩份裡**成對**移除。

為什麼需要：被長度上限切掉的程式碼可能正好少了含漏洞的那一段，偵測器掃不到，
於是被算成「安全」。ON 若寫得比 OFF 長，截斷率就會高得多（實測 14.4% vs 1.2%），
漏洞率的改善會有一部分是這樣來的，不是真的變安全。

為什麼要成對：只濾 ON 的話，等於拿「ON 寫得比較短、比較簡單的那些題」
去比 OFF 的全集，比較基礎就歪了。所以**任一邊截斷就兩邊一起丟**，
留下的永遠是同一組 (題目, 樣本索引)。

流程位置：gen_for_icd.py → **本腳本** → normalize_responses.py → detect_all.py → score_detected.py

用法：
  python eval/filter_truncated.py \
      --on  outputs/screen2_lora.on.jsonl \
      --off outputs/screen2_lora.off.jsonl \
      --suffix .untrunc
  # 產生 outputs/screen2_lora.on.untrunc.jsonl 與 .off.untrunc.jsonl
"""
import argparse
import json


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def out_path(path, suffix):
    # foo.on.jsonl -> foo.on<suffix>.jsonl
    return path[:-len(".jsonl")] + suffix + ".jsonl" if path.endswith(".jsonl") \
        else path + suffix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", required=True)
    ap.add_argument("--off", required=True)
    ap.add_argument("--suffix", default=".untrunc")
    args = ap.parse_args()

    on, off = load(args.on), load(args.off)
    if len(on) != len(off):
        raise SystemExit(f"ON/OFF 行數不同（{len(on)} vs {len(off)}），"
                         "兩份必須來自同一次 gen_for_icd.py 執行")
    if not any("truncated" in e for e in on):
        raise SystemExit(
            "輸入沒有 truncated 欄位 —— 這是舊版 gen_for_icd.py 的輸出。"
            "請用新版重新生成（它會逐筆記錄有沒有撞到 max_new_tokens）。")

    kept = dropped = 0
    f_on = open(out_path(args.on, args.suffix), "w")
    f_off = open(out_path(args.off, args.suffix), "w")
    try:
        for a, b in zip(on, off):
            ta = a.get("truncated") or [False] * len(a.get("responses", []))
            tb = b.get("truncated") or [False] * len(b.get("responses", []))
            if a.get("prompt") != b.get("prompt"):
                raise SystemExit("ON/OFF 的題目順序對不上，兩份必須來自同一次執行")
            keep = [i for i in range(len(a["responses"]))
                    if not (ta[i] if i < len(ta) else False)
                    and not (tb[i] if i < len(tb) else False)]
            dropped += len(a["responses"]) - len(keep)
            kept += len(keep)
            for src, fh in ((a, f_on), (b, f_off)):
                rec = dict(src)
                rec["responses"] = [src["responses"][i] for i in keep]
                rec["truncated"] = [False] * len(keep)
                rec["n_dropped_truncated"] = len(src["responses"]) - len(keep)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        f_on.close()
        f_off.close()

    total = kept + dropped
    print(f"保留 {kept} / {total} 組回覆（成對丟掉 {dropped}，"
          f"{100.0 * dropped / total if total else 0:.1f}%）")
    print(f"→ {out_path(args.on, args.suffix)}")
    print(f"→ {out_path(args.off, args.suffix)}")
    if total and dropped / total > 0.3:
        print("⚠️  丟掉超過 30%，剩下的樣本可能已經有選擇偏誤；"
              "應該調高 --max_new_tokens 重新生成，而不是靠這支腳本補救")


if __name__ == "__main__":
    main()
