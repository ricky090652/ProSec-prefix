"""S5-overlap：量測 ProSec 偏好對的 diff 到底是不是「安全修補」。

masked DPO（EXPERIMENTS.md S5）的前提是：把 loss 限制在 chosen/rejected 的
相異 token 上，梯度就會集中在安全修補本身，不再去推那些兩邊共有、
本來就正確的程式碼。這個前提有一個可以直接證偽的地方——

    如果 diff 裡大多數 hunk 只是「換個寫法」的無關重寫，
    masking 圈到的就是雜訊，S5 應該直接放棄。

data/edit_locality.py 已經量到 token 層級的 ρ 中位數 0.358、平均 17 個 hunk、
只有 0.1% 是單一連續 diff——散得很開。散開本身不是問題（一個修補可以同時動
import、宣告、呼叫三處），問題是「這 17 個 hunk 有幾個壓在 analyzer 標記的
漏洞行上」。這支腳本就是量這件事。

分兩段：

  A 段（永遠會跑，純 CPU，不需要 analyzer）
     行層級的 diff 統計：hunk 數、被改動的行佔比、hunk 的散佈範圍。
     和 edit_locality.py 的 token 層級互補——行層級才對得上 analyzer 的行號。

  B 段（要 --detected）
     把 analyzer 在 rejected（漏洞版）上標到的行號，和 A 段的 hunk 對照：
       covered   有多少 finding 落在某個 hunk 內  → masking 有沒有漏掉修補
       precision 有多少 hunk 內含 finding          → masking 有沒有圈到雜訊
     **precision 是決策變數。** 它低就代表 diff 大半與漏洞無關。

B 段要的 analyzer 輸出得自己先產：

  python data/diff_overlap.py --train_file data/train_pref.jsonl \\
      --emit_for_detector outputs/s5_rejected.jsonl
  python eval/normalize_responses.py --in_file outputs/s5_rejected.jsonl \\
      --out outputs/s5_rejected.norm.jsonl
  ( cd "$ICD" && python prosec_scripts/detect_all.py \\
      --fin "$REPO/outputs/s5_rejected.norm.jsonl" )
  python data/diff_overlap.py --train_file data/train_pref.jsonl \\
      --detected outputs/s5_rejected.norm.jsonl.detected.jsonl

detect_all.py 的 detection_results 內部長相沒有文件，本腳本用遞迴搜尋去找
行號欄位（line / line_number / start_line / start.line …），並在開頭印出
它實際認到的鍵，方便核對。認不到就會明講，不會默默回報 0。
"""
import argparse
import difflib
import json
import os
from collections import Counter


def pct(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    i = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
    return s[i]


def summarize(name, vals, fmt="{:.3f}"):
    if not vals:
        print(f"{name:<26} （無資料）")
        return
    m = sum(vals) / len(vals)
    print(f"{name:<26} mean {fmt.format(m):>9}   p25 {fmt.format(pct(vals,25)):>9}"
          f"   p50 {fmt.format(pct(vals,50)):>9}   p75 {fmt.format(pct(vals,75)):>9}"
          f"   max {fmt.format(max(vals)):>9}")


def line_hunks(rejected, chosen):
    """回傳 rejected 側被改動的行區間 [(start, end), ...]，0-based、end 不含。

    只取 rejected 側：analyzer 是在漏洞版上標行號的，要對照就得用同一套座標。
    純新增（chosen 多出幾行、rejected 沒有對應行）在 rejected 側是零寬區間，
    我們記成寬度 1 的區間，代表「插入點附近」。
    """
    a = rejected.split("\n")
    b = chosen.split("\n")
    ops = difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
    hunks, kinds = [], []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        hunks.append((i1, i2) if i2 > i1 else (i1, i1 + 1))
        kinds.append(tag)
    return a, hunks, kinds


def find_lines(obj, found_keys, depth=0):
    """在 detection_results 的任意巢狀結構裡遞迴找行號。

    不假設 schema：semgrep 用 start.line，weggli 和自製 regex 各有各的寫法。
    只收 1..100000 的整數，避免把 offset / column / cwe 編號誤當行號。
    """
    out = []
    if depth > 6:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = k.lower()
            if kl in ("line", "line_number", "lineno", "start_line", "linenumber"):
                if isinstance(v, int) and 1 <= v <= 100000:
                    out.append(v)
                    found_keys[k] += 1
                elif isinstance(v, str) and v.isdigit():
                    out.append(int(v))
                    found_keys[k] += 1
            elif kl in ("start", "end", "position", "location", "region"):
                if isinstance(v, int) and 1 <= v <= 100000:
                    out.append(v)
                    found_keys[k] += 1
                else:
                    out += find_lines(v, found_keys, depth + 1)
            else:
                out += find_lines(v, found_keys, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            out += find_lines(v, found_keys, depth + 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", default="data/train_pref.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = 全部")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--emit_for_detector", default=None,
                    help="把 rejected 寫成 detect_all.py 吃的格式（之後要先過 "
                         "eval/normalize_responses.py 補 code fence）")
    ap.add_argument("--detected", default=None,
                    help="detect_all.py 的輸出，開啟 B 段")
    ap.add_argument("--tolerance", type=int, default=1,
                    help="finding 行號與 hunk 的容差行數。analyzer 常標在呼叫行、"
                         "修補卻動在前一行（宣告或檢查），預設放寬 1 行")
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.train_file) if l.strip()]
    print(f"讀入 {len(rows)} 筆")
    if args.limit and args.limit < len(rows):
        import random
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.limit]
        print(f"隨機抽樣 {args.limit} 筆（seed {args.seed}）")

    # --- 產 analyzer 的輸入 ---------------------------------------------
    if args.emit_for_detector:
        os.makedirs(os.path.dirname(os.path.abspath(args.emit_for_detector)) or ".",
                    exist_ok=True)
        with open(args.emit_for_detector, "w") as f:
            for i, e in enumerate(rows):
                # 每筆一個 response，順序即 index，B 段靠這個對回來
                f.write(json.dumps({"idx": i, "prompt": e["prompt"],
                                    "responses": [e["rejected"]],
                                    "lang": e.get("lang", "?"),
                                    "cwe": e.get("cwe", "?")},
                                   ensure_ascii=False) + "\n")
        print(f"→ {args.emit_for_detector}（{len(rows)} 筆，順序即 idx）")
        print("  接著：normalize_responses.py 補 fence → detect_all.py → 用 --detected 跑回來")

    # --- A 段：行層級 diff ----------------------------------------------
    n_hunks, frac_changed, scatter, n_lines = [], [], [], []
    kind_ct = Counter()
    all_hunks = []
    for e in rows:
        lines, hunks, kinds = line_hunks(e["rejected"], e["chosen"])
        all_hunks.append((lines, hunks))
        kind_ct.update(kinds)
        n_lines.append(len(lines))
        n_hunks.append(len(hunks))
        changed = sum(h2 - h1 for h1, h2 in hunks)
        frac_changed.append(changed / max(1, len(lines)))
        if len(hunks) >= 2:
            scatter.append((hunks[-1][1] - hunks[0][0]) / max(1, len(lines)))

    print("\n" + "=" * 92)
    print("【A 段】行層級 diff（座標在 rejected 側，與 analyzer 行號同一套）")
    print("=" * 92)
    summarize("rejected 行數", n_lines, "{:.0f}")
    summarize("hunk 數（行層級）", n_hunks, "{:.1f}")
    summarize("被改動行佔比", frac_changed)
    summarize("hunk 散佈範圍佔比", scatter)
    tot = sum(kind_ct.values()) or 1
    print("\nhunk 種類：" + "  ".join(f"{k} {v:,} ({v/tot:.1%})"
                                     for k, v in kind_ct.most_common()))
    n1 = sum(1 for h in n_hunks if h == 1)
    print(f"單一連續 hunk：{n1:,} / {len(rows):,} ({n1/max(1,len(rows)):.1%})"
          "   ← 越高代表修補越集中，masking 的邊界越乾淨")

    # --- B 段：與 analyzer 行號對照 --------------------------------------
    result = {"n": len(rows), "hunks_mean": sum(n_hunks) / max(1, len(n_hunks)),
              "frac_changed_p50": pct(frac_changed, 50)}
    if args.detected:
        det = [json.loads(l) for l in open(args.detected) if l.strip()]
        print("\n" + "=" * 92)
        print(f"【B 段】與 analyzer 行號對照（容差 ±{args.tolerance} 行）")
        print("=" * 92)
        print(f"讀入 {len(det)} 筆 analyzer 輸出")

        found_keys = Counter()
        by_idx = {}
        for r in det:
            i = r.get("idx")
            if i is None:
                continue
            by_idx[i] = find_lines(r.get("detection_results", []), found_keys)

        if not found_keys:
            print("❌ 在 detection_results 裡找不到任何行號欄位。")
            print("   B 段無法進行——先看一筆的實際結構再決定怎麼取：")
            for r in det:
                if r.get("detection_results"):
                    print("   " + json.dumps(r["detection_results"][0],
                                             ensure_ascii=False)[:400])
                    break
            return
        print("認到的行號欄位：" +
              "  ".join(f"{k}×{v:,}" for k, v in found_keys.most_common()))

        n_pairs = n_find = n_cov = 0
        n_hunk_tot = n_hunk_hit = 0
        per_cwe = {}
        for i, (lines, hunks) in enumerate(all_hunks):
            fl = by_idx.get(i)
            if not fl:
                continue           # 這筆 analyzer 沒標到，無從對照
            n_pairs += 1
            tol = args.tolerance
            hit_h = set()
            cov = 0
            for ln in fl:
                z = ln - 1         # analyzer 1-based → 0-based
                ok = False
                for hi, (h1, h2) in enumerate(hunks):
                    if h1 - tol <= z < h2 + tol:
                        ok = True
                        hit_h.add(hi)
                if ok:
                    cov += 1
            n_find += len(fl)
            n_cov += cov
            n_hunk_tot += len(hunks)
            n_hunk_hit += len(hit_h)
            c = rows[i].get("cwe", "?")
            a_, b_, c_, d_ = per_cwe.get(c, (0, 0, 0, 0))
            per_cwe[c] = (a_ + len(fl), b_ + cov, c_ + len(hunks), d_ + len(hit_h))

        if not n_pairs:
            print("❌ 沒有任何一筆同時有 finding 與 diff，無法對照")
            return
        covered = n_cov / max(1, n_find)
        precision = n_hunk_hit / max(1, n_hunk_tot)
        print(f"\n可對照的配對         {n_pairs:,} / {len(rows):,}"
              f"（其餘 analyzer 沒在 rejected 上標到東西）")
        print(f"finding 總數         {n_find:,}")
        print(f"covered   落在 hunk 內的 finding    {n_cov:,}/{n_find:,} = {covered:.1%}"
              "   ← 低代表 masking 會漏掉修補")
        print(f"precision 內含 finding 的 hunk      {n_hunk_hit:,}/{n_hunk_tot:,} = {precision:.1%}"
              "   ← **決策變數**，低代表 masking 圈到的是無關重寫")

        print("\n依 CWE（finding 數前 12 名）：")
        print(f"{'cwe':<14}{'findings':>10}{'covered':>10}{'hunks':>9}{'precision':>11}")
        for c, (f_, cv, h_, hh) in sorted(per_cwe.items(),
                                          key=lambda kv: -kv[1][0])[:12]:
            print(f"{c:<14}{f_:>10,}{cv/max(1,f_):>9.1%}{h_:>9,}{hh/max(1,h_):>10.1%}")

        print("\n" + "-" * 92)
        if precision >= 0.5:
            print(f"✅ precision {precision:.1%} —— 過半 hunk 壓在漏洞行上，"
                  "masking 抓到的是訊號，S5 值得實作")
        elif precision >= 0.3:
            print(f"⚠️  precision {precision:.1%} —— 一半以上的 hunk 與 analyzer "
                  "標記無關。可先做「只保留含 finding 的 hunk」的嚴格版 mask，"
                  "但要預期訊號被稀釋")
        else:
            print(f"❌ precision {precision:.1%} —— 絕大多數 hunk 是無關重寫，"
                  "masking 圈到的主要是雜訊。依 EXPERIMENTS.md 的停損點，S5 應放棄，"
                  "方法創新改由 §10.1（CWE-specific 多 Prefix）承擔")
        result.update({"covered": covered, "precision": precision,
                       "n_findings": n_find, "n_pairs": n_pairs})

    if args.json_out:
        json.dump(result, open(args.json_out, "w"), indent=2, ensure_ascii=False)
        print(f"\n→ {args.json_out}")


if __name__ == "__main__":
    main()
