"""把 ProSec 最終混合資料集轉成 (prompt, chosen, rejected) 偏好格式。

ProSec 的最終資料（src/mix_and_upload_original_w_fixed_batch.py 產出 / 上傳到 HF）
每筆包含以下欄位（已確認自 ProSec repo）：
  - original_instruction : 觸發 CWE 的指令（或正常指令）
  - fixed_code           : 修復後 / 正常的程式碼  → 對應 chosen (win)
  - original_code        : 脆弱 / 較差的程式碼     → 對應 rejected (lose)
  - lang, cwe, benign

注意：ProSec 對 benign(Dnorm) 項已預先把欄位對調好，所以
      chosen=fixed_code, rejected=original_code 對 Dsec 與 Dnorm 都成立。
      展開來看：
        Dsec  (benign=False)：chosen = y_f（修好的碼）   rejected = y_v（漏洞碼）
        Dnorm (benign=True) ：chosen = y_n（正常碼）     rejected = y_f（修好的碼）
      → **Dnorm.rejected 與它對應的 Dsec.chosen 是同一份 y_f 字串**，
        §3.3 influence selection 就是靠這個字串把兩邊配對起來
        （見 ProSec influence_score/sample_refactored.py 的 res2ifv_idx）。

用法：
  # 從 HuggingFace dataset
  python convert_prosec_to_pref.py --hf_dataset <user>/<prosec-dataset> --out data/train_pref.jsonl
  # 或從本地 jsonl
  python convert_prosec_to_pref.py --in_jsonl raw_prosec.jsonl --out data/train_pref.jsonl
"""
import argparse
import json


def iter_entries(args):
    if args.hf_dataset:
        import datasets
        ds = datasets.load_dataset(args.hf_dataset, split=args.split)
        for e in ds:
            yield e
    else:
        with open(args.in_jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf_dataset", type=str, help="HuggingFace dataset 名稱")
    src.add_argument("--in_jsonl", type=str, help="本地 ProSec jsonl")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--out", type=str, required=True)
    # 欄位名稱可覆寫，以防 ProSec 版本差異
    ap.add_argument("--prompt_field", default="original_instruction")
    ap.add_argument("--chosen_field", default="fixed_code")
    ap.add_argument("--rejected_field", default="original_code")
    args = ap.parse_args()

    n_in, n_out = 0, 0
    with open(args.out, "w") as fout:
        for e in iter_entries(args):
            n_in += 1
            prompt = e.get(args.prompt_field)
            chosen = e.get(args.chosen_field)
            rejected = e.get(args.rejected_field)
            if not (prompt and chosen and rejected):
                continue
            if chosen.strip() == rejected.strip():
                continue  # 無偏好訊號，跳過
            rec = {
                "id": n_out,                  # 穩定 id，influence selection 用來回指
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "lang": e.get("lang", ""),
                "cwe": e.get("cwe", ""),
                # benign=False → Dsec；True → Dnorm。§3.3 的篩選與 --subset 都需要它
                "benign": bool(e.get("benign", False)),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"讀入 {n_in} 筆，輸出 {n_out} 筆偏好對 → {args.out}")


if __name__ == "__main__":
    main()
