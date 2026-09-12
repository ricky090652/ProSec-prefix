"""零初始化的 prefix，nvt 個位置是不是永遠保持相同？（純 CPU，幾秒）

為什麼要查：零初始化把 K 和 V 全設 0，於是 **nvt 個 prefix 位置在 step 0 完全相同**。
由對稱性它們會收到完全相同的梯度；Adam 的 m、v 初始也都是 0，所以更新量也相同。
在沒有任何隨機來源的情況下，它們會**永遠保持相同**。

若成立，意涵是：nvt=16 在「學到的內容」上等於 nvt=1，卻在「吸走 attention mass」
上付 16 個位置的代價（見 eval/measure_prefix_init.py 量的 Z/(Z+nvt)）。那就一次
解釋了「加長 prefix 只增加傷害、不增加能力」這個我們量到的單調趨勢。

我們的兩臂剛好是一組對照：
  prefix nvt=16  無 dropout       → 預測 16 列近乎完全相同
  prefix nvt=8   prefix_dropout=0.1 → dropout 打破對稱 → 預測各列不同
SVEN 預設帶 dropout（sven/model.py:20），所以它沒有這個問題。

判讀 max_pairwise_rel（列與列之間最大的相對差異）：
  < 1e-3   對稱性沒被打破 → 有效長度 ≈ 1，該臂的 nvt 是假的
  > 1e-1   各列已充分分化 → 有效長度接近 nvt
  中間      部分分化，看 effective_n 的估計

用法：
  python eval/check_prefix_symmetry.py outputs/dpo-arms/prefix outputs/dpo-arms/prefix_nvt8
"""
import argparse
import json
import os


def load_prefix(path):
    """從 adapter 目錄取出 prefix 張量，回傳 [nvt, D]。"""
    import torch
    from safetensors.torch import load_file
    cfg = json.load(open(os.path.join(path, "adapter_config.json")))
    nvt = cfg.get("num_virtual_tokens")
    if cfg.get("peft_type") != "PREFIX_TUNING":
        raise SystemExit(f"{path} 不是 PREFIX_TUNING（是 {cfg.get('peft_type')}）")
    f = os.path.join(path, "adapter_model.safetensors")
    sd = load_file(f) if os.path.exists(f) else torch.load(
        os.path.join(path, "adapter_model.bin"), map_location="cpu")
    # 取第一個維度等於 nvt 的 2-D 張量；PEFT 存的鍵名是 prompt_embeddings
    for k, v in sd.items():
        if v.ndim == 2 and v.shape[0] == nvt:
            return k, v.float(), nvt, cfg
    raise SystemExit(f"{path} 找不到第一維 = {nvt} 的 2-D 張量（鍵：{list(sd)}）")


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("adapters", nargs="+")
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    rows = []
    for path in args.adapters:
        key, W, nvt, cfg = load_prefix(path)
        norms = W.norm(dim=1)
        # 列與列之間的相對差異：||w_i − w_j|| / 平均範數
        d = torch.cdist(W, W) / norms.mean().clamp_min(1e-12)
        iu = torch.triu_indices(nvt, nvt, offset=1)
        pair = d[iu[0], iu[1]]
        # 有效列數：奇異值的參與熵，exp(H)。全相同 → 1；完全獨立 → nvt
        s = torch.linalg.svdvals(W)
        p = (s / s.sum().clamp_min(1e-12)).clamp_min(1e-12)
        eff = float(torch.exp(-(p * p.log()).sum()))
        r = {"arm": os.path.basename(path), "nvt": nvt,
             "max_pairwise_rel": float(pair.max()) if pair.numel() else 0.0,
             "mean_pairwise_rel": float(pair.mean()) if pair.numel() else 0.0,
             "effective_n": eff, "norm_mean": float(norms.mean())}
        rows.append(r)
        print(f"{r['arm']:<16} nvt={nvt:<3} 張量={key} {tuple(W.shape)}")

    print("\n" + "=" * 78)
    print(f"{'臂':<16}{'nvt':>5}{'列間最大相對差':>16}{'列間平均':>11}{'有效列數':>11}")
    print("=" * 78)
    for r in rows:
        print(f"{r['arm']:<16}{r['nvt']:>5}{r['max_pairwise_rel']:>16.3e}"
              f"{r['mean_pairwise_rel']:>11.3e}{r['effective_n']:>11.2f}")

    print("\n判讀：")
    for r in rows:
        m, n, e = r["max_pairwise_rel"], r["nvt"], r["effective_n"]
        if m < 1e-3:
            v = (f"❌ 對稱性未打破——{n} 個位置實質上是同一個，"
                 f"有效長度 ≈ 1 卻付 {n} 倍的 attention 吸收代價")
        elif m < 1e-1:
            v = f"⚠️  只有部分分化（有效列數 {e:.2f} / {n}）"
        else:
            v = f"✅ 各列已分化（有效列數 {e:.2f} / {n}）"
        print(f"  {r['arm']:<16}{v}")
    print("\n若無 dropout 的那一臂落在 ❌，就找到「加長 prefix 只增傷害不增能力」的")
    print("直接原因：零初始化 + 無隨機來源 → nvt 個位置永遠相同。修法是加 dropout")
    print("（SVEN 的預設）或改用隨機初始化 / MLP 重參數化（PT-PEFT 的做法）。")

    if args.json_out:
        json.dump(rows, open(args.json_out, "w"), indent=2, ensure_ascii=False)
        print(f"\n→ {args.json_out}")


if __name__ == "__main__":
    main()
