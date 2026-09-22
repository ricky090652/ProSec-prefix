"""載入 adapter，支援 prefix_then_lora 的套疊結構。

存檔結構（train_prefix.py --peft_method prefix_then_lora）：
    <dir>/                 外層 prefix（HF 原生格式）
    <dir>/lora/            內層 LoRA

⚠️ 不能直接用套疊的 PeftModel 來生成。PEFT 的 prefix 在 generate() 時是把
prepare_inputs_for_generation 的 hook 掛到它包住的那一層；套疊時那一層是 LoRA 的
PeftModel，而真正執行生成的是更底層的 HF 模型，hook 接不上，**prefix 會被默默漏掉**
——forward 有用到、generate 沒有。訓練只走 forward 所以不受影響，評測會錯。

所以 ON 改成：LoRA 用 merge_and_unload() 摺進權重 → 變回普通 HF 模型 → 再接單層
prefix。單層 prefix 的 generate 是正常的。OFF 另外載一份純 base。兩份模型約多佔
一份權重的 VRAM（Phi-3-mini bf16 約 7.6GB）。

單層 adapter 走原本的路徑，行為不變。
"""
import os
from contextlib import contextmanager


class _OnOff:
    """把 ON / OFF 兩個模型包成一個物件，讓評測腳本不必改動呼叫方式。"""

    def __init__(self, on, off):
        object.__setattr__(self, "_on", on)
        object.__setattr__(self, "_off", off)
        object.__setattr__(self, "_use_off", False)

    def _m(self):
        return self._off if self._use_off else self._on

    def generate(self, *a, **k):
        return self._m().generate(*a, **k)

    def __call__(self, *a, **k):
        return self._m()(*a, **k)

    def __getattr__(self, name):
        return getattr(self._m(), name)


def load_adapter(base, path):
    """回傳 (model, off)。off() 是讓 model 表現成純 base 的 context manager。"""
    from peft import PeftModel
    lora_dir = os.path.join(path, "lora")
    if not os.path.isdir(lora_dir):
        m = PeftModel.from_pretrained(base, path)
        return m, m.disable_adapter

    from transformers import AutoModelForCausalLM
    # ON：LoRA 摺進 base 的權重，再接單層 prefix
    merged = PeftModel.from_pretrained(base, lora_dir).merge_and_unload()
    on = PeftModel.from_pretrained(merged, path).eval()
    # OFF：另一份純 base，設定與傳進來的 base 相同
    dev = next(on.parameters()).device
    off_m = AutoModelForCausalLM.from_pretrained(
        base.config._name_or_path, torch_dtype=next(on.parameters()).dtype,
        device_map={"": dev} if dev.type == "cuda" else None).eval()
    w = _OnOff(on, off_m)

    @contextmanager
    def off():
        object.__setattr__(w, "_use_off", True)
        try:
            yield
        finally:
            object.__setattr__(w, "_use_off", False)

    return w, off
