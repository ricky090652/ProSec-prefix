"""載入 adapter，支援 prefix_then_lora 的套疊結構。

存檔結構（train_prefix.py --peft_method prefix_then_lora）：
    <dir>/                 外層 prefix（HF 原生格式）
    <dir>/lora/            內層 LoRA

兩層都要載，而且 OFF 對照要同時關掉兩層——只關外層的話留下的是 base + LoRA，
不是 base。單層 adapter 走原本的路徑，行為不變。
"""
import os
from contextlib import contextmanager


def load_adapter(base, path):
    """回傳 (model, off)。off() 是把所有 adapter 關掉的 context manager。"""
    from peft import PeftModel
    lora_dir = os.path.join(path, "lora")
    if not os.path.isdir(lora_dir):
        m = PeftModel.from_pretrained(base, path)
        return m, m.disable_adapter
    inner = PeftModel.from_pretrained(base, lora_dir)      # 內層 LoRA
    outer = PeftModel.from_pretrained(inner, path)         # 外層 prefix

    @contextmanager
    def off():
        with outer.disable_adapter(), inner.disable_adapter():
            yield

    return outer, off
