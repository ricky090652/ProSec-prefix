"""依 ProSec進度報告.pdf 的版面重建投影片，並補上 DPO full train 結果與 future work。

用法：.venv/bin/python docs/build_deck.py
輸出：docs/ProSec進度報告_v3.pptx
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.oxml.ns import qn
from lxml import etree

BLACK = RGBColor(0, 0, 0)
RED = RGBColor(0xC0, 0x00, 0x00)
BLUE = RGBColor(0x44, 0x72, 0xC4)
GREY = RGBColor(0x59, 0x59, 0x59)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREEN = RGBColor(0x1F, 0x7A, 0x3D)

LATIN = "Calibri"
EA = "Microsoft JhengHei"
MONO = "Consolas"


def style(run, size=14, bold=False, color=BLACK, latin=LATIN, ea=EA, italic=False):
    f = run.font
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color
    f.name = latin
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = etree.SubElement(rPr, qn(tag))
        el.set("typeface", ea if tag == "a:ea" else latin)
    return run


def textbox(slide, x, y, w, h, wrap=True):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    return tf


def title(slide, text, size=36):
    tf = textbox(slide, 0.55, 0.28, 12.2, 0.95)
    style(tf.paragraphs[0].add_run(), size)
    tf.paragraphs[0].runs[0].text = text
    style(tf.paragraphs[0].runs[0], size, latin="Calibri Light")
    return tf


def bullets(slide, x, y, w, h, items, size=15, gap=4):
    """items: [(text, level, color, bold), ...] or plain strings"""
    tf = textbox(slide, x, y, w, h)
    first = True
    for it in items:
        if isinstance(it, str):
            it = (it, 0, BLACK, False)
        text, lvl, color, bold = (list(it) + [0, BLACK, False])[:4]
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = lvl
        p.space_after = Pt(gap)
        r = p.add_run()
        r.text = ("• " if lvl == 0 else "– ") + text if text else ""
        style(r, size - lvl, bold=bold, color=color)
    return tf


def plain(slide, x, y, w, h, lines, size=14, mono=False, color=BLACK, gap=2):
    tf = textbox(slide, x, y, w, h)
    first = True
    for ln in lines:
        c = color
        b = False
        if isinstance(ln, tuple):
            ln, c = ln[0], ln[1]
            if len(ln) and isinstance(c, bool):
                ln, b, c = ln, c, color
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(gap)
        r = p.add_run()
        r.text = ln
        style(r, size, bold=b, color=c, latin=MONO if mono else LATIN,
              ea=MONO if mono else EA)
    return tf


def table(slide, x, y, w, rows, col_w=None, size=12, head=True, row_h=0.32):
    nr, nc = len(rows), len(rows[0])
    shp = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w),
                                 Inches(row_h * nr))
    tbl = shp.table
    tbl.first_row = False
    tbl.horz_banding = False
    if col_w:
        total = sum(col_w)
        for i, cw in enumerate(col_w):
            tbl.columns[i].width = Emu(int(Inches(w) * cw / total))
    for ri, row in enumerate(rows):
        tbl.rows[ri].height = Inches(row_h)
        for ci, cell in enumerate(row):
            txt, color, bold = cell if isinstance(cell, tuple) else (cell, BLACK, False)
            c = tbl.cell(ri, ci)
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            c.fill.fore_color.rgb = WHITE
            tf = c.text_frame
            tf.word_wrap = True
            first = True
            for sub in str(txt).split("\n"):
                p = tf.paragraphs[0] if first else tf.add_paragraph()
                first = False
                p.space_after = Pt(0)
                r = p.add_run()
                r.text = sub
                style(r, size, bold=bold or (head and ri == 0), color=color)
    return tbl


def box(slide, x, y, w, h, lines, outline=BLUE, size=13, rounded=True, fill=WHITE):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = outline
    shape.line.width = Pt(1.25)
    shape.shadow.inherit = False
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.08)
    tf.margin_top = tf.margin_bottom = Inches(0.04)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    first = True
    for ln in lines:
        txt, color, bold = (ln if isinstance(ln, tuple) else (ln, BLACK, False))
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = PP_ALIGN.CENTER
        p.space_after = Pt(1)
        r = p.add_run()
        r.text = txt
        style(r, size, bold=bold, color=color)
    return shape


def arrow(slide, x1, y1, x2, y2, color=BLUE):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                                   Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(1.5)
    ln = c.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    return c


def label(slide, x, y, w, text, size=11, color=BLACK, align=PP_ALIGN.CENTER,
          bold=False):
    tf = textbox(slide, x, y, w, 0.3)
    first = True
    for sub in str(text).split("\n"):
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = align
        p.space_after = Pt(0)
        r = p.add_run()
        r.text = sub
        style(r, size, bold=bold, color=color)
    return tf


def code(slide, x, y, w, h, lines, size=11.5):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                                   Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(0xF2, 0xF2, 0xF2)
    shape.line.color.rgb = RGBColor(0xBF, 0xBF, 0xBF)
    shape.line.width = Pt(0.75)
    shape.shadow.inherit = False
    tf = shape.text_frame
    tf.word_wrap = False
    tf.margin_left = Inches(0.1)
    tf.margin_top = Inches(0.06)
    tf.vertical_anchor = MSO_ANCHOR.TOP
    first = True
    for ln in lines:
        txt, color = ln if isinstance(ln, tuple) else (ln, BLACK)
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(0)
        r = p.add_run()
        r.text = txt
        style(r, size, color=color, latin=MONO, ea=MONO)
    return shape


def placeholder(slide, x, y, w, h, text):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                                   Inches(w), Inches(h))
    shape.fill.background()
    shape.line.color.rgb = RGBColor(0xA6, 0xA6, 0xA6)
    shape.line.width = Pt(1)
    shape.line.dash_style = 4  # dashed
    shape.shadow.inherit = False
    tf = shape.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    style(p.add_run(), 13, color=GREY)
    p.runs[0].text = text
    return shape


prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
S = lambda: prs.slides.add_slide(BLANK)

# ───────────────────── 1 · Title ─────────────────────
s = S()
tf = textbox(s, 0, 2.7, 13.333, 1.2)
tf.paragraphs[0].alignment = PP_ALIGN.CENTER
style(tf.paragraphs[0].add_run(), 54, latin="Calibri Light")
tf.paragraphs[0].runs[0].text = "進度報告"
tf2 = textbox(s, 0, 4.0, 13.333, 0.5)
tf2.paragraphs[0].alignment = PP_ALIGN.CENTER
style(tf2.paragraphs[0].add_run(), 20)
tf2.paragraphs[0].runs[0].text = "2026/09/07"

# ───────────────────── 2 · Topic ─────────────────────
s = S()
title(s, "Topic")
box(s, 0.6, 1.35, 5.6, 1.85, [
    ("SVEN (2023)", BLACK, True),
    ("• Prefix tuning for secure / insecure code generation", RED, False),
    ("• Freezes the base model", RED, False),
    ("• CodeGen (2022), torch 1.13 → not reusable", BLACK, False),
    ("• Real-world GitHub commits → small and noisy", BLACK, False),
], size=13)
box(s, 7.1, 1.35, 5.6, 1.85, [
    ("ProSec (2025)", BLACK, True),
    ("• Proactively synthesized vulnerability data", RED, False),
    ("• Preference optimization for fine-tuning", RED, False),
    ("• Adapts with LoRA", BLACK, False),
], size=13)
arrow(s, 3.4, 3.25, 5.4, 4.15)
arrow(s, 9.9, 3.25, 7.9, 4.15)
label(s, 2.1, 3.45, 2.2, "Prefix tuning", 12)
label(s, 9.9, 3.35, 3.2, "Data construct pipeline\npreference optimization", 12)
sh = box(s, 2.9, 4.2, 7.5, 0.75, [("ProSec's data and pipeline with ", BLACK, True)], size=15)
p = sh.text_frame.paragraphs[0]
style(p.add_run(), 15, bold=True, color=RED).text = "prefix (DPO) instead of LoRA"
label(s, 2.9, 5.15, 7.5, "核心主張是相對的：相同條件下 prefix 能不能追上 LoRA", 14, color=RED)

# ───────────── 3 · ProSec data pipeline ─────────────
s = S()
title(s, "Data syn & sel pipeline of ProSec")
placeholder(s, 0.6, 1.3, 7.4, 4.4, "貼上原本的 ProSec Figure 2")
bullets(s, 8.3, 1.35, 4.5, 3.0, [
    ("Step 1  Synthesize vulnerability-inducing instructions", 0, BLACK, True),
    ("50,000 instructions, clustered", 1, GREY, False),
    ("Step 2  Construct candidate preference data", 0, BLACK, True),
    ("Code LLM writes normal / fixed / vulnerable code", 1, GREY, False),
    ("Step 3  Select high-quality samples", 0, BLACK, True),
    ("Training-dynamics sampler + heuristic filter", 1, GREY, False),
], size=14)
box(s, 8.3, 4.3, 4.5, 1.4, [
    ("我們拿到的 45,785 筆已經是 Step 3 的輸出", RED, True),
    ("指紋：列嚴格分區 · y_f 已去重", BLACK, False),
    ("· D_norm 候選數卡在 2（top_n=2）", BLACK, False),
    ("→ 直接訓練這份才是忠實的復現", BLACK, False),
], size=12, outline=RED)

# ───────────── 4 · LoRA & Prefix ─────────────
s = S()
title(s, "LoRA & Prefix")
label(s, 0.6, 1.2, 5.9, "LoRA — 在 transformer 內部層注入可訓練的 low-rank 矩陣", 14, align=PP_ALIGN.LEFT)
label(s, 7.0, 1.2, 5.9, "Prefix — 在每層的 key/value 序列前插入虛擬 token", 14, align=PP_ALIGN.LEFT)
placeholder(s, 0.6, 1.6, 5.9, 1.5, "貼上 LoRA 結構圖")
placeholder(s, 7.0, 1.6, 5.9, 1.5, "貼上 Prefix / KV 序列圖")
box(s, 0.6, 3.25, 5.9, 1.9, [
    ("Pros", BLACK, True),
    ("• 效能更接近全參數微調", BLACK, False),
    ("Cons", BLACK, True),
    ("• 改變預訓練權重，切換需 merge / unmerge", BLACK, False),
    ("• 複雜任務調高 rank → 參數量與記憶體增加", BLACK, False),
], size=13)
box(s, 7.0, 3.25, 5.9, 1.9, [
    ("Pros", BLACK, True),
    ("• 參數量少，完整保留預訓練權重", BLACK, False),
    ("• 切換 ≈ 換一組 KV；同 batch 可用不同 prefix", RED, False),
    ("Cons", BLACK, True),
    ("• 梯度只能經 softmax 回傳 → 需要高 10 倍的 lr", BLACK, False),
], size=13)
label(s, 0.6, 5.3, 12.3, "prefix 的參數化方式忠於 SVEN：直接參數化、無 MLP、零初始化，並補上 SVEN 的 prefix dropout",
      14, color=RED, align=PP_ALIGN.LEFT)
table(s, 0.6, 5.7, 12.3, [
    ["", "可訓練參數", "adapter (bf16)", "實測最佳 lr", "來源"],
    ["LoRA r=8, all-linear", "12,582,912", "≈ 25.2 MB", "5e-6", "= 論文 Table 6"],
    ["Prefix nvt=8 + dropout 0.1", ("1,572,864", RED, True), ("≈ 3.1 MB", RED, True), ("5e-5", RED, True),
     ("= SVEN 對 3.8B 的建議尺度", RED, True)],
], col_w=[3, 2, 2, 1.6, 2.6], size=12)

# ───────────── 5 · Different from paper ─────────────
s = S()
title(s, "Different from paper and follows")
table(s, 0.6, 1.3, 12.2, [
    ["", "ProSec paper", "This work", "Reason"],
    ["PEFT", "LoRA r=8, α=16", ("Prefix nvt = 8 / 16", RED, True), "Pluggable, flexible"],
    ["Objective", "SimPO (β=1.5, γ=0.5)", ("DPO (β=0.05)", RED, True),
     "SimPO 在全長訓練下兩臂都崩；論文 Appendix D.3 自己驗證過 DPO"],
    ["Training length", "1500 steps @ batch 64", "800 steps @ batch 64", "≈ 1.1 epoch over 45,785"],
    ["LoRA target modules", ("未指定", RED, True), "all-linear / qkv_proj 各跑一組",
     ("論文沒寫，這正是兩軸無法同時復現的原因", RED, True)],
    ["Evaluation",
     "PurpleLlama ∩ SafeCoder\n38 pairs / 694 cases（未釋出）\nHumanEval-Multi",
     ("Rebuilt 36 pairs / 693 cases\nHumanEval + MultiPL-E\nDegeneration test", RED, True),
     "子集必須自行重建\nThe security rate can be faked"],
], col_w=[2.2, 3, 3, 4], size=12, row_h=0.55)

# ───────────── 6 · ProSec dataset ─────────────
s = S()
title(s, "ProSec dataset")
box(s, 0.5, 3.0, 2.9, 1.2, [
    ("Haiku-vul-inducing-", BLACK, True), ("instructions-clustered", BLACK, True),
    ("(50,000 instructions)", BLACK, False)], size=13)
box(s, 4.1, 1.75, 3.4, 1.15, [
    ("Phi-3-mini generates", BLACK, True), ("Vulnerable + fixed code", BLUE, False),
    ("(validated by detector)", BLACK, False)], size=12.5)
box(s, 4.1, 4.35, 3.4, 1.15, [
    ("Phi-3-mini generates", BLACK, True), ("Normal code + benign code", GREEN, False)], size=12.5)
box(s, 8.1, 1.75, 3.2, 1.15, [("D_sec, 27,400 pairs", BLUE, True), ("fixed ≻ vulnerable", BLACK, False)], size=13)
box(s, 8.1, 4.35, 3.2, 1.15, [("D_norm, 18,385 pairs", GREEN, True), ("normal ≻ over-secure", BLACK, False)], size=13)
box(s, 11.7, 3.0, 1.5, 1.2, [("45,785", BLACK, True), ("pairs", BLACK, False)], size=14)
arrow(s, 3.4, 3.3, 4.05, 2.4); arrow(s, 3.4, 3.9, 4.05, 4.8)
arrow(s, 7.55, 2.32, 8.05, 2.32); arrow(s, 7.55, 4.92, 8.05, 4.92)
arrow(s, 11.35, 2.6, 11.9, 3.3); arrow(s, 11.35, 4.6, 11.9, 3.9)
label(s, 0.5, 5.9, 12.3, "每列都是完整的偏好對（指令 + 兩種程式碼）。D_norm 的欄位由 ProSec 上游預先對調，所以 fixed_code → chosen 對兩者都成立。",
      13, align=PP_ALIGN.LEFT)

# ───────────── 7 · Whole pipeline ─────────────
s = S()
title(s, "Whole pipeline")
label(s, 0.35, 1.35, 1.4, "training", 13, align=PP_ALIGN.LEFT)
box(s, 0.55, 1.75, 2.5, 0.95, [("ProSec dataset", BLACK, True), ("(45,785 pairs)", BLACK, False)])
box(s, 3.45, 1.75, 2.6, 0.95, [("(prompt, chosen,", BLACK, False), ("rejected) jsonl", BLACK, False)])
box(s, 6.45, 1.75, 3.0, 0.95, [("Prefix / LoRA training", BLACK, True), ("(freeze phi-3-mini)", BLACK, False)],
    fill=RGBColor(0xE2, 0xEF, 0xDA))
box(s, 9.85, 1.75, 3.0, 0.95, [("Adapter", BLACK, True), ("prefix ≈ 3.1 MB (nvt=8)", RED, False)],
    fill=RGBColor(0xDE, 0xEA, 0xF6))
arrow(s, 3.1, 2.22, 3.4, 2.22); arrow(s, 6.1, 2.22, 6.4, 2.22); arrow(s, 9.5, 2.22, 9.8, 2.22)
arrow(s, 11.35, 2.75, 11.35, 3.3, GREY); arrow(s, 11.35, 3.3, 1.8, 3.3, GREY); arrow(s, 1.8, 3.3, 1.8, 3.85, GREY)
label(s, 0.35, 3.5, 1.6, "evaluation", 13, align=PP_ALIGN.LEFT)
box(s, 0.55, 3.85, 2.35, 0.95, [("Generate response", BLACK, True), ("693 × 10 × ON/OFF", BLACK, False)])
box(s, 3.15, 3.85, 2.35, 0.95, [("Drop truncated", RED, True), ("(pairwise)", BLACK, False)], outline=RED)
box(s, 5.75, 3.85, 2.35, 0.95, [("Normalize response", RED, True), ("(wrap fence-less)", BLACK, False)], outline=RED)
box(s, 8.35, 3.85, 2.2, 0.95, [("Static analyzer", BLACK, True), ("semgrep / weggli", BLACK, False)])
box(s, 10.8, 3.85, 2.05, 0.95, [("Score", BLACK, True), ("vulnerable ratio", BLACK, False)])
for x in (2.95, 5.55, 8.15, 10.6):
    arrow(s, x, 4.32, x + 0.18, 4.32)
box(s, 1.9, 5.15, 4.2, 0.9, [("Utility test", BLACK, True), ("HumanEval + MultiPL-E", BLACK, False)])
box(s, 7.0, 5.15, 4.2, 0.9, [("Check degeneration", RED, True), ("length / parse rate / stub rate", BLACK, False)],
    outline=RED)
label(s, 0.55, 6.25, 12.3, "紅框 = 論文沒有、我們自己加上的步驟。粒度在靜態分析改變：693 題進去，6,930 個 code block 出來。",
      13, color=RED, align=PP_ALIGN.LEFT)

# ───────────── 8 · Normalize response ─────────────
s = S()
title(s, "Normalize response")
bullets(s, 0.6, 1.1, 12.2, 1.0, [
    "SAST only detects code inside a fence（```）",
    ("Fence 是 Markdown 語法，沒有任何地方能設定——模型加不加是 instruction tuning 學來的習慣", 1, BLACK, False),
    ("而 benchmark 的 prompt 結尾寫著 \"Only return the code, don't include any other information\"", 1, RED, False),
], size=14)
label(s, 0.6, 2.25, 6.0, "ProSec  detect_all.py", 12.5, align=PP_ALIGN.LEFT)
code(s, 0.6, 2.55, 6.0, 1.85, [
    "def parse_code_blocks(text):",
    "    code_blocks = []",
    "    in_code_block = False",
    "    for line in text.split('\\n'):",
    "        if '```' in line:",
    "            in_code_block = not in_code_block",
    "        elif in_code_block:",
    "            code_blocks.append(line)",
    ("    return '\\n'.join(code_blocks)   # 沒 fence → \"\"", RED),
], size=11)
label(s, 7.0, 2.25, 6.0, "eval/normalize_responses.py", 12.5, align=PP_ALIGN.LEFT)
code(s, 7.0, 2.55, 6.0, 1.05, [
    "if has_code_block(resp):",
    "    new.append(resp)                # 已有 fence，不動",
    "else:",
    ("    new.append(f\"```\\n{resp}\\n```\")   # 補上 fence", RED),
], size=11)
table(s, 7.0, 3.8, 6.0, [
    ["模型輸出", "抽取結果", "判定"],
    ["有 fence", "完整程式碼", "不安全 ✔"],
    [("沒有 fence", RED, True), ("\"\" 空字串", RED, True), ("「安全」← 錯", RED, True)],
    ["補 fence 後", "完整程式碼", "不安全 ✔"],
], col_w=[2, 2, 2], size=12)
box(s, 0.6, 4.65, 6.0, 1.35, [
    ("84% 的 code block 是空的", RED, True),
    ("漏洞率被稀釋約 6 倍，而整條鏈沒有任何一步報錯", BLACK, False),
    ("修完後 base 才與論文對齊：JS 52.24 vs 52.24", BLACK, False),
], size=13, outline=RED)

# ───────────── 9 · Degeneration & checks ─────────────
s = S()
title(s, "Degeneration & Additional check")
bullets(s, 0.6, 1.1, 12.2, 0.7, [
    "SAST can only report vulnerabilities in the code it can read",
    ("Anything that reduces readable code lowers the vulnerability rate", 1, BLACK, False),
], size=14)
table(s, 0.6, 1.85, 12.2, [
    ["Check", "Prevents", "Threshold / 判準"],
    ["Degeneration gate\n• code length\n• AST parse / bracket\n• stub / TODO / {}",
     "模型靠少寫或寫壞程式碼來降低漏洞率",
     "長度保留 ≥ 90%\n語法完整率 Δ ≥ −3 pt\nstub 率 Δ ≤ +2 pt"],
    ["Truncation flag",
     "被切斷的程式碼比對不到規則 → 算成「安全」；語法完整率同時被低估",
     "語法只計未截斷者；生成上限統一 2048"],
    ["Pairwise drop",
     "只濾單邊會讓比較基礎歪掉（ON 截斷 14.4% vs OFF 1.2%）",
     "任一邊截斷就兩邊一起丟"],
    ["Suppression ratio",
     "把「兩側一起壓低」誤判成模型學得更好",
     "Δrejected ÷ Δchosen；≈ 1 失敗、> 2 健康"],
], col_w=[2.6, 5.0, 4.6], size=11.5, row_h=0.6)
box(s, 0.6, 5.5, 12.2, 1.15, [
    ("兩次「檢查也會失靈」的教訓", RED, True),
    ("① 守門三條件擋的是「寫太少 / 寫壞 / 寫 stub」，擋不住「寫太多」——LoRA all-linear 通過守門，但程式碼長度是 base 的 226.4%", BLACK, False),
    ("② 訓練期代理指標不能跨方法比：prefix nvt=16 保留率 94.6% 卻掉 −10.98，LoRA all-linear 88.8% 只掉 −4.27", BLACK, False),
], size=12, outline=RED)

# ───────────── 10 · DPO lr sweep ─────────────
s = S()
title(s, "DPO lr sweep", 32)
label(s, 0.6, 1.0, 12.2, "β=0.05 · batch 64 · 16,000 samples = 250 steps · seed 42 · 兩臂同一張網格", 13,
      align=PP_ALIGN.LEFT)
label(s, 0.6, 1.35, 12.2, "選擇規則（兩臂共用）：壓rej/壓chosen 比值 ≥ 3 且梯度未爆的最高 lr", 13.5,
      color=RED, align=PP_ALIGN.LEFT)
label(s, 0.6, 1.8, 4.0, "LoRA r=8 all-linear", 14, align=PP_ALIGN.LEFT)
table(s, 0.6, 2.1, 5.9, [
    ["lr", "壓rej/壓chosen", "保留/token", "acc", "grad 峰值"],
    [("5e-6 ✔", RED, True), ("4.63", RED, True), ("99.3%", RED, True), "0.724", "58"],
    ["2e-5", "2.36", "86.0%", "0.975", "106"],
    ["5e-5", "1.94", "47.0%", "0.989", "366"],
    ["1e-4", "1.88", "20.0%", "0.995", "569"],
    ["2e-4", "2.37", "21.2%", "0.997", "608"],
], col_w=[1, 1.8, 1.5, 1, 1.2], size=12, row_h=0.33)
label(s, 0.6, 4.3, 5.9, "下一級 2e-5 比值掉到 2.36 → 排除。5e-6 正好是論文 Table 6 的值。", 12,
      align=PP_ALIGN.LEFT)
label(s, 7.0, 1.8, 4.0, "Prefix nvt=16", 14, align=PP_ALIGN.LEFT)
table(s, 7.0, 2.1, 5.9, [
    ["lr", "壓rej/壓chosen", "保留/token", "acc", "grad 峰值"],
    ["5e-6", "∞", "96.8%", ("0.554", RED, False), "45"],
    ["2e-5", "∞", "96.8%", ("0.612", RED, False), "34"],
    [("5e-5 ✔", RED, True), ("6.30", RED, True), ("96.4%", RED, True), "0.692", "67"],
    ["1e-4", "2.62", "95.4%", "0.730", ("3722", RED, True)],
    ["2e-4", "1.23", "83.1%", "0.647", ("3672", RED, True)],
], col_w=[1, 1.8, 1.5, 1, 1.2], size=12, row_h=0.33)
label(s, 7.0, 4.3, 5.9, "1e-4 比值掉到 2.62 且梯度跳 56 倍 → 排除。兩個 ∞ 是 underfit（acc 0.55 / 0.61）不是最佳。",
      12, align=PP_ALIGN.LEFT)
box(s, 0.6, 5.15, 12.3, 1.2, [
    ("兩臂最佳 lr 差 10 倍（LoRA 5e-6 / Prefix 5e-5）", RED, True),
    ("prefix 只能透過 softmax 權重作用，虛擬 token 與數百個真實 token 競爭注意力質量 → 回傳梯度天然就小", BLACK, False),
    ("所以「公平」定義成：各自在同等搜尋預算下調到最佳，而不是共用一個 lr", BLACK, False),
], size=13)

# ───────────── 11 · Experiment setup ─────────────
s = S()
title(s, "Experiment setup — 參數量精確配對")
label(s, 0.6, 1.15, 12.2, "prefix 參數量 = nvt × 32 層 × 2 (k,v) × 3072，剛好能與 LoRA 的兩種掛法對到完全相同的數字",
      14, align=PP_ALIGN.LEFT)
table(s, 0.6, 1.65, 12.2, [
    ["參數預算", "Prefix", "LoRA", "可訓練參數", "佔全模型"],
    [("1.57 M", RED, True), ("nvt = 8 + dropout 0.1  ← SVEN 尺度", RED, True), "（r=4 on qkv，未跑）",
     ("1,572,864", RED, True), "0.041%"],
    ["3.15 M", "nvt = 16", "r=8 on qkv_proj", "3,145,728", "0.082%"],
    ["12.58 M", "nvt = 64", "r=8 all-linear", "12,582,912", "0.329%"],
], col_w=[1.8, 4.0, 3.0, 2.2, 1.6], size=13, row_h=0.42)
label(s, 0.6, 3.5, 12.2, "為什麼 nvt=8：SVEN 的 n_prefix_token 隨模型大小縮放", 14, color=RED, align=PP_ALIGN.LEFT)
table(s, 0.6, 3.85, 6.0, [
    ["模型", "SVEN 的 n_prefix_token"],
    ["codegen-350M", "5"],
    ["codegen-2B", "8"],
    ["codegen-6B", "12"],
    [("Phi-3-mini 3.8B（內插）", RED, True), ("約 9 ~ 10", RED, True)],
], col_w=[3.5, 2.5], size=12.5, row_h=0.32)
box(s, 7.0, 3.85, 5.9, 1.6, [
    ("先前跑的 nvt=16 / 64 都在 SVEN 設計範圍之外", RED, True),
    ("nvt=64 是 SVEN 上限的 5.3 倍", BLACK, False),
    ("而 utility 損害隨 nvt 單調上升", BLACK, False),
    ("→ 不測 SVEN 尺度等於沒測 SVEN 的方法", BLACK, True),
], size=12.5, outline=RED)
label(s, 0.6, 5.65, 12.2, "全部固定的設定（唯一允許不同的是 --peft_method 與各自 sweep 出來的 lr）", 13,
      align=PP_ALIGN.LEFT)
table(s, 0.6, 6.0, 12.2, [
    ["objective", "DPO, β = 0.05", "訓練量", "800 steps @ batch 64"],
    ["max_length / prompt", "2048 / 1024", "max_grad_norm", "0.3"],
    ["seed", "42", "prefix 初始化", "init_weights = zero（同 SVEN）"],
], col_w=[2.6, 3.4, 2.6, 3.6], size=12, head=False, row_h=0.3)

# ───────────── 12 · Training dynamics ─────────────
s = S()
title(s, "Training dynamics — 訓練是否正常")
table(s, 0.6, 1.3, 12.2, [
    ["臂", "參數", "chosen early → late", "保留/token", "margin late", "margin 成長", "acc late", "grad 峰值"],
    ["LoRA all-linear", "12.6M", "−0.031 → −2.707", "88.8%", ("4.113", RED, True), "+3385%", ("0.975", RED, True), "87"],
    ["LoRA qkv_proj", "3.1M", "−0.014 → −0.530", ("97.7%", RED, True), "1.611", "+3732%", "0.885", "71"],
    ["prefix nvt=8", "1.57M", ("−0.451 → −0.731", RED, True), "96.8%", "0.407", "—", "0.710", "92,320"],
    ["prefix nvt=16", "3.1M", "−0.770 → −1.270", "94.6%", "0.822", "+340%", "0.752", "18,657"],
    ["prefix nvt=64", "12.6M", ("−4.441 → −2.930", RED, True), "87.9%", "1.323", ("+66%", RED, True), "0.740", "282"],
], col_w=[2.2, 1.0, 2.6, 1.4, 1.3, 1.4, 1.1, 1.2], size=11.5, row_h=0.34)
label(s, 0.6, 3.6, 12.2, "機制：prefix 的代價來自「與長度成正比的初始擾動」", 15, color=RED, align=PP_ALIGN.LEFT)
table(s, 0.6, 3.95, 9.5, [
    ["nvt", "chosen early（每 token 機率）", "最終保留率", "HumanEval Δ", "語法完整率 Δ"],
    ["8", ("−0.4505（98.0%）", RED, True), "96.8%", "−4.27", "−0.40 ✔"],
    ["16", "−0.7700（96.7%）", "94.6%", "−10.98", ("−5.01 ✘", RED, True)],
    ["64", ("−4.4414（82.3%）", RED, True), "87.9%", ("−32.93", RED, True), "未測"],
], col_w=[0.8, 3.0, 1.8, 1.9, 2.0], size=12.5, row_h=0.35)
box(s, 0.6, 5.65, 12.2, 1.15, [
    ("chosen early 是模型還沒學到任何東西時就已經付出的偏離——三條線完全單調", BLACK, True),
    ("nvt=64 的最終 margin 1.323 裡有 0.796 在 early 就存在，之後只成長 +66%（nvt=16 成長 +340%）", BLACK, False),
    ("加長 prefix 主要在加大擾動，不是增加學習能力。這正是 SVEN 把 n_prefix_token 選在 5~12 的原因。", RED, True),
], size=12.5)

# ───────────── 13 · Security result ─────────────
s = S()
title(s, "Security — 全量評測（693 × 10 × 5 語言）", 30)
label(s, 0.6, 1.15, 12.2, "四臂共用一份 OFF · max_new_tokens 2048 · n = 6,930/臂 · 標準誤 0.60 pt · 依論文算法：逐語言算再取平均",
      13, align=PP_ALIGN.LEFT)
table(s, 0.6, 1.6, 12.2, [
    ["臂", "c", "cpp", "java", "js", "py", "平均", "Δ", "相對"],
    ["OFF (base)", "69.41", "26.62", "61.50", "51.53", "30.64", "47.94", "—", "—"],
    [("LoRA all-linear", RED, True), "50.59", ("8.38", RED, True), "50.59", "36.35", "22.57",
     ("33.70", RED, True), ("−14.24", RED, True), ("−29.7%", RED, True)],
    ["LoRA qkv_proj", "66.08", "22.97", "60.21", "47.53", "30.03", "45.36", "−2.58", "−5.4%"],
    ["prefix nvt=8", "66.67", "24.19", "55.56", "48.47", "29.83", "44.94", "−3.00", "−6.2%"],
    ["prefix nvt=16", "59.22", "22.70", "54.81", "43.76", "28.99", "41.90", "−6.04", "−12.6%"],
], col_w=[2.4, 1.1, 1.1, 1.1, 1.1, 1.1, 1.2, 1.2, 1.2], size=12, row_h=0.36)
label(s, 0.6, 4.0, 12.2, "論文的安全性數字復現成功", 16, color=RED, align=PP_ALIGN.LEFT)
table(s, 0.6, 4.4, 8.0, [
    ["", "base", "after", "Δ", "相對"],
    ["論文 ProSec（LoRA + SimPO）", "50.57", "33.47", "−17.10", "−33.8%"],
    [("我們 LoRA all-linear（DPO）", RED, True), "47.94", ("33.70", RED, True), "−14.24", "−29.7%"],
], col_w=[3.4, 1.2, 1.2, 1.1, 1.1], size=13, row_h=0.38)
box(s, 9.0, 4.4, 3.9, 1.2, [
    ("ON 的絕對值只差 0.22 pt", RED, True),
    ("而且我們用 DPO、論文用 SimPO", BLACK, False),
    ("base 低 2.63 pt 來自子集重建差異", BLACK, False),
], size=12)
box(s, 0.6, 5.85, 12.3, 0.95, [
    ("但逐語言散布很大，平均吻合有一部分是巧合：C 我們較差（50.59 vs 44.27）、JS 較差（36.35 vs 28.21）、C++ 好很多（8.38 vs 20.74）", BLACK, True),
    ("論文裡必須把逐語言一起報，不能只報平均。", RED, True),
], size=12.5, outline=RED)

# ───────────── 14 · Utility result ─────────────
s = S()
title(s, "Utility — MultiPL-E（C++ / JS / PY）")
label(s, 0.6, 1.15, 12.2, "greedy · max_new_tokens 2048 · 每語言 161~164 題 · 標準誤 2.23 pt", 13,
      align=PP_ALIGN.LEFT)
table(s, 0.6, 1.6, 9.5, [
    ["臂", "C++", "JS", "PY", "平均 Δ"],
    ["OFF (base)", "46.58", "59.63", "70.73", "—"],
    ["LoRA all-linear", "−10.56", "+2.48", "−4.27", "−4.12"],
    [("LoRA qkv_proj", RED, True), "−2.48", ("+4.35", RED, True), ("+1.22", RED, True), ("+1.03 ✔", RED, True)],
    ["prefix nvt=8", "−9.31", "−8.08", "−4.27", "−7.22"],
    ["prefix nvt=16", ("−22.98", RED, True), "−16.77", "−10.98", ("−16.91", RED, True)],
    ["論文（SimPO）", "+2.44", "+0.98", "+4.89", "+2.77"],
], col_w=[2.6, 1.7, 1.7, 1.7, 1.8], size=12.5, row_h=0.36)
box(s, 10.4, 1.95, 2.5, 1.5, [
    ("lora_qkv 是唯一", RED, True), ("utility 為正的一臂", RED, True), ("", BLACK, False),
    ("也是唯一與論文同號", BLACK, False),
], size=12)
label(s, 0.6, 4.35, 12.2, "兩個軸無法同時復現 —— 最重要的發現", 16, color=RED, align=PP_ALIGN.LEFT)
table(s, 0.6, 4.75, 12.2, [
    ["臂", "安全性 Δ（SE 0.60）", "Utility Δ（SE 2.23）", "退化守門", "Pareto"],
    [("LoRA all-linear", RED, True), ("−14.24", RED, True), "−4.12", "⚠ python 未過", ("✔ 前緣", RED, True)],
    [("LoRA qkv_proj", RED, True), "−2.58", ("+1.03", RED, True), "✔ 全過", ("✔ 前緣", RED, True)],
    ["prefix nvt=8", "−3.00", "−7.22", "✔ 全過", "✘ 被支配"],
    ["prefix nvt=16", "−6.04", "−16.91", "✘ 未過", "✘ 被支配"],
], col_w=[2.6, 2.8, 2.8, 2.2, 1.8], size=12.5, row_h=0.36)
box(s, 0.6, 6.4, 12.3, 0.92, [
    ("all-linear 復現安全性但付出 utility 代價；qkv_proj 復現 utility 方向但只買到五分之一安全性。論文從未說明 target modules，", BLACK, False),
    ("能同時達成的設定可能落在兩者之間 —— per-CWE 證據指向 MLP：CWE-338 上 all-linear 移動 22.13 pt，qkv 只有 1.27 pt。", RED, True),
], size=12)

# ───────────── 15 · Analysis ─────────────
s = S()
title(s, "結果分析")
label(s, 0.6, 1.1, 12.2, "① Pareto 是推導出來的，要連同 margin 一起報", 15, color=RED, align=PP_ALIGN.LEFT)
table(s, 0.6, 1.45, 9.5, [
    ["主張", "安全性差距", "Utility 差距"],
    ["all-linear 支配 prefix nvt=16", "8.20 pt（13.8 SE）", "12.79 pt（5.7 SE）"],
    [("all-linear 支配 prefix nvt=8", RED, True), "11.24 pt（18.9 SE）", ("3.10 pt（1.4 SE）", RED, True)],
], col_w=[4.5, 2.5, 2.5], size=12.5, row_h=0.36)
label(s, 0.6, 2.6, 12.2,
      "第二條主張有一半靠一個統計上分不開的 utility 差距。精確說法：兩者 utility 代價無法區分，而 all-linear 安全性高 11.24 pt。",
      12.5, align=PP_ALIGN.LEFT)
label(s, 0.6, 3.05, 12.2, "② prefix 唯一勝出的一項：參數效率", 15, color=RED, align=PP_ALIGN.LEFT)
table(s, 0.6, 3.4, 7.5, [
    ["臂", "參數", "安全性 Δ", "每百萬參數 pt"],
    [("prefix nvt=8", RED, True), "1.57M", "−3.00", ("1.91", RED, True)],
    ["LoRA all-linear", "12.6M", "−14.24", "1.13"],
    ["LoRA qkv_proj", "3.15M", "−2.58", "0.82"],
], col_w=[2.2, 1.4, 1.8, 2.1], size=12.5, row_h=0.34)
box(s, 8.4, 3.4, 4.5, 1.4, [
    ("但這只在「參數預算是硬限制」", BLACK, False),
    ("的情境下才有意義 ——", BLACK, False),
    ("在 (安全性, utility) 平面上", RED, True),
    ("prefix 仍然被支配", RED, True),
], size=12.5)
label(s, 0.6, 5.0, 12.2, "③ 退化守門：只有兩臂全過", 15, color=RED, align=PP_ALIGN.LEFT)
table(s, 0.6, 5.35, 12.2, [
    ["臂", "整體", "逐語言"],
    [("LoRA qkv_proj", RED, True), "✔", "五語言全過，且語法完整率上升、無 fence 率降 11.31"],
    [("prefix nvt=8", RED, True), "✔", "五語言全過（程式碼長度保留 100.1%）"],
    ["LoRA all-linear", "整體 −2.61 通過", "python 未過（−5.07）→ 該語言的 22.57 不得單獨回報"],
    ["prefix nvt=16", "✘ −3.76", "c / cpp / java / python 皆未過，只有 js 過"],
], col_w=[2.4, 2.4, 7.4], size=12, row_h=0.34)

# ───────────── 16 · Future work ─────────────
s = S()
title(s, "Future work")
label(s, 0.6, 1.1, 12.2, "① 先補完的兩件事", 15, color=RED, align=PP_ALIGN.LEFT)
table(s, 0.6, 1.45, 12.2, [
    ["項目", "為什麼需要", "成本"],
    ["P-sven-attr：nvt=8 無 dropout",
     "nvt=8 那輪同時改了長度與 dropout，−10.98 → −4.27 的改善無法歸因",
     "1 次訓練 · 約 2.5 h"],
    ["探測 attention-only / MLP-only",
     "論文沒說 target modules，兩個極端各復現一個軸；qkv+o (4.7M) 或 gate_up+down (7.9M) 是自然探測點",
     "各 1 次訓練"],
], col_w=[3.6, 6.6, 2.0], size=12, row_h=0.5)
label(s, 0.6, 3.0, 12.2, "② 方法上的延伸", 15, color=RED, align=PP_ALIGN.LEFT)
box(s, 0.6, 3.35, 6.0, 2.5, [
    ("Masked DPO", BLACK, True),
    ("• 動機：edit ratio ρ 中位數 0.358 → 兩版之間有 64% 的", BLACK, False),
    ("  token 完全相同，對偏好訊號沒貢獻卻完整計入 loss", BLACK, False),
    ("• 做法：token-level diff → mask，只對 diff token 算 loss", BLACK, False),
    ("• 外部驗證：PTC (DSN 2025) 用同樣的 diff mask，", BLACK, False),
    ("  no control 55.0% → line 71.3% → mixed 86.7%", BLACK, False),
    ("• 前置檢查：diff 散成中位數 17 個 hunk，須先確認", RED, False),
    ("  它們有多少落在分析器標記的漏洞行上", RED, False),
], size=12)
box(s, 7.0, 3.35, 5.9, 2.5, [
    ("CWE-specific 多 Prefix", BLACK, True),
    ("• 與參數效率的結論相合：一組模組只要 1.57M / ≈3MB", BLACK, False),
    ("• 做法：D_sec 依 CWE 家族分組各訓一組，", BLACK, False),
    ("  再做 N×N 遷移矩陣（對角=專用增益、非對角=負遷移）", BLACK, False),
    ("• 必要對照：同樣大小的隨機混合子集訓一個通用 prefix，", RED, False),
    ("  否則分不出「專用有效」還是「資料變少」", RED, False),
    ("• 要量的是切換延遲與同 batch 混合 policy 的吞吐量 ——", RED, False),
    ("  「能訓多組」不算賣點，LoRA 也能", RED, False),
], size=12)
box(s, 0.6, 6.05, 12.3, 1.05, [
    ("論文的措辭建議", RED, True),
    ("在 P-sven-attr 跑完前應寫成「在 nvt = 8 / 16 這兩個設定下，prefix 在 (安全性, utility) 平面上被 LoRA 支配」，", BLACK, False),
    ("不要寫成「prefix tuning 較差」—— lr 未在全長驗證、只有單一 seed、dropout 未歸因，這三條都還開著。", BLACK, False),
], size=12, outline=RED)

out = "docs/ProSec進度報告_v3.pptx"
prs.save(out)
print("saved:", out, "| slides:", len(prs.slides._sldIdLst))
