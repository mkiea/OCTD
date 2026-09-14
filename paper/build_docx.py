# -*- coding: utf-8 -*-
"""
按中文期刊单栏版式，把 pandoc 默认 reference.docx 改造成学术 reference.docx。

改造项：
  - 页面：A4，上下 2.54cm、左右 3.17cm
  - 字体：正文 宋体（西文 Times New Roman）五号 10.5pt、两端对齐、首行缩进 2 字符、1.5 倍行距
  - 标题：主标题 黑体小二居中；各节 黑体加粗左对齐；小节逐级减小
  - 图题/表题：宋体小五 9pt 居中
  - 块引用（内部注释）：宋体八号灰色
  - 页脚：居中页码域
运行：python paper/build_docx.py   （生成 paper/reference.docx）
"""
import docx
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def set_font(style, east="宋体", west="Times New Roman", size=None,
             bold=None, color=None):
    """同时设置中西文字体（eastAsia + ascii/hAnsi 落在样式级）。"""
    f = style.font
    f.name = west
    rPr = style.element.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    rFonts.set(qn("w:eastAsia"), east)
    if size is not None:
        f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    if color is not None:
        f.color.rgb = color


def add_page_number_footer(section):
    section.footer.is_linked_to_previous = False
    for p in list(section.footer.paragraphs):
        p.text = ""
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run()
        r.font.name = "Times New Roman"
        rPr = r._r.get_or_add_rPr()
        rFonts = rPr.get_or_add_rFonts()
        rFonts.set(qn("w:eastAsia"), "宋体")
        r.font.size = Pt(9)
        fld1 = OxmlElement("w:fldChar"); fld1.set(qn("w:fldCharType"), "begin")
        instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve"); instr.text = " PAGE "
        fld2 = OxmlElement("w:fldChar"); fld2.set(qn("w:fldCharType"), "end")
        r._r.append(fld1); r._r.append(instr); r._r.append(fld2)


def main():
    doc = docx.Document(os.path.join(HERE, "default_ref.docx"))

    # ---------- 页面：A4 + 页边距 ----------
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.54)
    sec.bottom_margin = Cm(2.54)
    sec.left_margin = Cm(3.17)
    sec.right_margin = Cm(3.17)
    add_page_number_footer(sec)

    S = doc.styles

    # ---------- Normal：宋体/Times，1.5 倍行距、两端对齐 ----------
    n = S["Normal"]
    set_font(n, east="宋体", west="Times New Roman", size=10.5)
    n.paragraph_format.line_spacing = 1.5
    n.paragraph_format.space_before = Pt(0)
    n.paragraph_format.space_after = Pt(0)

    # ---------- 正文段落：首行缩进 2 字符（0.74cm） ----------
    for name in ("Body Text", "First Paragraph"):
        st = S[name]
        set_font(st, east="宋体", west="Times New Roman", size=10.5)
        st.paragraph_format.first_line_indent = Cm(0.74)
        st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        st.paragraph_format.line_spacing = 1.5

    # ---------- 标题 ----------
    t = S["Title"]
    set_font(t, east="黑体", west="Times New Roman", size=16, bold=True)
    t.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t.paragraph_format.space_before = Pt(12)
    t.paragraph_format.space_after = Pt(6)

    au = S["Author"]
    set_font(au, east="宋体", west="Times New Roman", size=11.5)
    au.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    au.paragraph_format.space_after = Pt(4)

    sub = S["Subtitle"]
    set_font(sub, east="宋体", west="Times New Roman", size=10.5)
    sub.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 各级标题：黑体加粗，无首行缩进
    h_cfg = {
        "Heading 1": (14, True), "Heading 2": (13, True),
        "Heading 3": (12, True), "Heading 4": (11, True),
        "Heading 5": (10.5, True), "Heading 6": (10, True),
    }
    for hname in ("Heading 1", "Heading 2", "Heading 3",
                  "Heading 4", "Heading 5", "Heading 6"):
        hs = S[hname]
        size, bold = h_cfg[hname]
        set_font(hs, east="黑体", west="Times New Roman", size=size, bold=bold,
                 color=RGBColor(0, 0, 0))
        hs.paragraph_format.first_line_indent = Cm(0)
        hs.paragraph_format.space_before = Pt(12 if hname in ("Heading 1", "Heading 2") else 6)
        hs.paragraph_format.space_after = Pt(4)

    # ---------- 图题 / 表题：宋体小五居中 ----------
    for cname in ("Caption", "Image Caption", "Table Caption"):
        try:
            cs = S[cname]
        except KeyError:
            continue
        set_font(cs, east="宋体", west="Times New Roman", size=9)
        cs.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cs.paragraph_format.space_before = Pt(3)
        cs.paragraph_format.space_after = Pt(3)
        cs.paragraph_format.first_line_indent = Cm(0)

    # 图容器居中
    for fname in ("Figure", "Captioned Figure"):
        try:
            fs = S[fname]
        except KeyError:
            continue
        fs.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ---------- 块引用（内部注释）：宋体八号浅灰 ----------
    bq = S["Block Text"]
    set_font(bq, east="宋体", west="Times New Roman", size=8.5,
             color=RGBColor(0x66, 0x66, 0x66))
    bq.paragraph_format.left_indent = Cm(0.5)
    bq.paragraph_format.right_indent = Cm(0.5)
    bq.paragraph_format.line_spacing = 1.3

    # ---------- 参考文献：宋体小五两端对齐 ----------
    bib = S["Bibliography"]
    set_font(bib, east="宋体", west="Times New Roman", size=8.5)
    bib.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    bib.paragraph_format.line_spacing = 1.3

    out = os.path.join(HERE, "reference.docx")
    doc.save(out)
    print("reference.docx written:", out)


if __name__ == "__main__":
    main()