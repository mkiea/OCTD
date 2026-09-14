# -*- coding: utf-8 -*-
"""
用 pandoc 将 paper.md 转成 paper.docx，并对宽表做可读性调优。

策略：
  - 转换：--reference-doc=reference.docx（学术模板）、--resource-path=. 以便找图
  - 表格：居中；列数越多字号越小（≥8 列 8pt、≥6 列 9pt，其余 10.5pt），
    缓解 A4 纵向排版下宽表单元格文字拥挤的问题（列宽仍由 pandoc 按页面宽度比例给出，不会溢出）。

运行：python build_paper.py
"""
import os
import docx
import pypandoc
from docx.shared import Pt
from docx.enum.table import WD_TABLE_ALIGNMENT

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "paper.md")
OUT = os.path.join(HERE, "paper.docx")
REF = os.path.join(HERE, "reference.docx")

pypandoc.convert_file(
    MD, "docx", outputfile=OUT,
    extra_args=["--resource-path=.", "--reference-doc=" + REF],
)

d = docx.Document(OUT)
for t in d.tables:
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    nc = len(t.columns)
    fs = 10.5
    if nc >= 8:
        fs = 8
    elif nc >= 6:
        fs = 9
    if fs < 10.5:
        for row in t.rows:
            for c in row.cells:
                for p in c.paragraphs:
                    for r in p.runs:
                        r.font.size = Pt(fs)
d.save(OUT)
print("paper.docx regenerated:", OUT, "tables:", len(d.tables))