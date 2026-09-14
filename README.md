# OCTD — 可复现材料

本文档与代码配套论文：

**《Oracle 标定的三层可预测性上界分解（OCTD）：知识追踪数据可量化诊断方法》**

核心方法：在受控生成数据上把知识追踪（KT）的可预测性上限分解为三层——
**Ⅰ Oracle 上界**（pOk 名义上界 / 严格因果下界）、**Ⅱ 可提取上界**（容量/结构扫描）、**Ⅲ 现实模型**。
然后用两级决策规则（gap①＝Ⅰ−Ⅱ、gap②＝Ⅱ−Ⅲ）把改进方向归因为 **A 换数据 / B 加特征 / C 换模型**，并通过跨分布标定排除单一分布的偶然。

本仓库只归档**论文与训练的可复现材料**：源稿、图表/排版脚本、DL/自检脚本、训练与数据生成脚本、以及 §7 复现清单引用的关键数据子集。
不包含训练中间检查点（`*.pt`/`*.onnx`）与超大原始数据集（若需原始 Junyi/ASSISTments/XES3G5M，请按 §7 指明的原始来源获取）。

## 目录结构

```
paper/       论文源稿与排版
  paper.md       正文（LaTeX 数学符号 + Markdown 源）
  references.bib 参考文献（GB/T 7714）
  build_paper.py / build_docx.py / verify_docx.py   重建并核验 paper.docx
  make_figs.py   重新生成 paper/figs 下全部图表
  delong_test.py、bootstrap_ci.py、selfcheck_sim.js  统计显著性与自检
  NEXT_TODO.md   开放项与收口记录
  _run/          第三方生成器（DINA/pyBKT/KS-ELO）、IRT 代理、口径对齐等脚本
  figs/          六张图（three_layer / auc_ablation / architecture /
                 cross_calib / diff_x_cap / controlled_matrix）
training/    模型训练与数据生成
  train_dkt.py   主训练脚本（--arch gru/sakt/saint/akt/gkt/dkvmn，--split-mode fixed）
  requirements.txt
  oced_toy/      最小可运行演示
  data/          关键源数据与生成脚本（见下）
  data/junyi/    真实 Junyi 对齐口径子集与重建脚本
```

## 复现步骤（概述）

1. **重建论文与图表**
   ```bash
   cd paper
   python make_figs.py          # 生成 figs/*.png|svg
   python build_paper.py        # 生成 paper.docx
   python verify_docx.py        # 校验格式与关键数据
   ```
2. **训练主模型（合成 p0_prereq，固定 split 多 seed）**
   ```bash
   pip install -r training/requirements.txt
   python training/train_dkt.py --arch gru --split-mode fixed --seed 1 .. 5
   ```
3. **跨架构饱和扫描**
   ```bash
   python training/train_dkt.py --arch {sakt,saint,akt,gkt,dkvmn} --split-mode fixed
   ```
4. **跨分布标定 / 三层分解**
   - `paper/_run/`: `junyi_irt_proxy.py`（真实侧 Ⅰ′ IRT 代理）、`oracle_aligned_test.py`（Oracle 口径 A/B）、`dina_generate.py` / `pybkt_generate.py` / `ksgen_eval.py`（第三方受控矩阵）。

口径与判定规则详见 `paper/paper.md` §4–§5；统计检验见 `delong_test.py`（gap① DeLong Z 检验 + 配对 Bootstrap）。

## 数据说明

`training/data/` 仅收录复现所需的关键子集（JSONL + split 均为少量小文件）：

- `p0_prereq.jsonl` / `.split.json` —— 主合成数据（固定测试集 n=2880）
- `p1_prereq.*`、`dina_prereq.*`、`pybkt_prereq.*`、`ksgen_prereq.*`(+`m50`) —— 跨分布标定
- `junyi/` —— 真实 Junyi 对齐口径：`junyi_aligned_fa2.jsonl`、`fa2_200000_7_diff.jsonl`、`fa2.split.json`、IRT `*.npy`、重建脚本（`junyi_rebuild_aligned.py` 等）

不包含：`*.pt`/`*.onnx` 检查点、`junyi.rar`/`junyi_rt_*`(≥700MB)、`srclog` 原始 CSV(2.6GB)、`xes3g5m_dkt_dataset.txt`(62MB) 等超大文件。需要完整原始数据时请运行 `junyi_rebuild_aligned.py` 等脚本从官方源重建。

## 许可

Apache License 2.0（见 `LICENSE`）。