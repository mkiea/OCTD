# -*- coding: utf-8 -*-
"""
论文图稿生成脚本（可复现）
=========================
生成 paper/figs/ 下六张图：
  1. three_layer.png      —— 三层分解示意（I Oracle / II 可提取 / III 模型 + gap + 严格因果下界参考线）
  2. auc_ablation.png     —— 各模型/基线/配置 AUC 对比条形图
  3. architecture.png     —— DKT/HybridKT 架构示意
  4. cross_calib.png      —— 跨分布标定对比（合成/真实多数据源并置）
  5. diff_x_cap.png       —— 真实 Junyi「难度 × 容量」二维扫描
  6. controlled_matrix.png—— 第三方受控矩阵（DINA / pyBKT / KS-ELO）

数据来源：paper/paper.md §5 实测与闭环计划方向⑥d/⑦c/⑦d 记录（脚本内固化，便于复现）。
运行：python paper/make_figs.py
图内标签用英文，避免缺字体的乱码。所有数学符号用真实 Unicode 字符（≈、⊕、→）。
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

# 配色
C1, C2, C3 = "#4C72B0", "#55A868", "#D55E00"  # 蓝 / 绿 / 橙

# ---------- 图 1：三层分解 ----------
def fig_three_layer():
    fig, ax = plt.subplots(figsize=(9, 4.4))
    layers = [
        ("I  Oracle upper bound   (nominal-oracle: pOk[t+1] vs y[t+1])", 0.759, C1),
        ("II Extractable          (capacity scan)", 0.717, C2),
        ("III Model               (DKT, 5-seed)", 0.717, C3),
    ]
    for i, (name, val, c) in enumerate(layers):
        y = 2 - i
        ax.barh(y, val, color=c, height=0.5, alpha=0.9)
        # 数值标在条内右端（白字、加粗），避开右侧 gap 标注区
        ax.text(val - 0.006, y, f"{val:.3f}", ha="right", va="center",
                fontsize=10, color="white", fontweight="bold")
        ax.text(-0.004, y + 0.23, name, ha="right", va="center", fontsize=10)
    # gap 1：II→I 之间（条间空白带 y≈1.5），标签放箭头上方，不与数值重叠
    ax.annotate("", xy=(0.717, 1.5), xytext=(0.759, 1.5),
                arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.2))
    ax.text(0.738, 1.57, "gap 1 ≈ 0.042\n(observability gap)", ha="center",
            fontsize=9, color="0.2")
    # gap 2：III→II 之间（y≈0.5）
    ax.annotate("", xy=(0.710, 0.5), xytext=(0.717, 0.5),
                arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.2))
    ax.text(0.7135, 0.57, "gap 2 ≈ 0", ha="center", fontsize=9, color="0.2")
    # 严格因果下界参考线：pOk[t] vs y[t+1]，提示生成器每步近独立
    ax.axvline(0.549, ls="--", c="#aa3333", lw=1.3)
    ax.text(0.549 + 0.008, 2.40, "strict causal lower bound  0.549\n(pOk[t] -> y[t+1])",
            fontsize=9, color="#aa3333", va="bottom")
    ax.set_xlim(0, 0.85)
    ax.set_ylim(-0.5, 2.65)
    ax.set_xlabel("AUC", fontsize=12)
    ax.set_title("Oracle 3-layer decomposition (synthetic KT, p0_prereq)", fontsize=13)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "three_layer.svg"))
    fig.savefig(os.path.join(OUT, "three_layer.png"), dpi=200)
    plt.close(fig)

# ---------- 图 2：AUC 对比 ----------
def fig_auc_ablation():
    # 主结果（合成 p0_prereq，固定 split）
    labels = ["BKT\n(baseline)", "DKT\n(no difficulty variance)", "DKT c0\n[ours, 5-seed]",
              "+ attention\n(capacity scan)",
              "GRU\n(cross-arch)", "SAKT\n(attn)", "GKT\n(graph)", "DKVMN\n(memory)",
              "Oracle\n(upper bound)"]
    vals = [0.572, 0.594, 0.717, 0.710, 0.7189, 0.7154, 0.7148, 0.7038, 0.759]
    errs = [0,     0,     0.001, 0,      0,      0.0016, 0.0027, 0.0008, 0]
    colors = ["#c55", "#dd8452", "#55A868", "#7f7f7f",
              "#4C72B0", "#9999CC", "#17becf", "#e377c2", C1]
    fig, ax = plt.subplots(figsize=(13, 4.6))
    bars = ax.bar(labels, vals, yerr=errs, capsize=3, color=colors, alpha=0.9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", fontsize=8)
    ax.axhline(0.75, ls="--", c="0.55", lw=1)
    ax.text(4.3, 0.755, "break-0.75 target", fontsize=9, color="0.4", ha="center")
    ax.set_ylim(0, 0.84)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title("AUC across baselines / configurations (fixed test set)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "auc_ablation.png"), dpi=200)
    plt.close(fig)

# ---------- 图 3：架构示意 ----------
def _box(ax, x, y, w, h, text, fc, ec="black", fs=10, tc="black"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                       fc=fc, ec=ec, lw=1.2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)

def _arr(ax, x1, y1, x2, y2, color="black", style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=14, lw=1.4, color=color))

def fig_architecture():
    # 重新布局：加大画布、融合层与 GRU 加宽加高、文字用真实字符且足够空间，避免文字重叠
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 6)
    ax.axis("off")
    # 输入（4 个，纵向堆叠）
    _box(ax, 0.2, 4.55, 1.6, 0.7, "seq_topic", "#9ecae1", fs=10)
    _box(ax, 0.2, 3.45, 1.6, 0.7, "seq_correct", "#9ecae1", fs=10)
    _box(ax, 0.2, 2.35, 1.6, 0.7, "seq_diff", "#9ecae1", fs=10)
    _box(ax, 0.2, 1.15, 1.6, 0.7, "latency / pres\n(optional)", "#c6dbef", fs=8.5)
    # 融合层（够宽，3 行文字）
    _box(ax, 2.3, 2.55, 2.4, 1.8,
         "concat & project\n(topic ⊕ correct\n⊕ diff)  (+ opt.)",
         "#deebf7", fs=9)
    # GRU 序列编码
    _box(ax, 5.2, 2.75, 1.7, 1.2, "GRU\nsequence\nencoder", "#fdd0a2", fs=9)
    # 可选注意力
    _box(ax, 7.2, 3.1, 1.3, 0.7, "attn\n(optional)", "#ffe699", fs=9)
    # 三个头
    _box(ax, 9.0, 4.4, 1.7, 0.8, "mastery head", "#d5e8f0", fs=10)
    _box(ax, 9.0, 3.0, 1.7, 0.8, "next head\n(+ target diff)", "#d5e8f0", fs=9)
    _box(ax, 9.0, 1.4, 1.7, 0.8, "ability head", "#d5e8f0", fs=10)
    # 输出
    _box(ax, 11.0, 4.4, 1.3, 0.8, "mastery_score", "#fff2cc", fs=8.5)
    _box(ax, 11.0, 3.0, 1.3, 0.8, "p_next_correct", "#fff2cc", fs=8.5)
    _box(ax, 11.0, 1.4, 1.3, 0.8, "ability_level", "#fff2cc", fs=8.5)
    # 连线（输入 → 融合）
    _arr(ax, 1.8, 4.9, 2.3, 4.0)
    _arr(ax, 1.8, 3.8, 2.3, 3.6)
    _arr(ax, 1.8, 2.7, 2.3, 3.3)
    _arr(ax, 1.8, 1.5, 2.3, 3.0)
    # 融合 → GRU → 可选 attn → 三个头
    _arr(ax, 4.7, 3.45, 5.2, 3.35)
    _arr(ax, 6.9, 3.35, 7.2, 3.45)
    _arr(ax, 8.5, 3.45, 9.0, 4.8)
    _arr(ax, 8.5, 3.45, 9.0, 3.4)
    _arr(ax, 8.5, 3.45, 9.0, 1.8)
    # 头 → 输出
    _arr(ax, 10.7, 4.8, 11.0, 4.8)
    _arr(ax, 10.7, 3.4, 11.0, 3.4)
    _arr(ax, 10.7, 1.8, 11.0, 1.8)
    ax.set_title("DKT / HybridKT architecture (3-in / 3-out ONNX contract)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "architecture.svg"))
    fig.savefig(os.path.join(OUT, "architecture.png"), dpi=200)
    plt.close(fig)

# ---------- 图 4：跨分布标定（天花板性质判定）+ 抖动对比带 ----------
def fig_cross_calib():
    # 同一 DKT 管线，6 个数据分布（读取自 paper.md §5.2.1 表 + §5.3.2）
    labels = ["ASSISTments2009\n(real)", "pyBKT\n(3rd-party)", "KS-ELO\n(3rd-party)",
              "ours p0_prereq\n(synthetic)", "JunyiAcademy\n(real)",
              "ktbd synthetic\n(3rd-party)"]
    vals = [0.675, 0.6937, 0.6993, 0.7191, 0.8020, 0.824]
    errs = [0.0006, 0.0002, 0.0015, 0.0002, 0.0001, 0.010]
    colors = ["#d45a5a", "#8172B3", "#CCB974", "#4C72B0", "#9999CC", "#55A868"]
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    bars = ax.bar(labels, vals, yerr=errs, capsize=3, color=colors, alpha=0.9,
                  width=0.62)
    no_oracle = {0, 4, 5}   # ASSIST2009 / Junyi / ktbd：pOk 不可观测，仅 Ⅱ/Ⅲ（ktbd 用 IRT 近似）
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.009, f"{v:.3f}{'†' if i in no_oracle else ''}",
                ha="center", fontsize=10)
    ax.axhline(0.717, ls="--", c="0.4", lw=1)
    # 天花板标注放左侧空白区（ASSIST 顶≈0.675 之上），用引线指到虚线，避免压住 0.719
    ax.annotate("extractable ≈ 0.717 (ours)", xy=(4.0, 0.717), xytext=(0.15, 0.742),
                fontsize=9, color="0.3",
                arrowprops=dict(arrowstyle="-", color="0.4", lw=0.9,
                                shrinkA=2, shrinkB=2))
    # —— 跨分布浮动 vs 单分布 3-seed 抖动 对比带 ——
    sigma_x1p5 = 0.003  # 单分布内 3-seed 抖动 ×1.5 ≈ ±0.003
    floor_y, band_h = 0.578, 2 * sigma_x1p5
    ax.axhspan(floor_y, floor_y + band_h, xmin=0.03, xmax=0.97,
               color="0.78", alpha=0.7)
    ax.text(2.5, floor_y + band_h + 0.006,
            "within-distribution 3-seed jitter x1.5 = ±0.003", ha="center",
            fontsize=8.5, color="0.35")
    ax.annotate('', xy=(5.55, floor_y), xytext=(-0.42, floor_y),
                arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.4,
                                shrinkA=40, shrinkB=0))
    ax.text(2.5, floor_y - 0.018, "across-distribution spread 0.675 -> 0.824",
            ha="center", va="top", fontsize=9.5, color='#333333')
    ax.set_ylim(0.52, 0.865)
    ax.set_ylabel("DKT AUC (same pipeline, 3-seed)", fontsize=11)
    ax.set_title("Cross-distribution calibration:\nceiling is data-driven, not a model ceiling\n"
                 "† = Ⅱ/Ⅲ only (no true Oracle); ktbd 0.824 uses IRT-fitted pOk",
                 fontsize=11.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cross_calib.png"), dpi=200)
    plt.close(fig)

# ---------- 图 5：真实 Junyi「难度 × 容量」二维扫描 ----------
def fig_diff_x_cap():
    # 读取自 paper.md §5.5 「难度 × 容量」二维表（真实 Junyi 双版本，共享固定测试集，3-seed）
    caps = ["c0\nGRU 32/16", "c2\nGRU 64/32", "c4\nGRU 256/64\n+ attn"]
    X = [0, 1, 2]
    w = 0.36
    diff = [0.7455, 0.7455, 0.8053];   diff_e = [0.0003, 0.0004, 0.0012]
    nodiff = [0.7233, 0.7236, 0.7952]; nodiff_e = [0.0001, 0.0005, 0.0012]
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    b1 = ax.bar([v - w / 2 for v in X], diff, w, yerr=diff_e, capsize=3,
                label="real difficulty (diff)", color=C1, alpha=0.9)
    b2 = ax.bar([v + w / 2 for v in X], nodiff, w, yerr=nodiff_e, capsize=3,
                label="constant difficulty = 3 (nodiff)", color="#DD8452", alpha=0.9)
    for b, v in zip(list(b1) + list(b2), diff + nodiff):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.4f}",
                ha="center", fontsize=9)
    # 二级增益标注：c0 档难度增益 +0.022（diff−nodiff）；c0→c4 结构增益 nodiff +0.072
    ax.annotate('', xy=(0.82, 0.775), xytext=(0.18, 0.775),
                arrowprops=dict(arrowstyle='<-', color='0.35', lw=1.2))
    ax.text(0.5, 0.786, "difficulty gap (c0)\n+0.022", ha="center", fontsize=8, color="0.35")
    ax.annotate('', xy=(2.23, 0.862), xytext=(-0.23, 0.862),
                arrowprops=dict(arrowstyle='<->', color='#333', lw=1.3))
    ax.text(1.0, 0.872, "structure bump c0->c4 (nodiff): +0.072,\nnot difficulty-gated",
            ha="center", fontsize=8.5, color="#333")
    ax.set_xticks(X); ax.set_xticklabels(caps)
    ax.set_ylim(0.70, 0.895)
    ax.set_ylabel("AUC (same pipeline, fixed test set, 3-seed)", fontsize=10.5)
    ax.set_title("Difficulty x capacity on real Junyi:\ncap bump is structure-driven, not difficulty-gated",
                 fontsize=12)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "diff_x_cap.png"), dpi=200)
    plt.close(fig)

# ---------- 图 6：第三方受控矩阵三机制可用其上界对比（§5.3.2） ----------
def fig_controlled_matrix():
    # 受控矩阵（n_skills=50 拉齐），数据来自 paper.md §5.3.2 表
    mechanisms = ["DINA\n(cognitive\nconjunctive)", "pyBKT\n(BKT-HMM)", "KS-ELO\n(ELO+prereq\n+forgetting)"]
    oracle = [0.6808, 0.7142, 0.7438]
    ext    = [0.6548, 0.6996, 0.7416]     # Ⅱ(c0)，模型可提取上界
    ext_e  = [0.0010, 0.0010, 0.0007]
    X = [0, 1, 2]; w = 0.32
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    b1 = ax.bar([v - w / 2 for v in X], oracle, w, color="#C44E52",
                label="I  Oracle upper bound (pOk)", alpha=0.9)
    b2 = ax.bar([v + w / 2 for v in X], ext, w, yerr=ext_e, capsize=3,
                color=C2, label="II  extractable upper bound (c0)", alpha=0.9)
    # 数值：Oracle 标签抬高、extractable 标签压低，避免 KS-ELO 两数粘连
    for b, v in zip(list(b1), oracle):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.018, f"{v:.4f}",
                ha="center", fontsize=9)
    for b, v, e in zip(list(b2), ext, ext_e):
        ax.text(b.get_x() + b.get_width() / 2, v + e + 0.001, f"{v:.4f}",
                ha="center", fontsize=9)
    # gap① 标注：DINA 最大（≈0.026），KS-ELO 最小（≈0.002）
    for x, o, e_, lab in zip(X, oracle, ext, ["gap 1\n≈0.026", "gap 1\n≈0.015", "gap 1\n≈0.002"]):
        ax.annotate("", xy=(x + w / 2, e_), xytext=(x - w / 2, o - 0.004),
                    arrowprops=dict(arrowstyle='->', color='0.45', lw=1))
    ax.set_xticks(X); ax.set_xticklabels(mechanisms, fontsize=9)
    ax.set_ylim(0.60, 0.80)
    ax.set_ylabel("AUC (same DKT pipeline, fixed split, 3-seed)", fontsize=10.5)
    ax.set_title("Controlled 3rd-party matrix (n_skills=50):\nextractable ceiling separates across mechanisms",
                 fontsize=11.5)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "controlled_matrix.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_three_layer()
    fig_auc_ablation()
    fig_architecture()
    fig_cross_calib()
    fig_diff_x_cap()
    fig_controlled_matrix()
    print("figures written to:", OUT)
    print(sorted(os.listdir(OUT)))