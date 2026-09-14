#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DKT 经典合成生成器（Piech et al. 2015 chaining-lattice）+ pOk 透传 + Oracle(Ⅰ)
================================================================================
需求：在 pyBKT(BKT-HMM) 与 KS-ELO(ELO+前置条件+遗忘) 之外，找第三个机制独立、
且能逐位透传真概率 pOk 的第三方生成器，做完整三层分解的跨分布锚点。

DKT 经典合成机制（faithful re-implementation of Piech's learning-lattice）：
  - n_skill=15 排成 3×5 晶格；每个技能有两类前置依赖（left 同行左侧、up 上一行），
    并解锁 right 同行右侧 / down 下一行。
  - 技能掌握单调累积、无遗忘（与 KS-ELO 的遗忘、pyBKT 的 forget 刻意区分）。
  - 每一步以概率 stay 留在当前技能、否则沿晶格推进到 related 技能（right/down）。
  - 每技能有连续潜在掌握度 m_s∈[0,1]；答对则以学习率向 1 逼近，且仅在
    "该技能前置条件已掌握(pr_dep>θ)" 时生效 ⇒ 体现晶格前置依赖链。
  - 教师真概率 pOk[t] = guess + (1-slip-guess) * m_{s_t}，随潜在掌握度与晶格推进
    而连续演化 ⇒ 可逐位透传。

Oracle(Ⅰ) 口径与主管线 / pybkt_generate 完全对齐（同口径 n）：
  - 上帝视角（主，与 DKT 同目标）：pOk[t+1] -> correct[t+1]，t∈[0,L-2]
  - 严格因果（对照）：pOk[t] -> correct[t+1]  —— 若显著小于上帝视角，说明该生成器
    "下一步内部态"对该步结果几乎不可测，需在正交文中并列报告（呼应 oracle_dual_calibration）。

用法：python paper/_run/dktgrid_generate.py
"""
import os
import json
import random

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "training", "data")
SAVE = os.path.join(DATA, "dktgrid_piech.jsonl")
SPLIT = os.path.join(DATA, "dktgrid_piech.split.json")

N_SKILL = 15
ROWS, COLS = 3, 5


def build_lattice():
    """3×5 晶格。每技能依赖 left(同行) 与 up(上一行)，解锁 right/down。

    返回 deps[s]=list, unlocks[s]=list。
    """
    deps = {s: [] for s in range(N_SKILL)}
    unlocks = {s: [] for s in range(N_SKILL)}
    for s in range(N_SKILL):
        r, c = divmod(s, COLS)
        if c > 0:                # left 前置
            deps[s].append(s - 1)
            unlocks[s - 1].append(s)
        if r > 0:                # up 前置
            deps[s].append(s - COLS)
            unlocks[s - COLS].append(s)
    return deps, unlocks


def gen_grid_student(deps, unlocks, stay, theta, T, rng, py):
    """单学生单序列。返回 (corrects, diffs, pOk)。

    为产出有效三层分解（模型 Ⅲ < Oracle Ⅰ）修正 v1 的 AUC=1.0 泄漏：
    v1 用技能级共享 guess/slip ⇒ 潜在空间小、不同行近重复，模型在 row-split 下
    记忆 test 模板。本版将噪声档、能力、学习步长全部**每学生独立抽取** ⇒ 每行唯一，
    模型无法跨行记忆；同时保留 Ⅰ 信号强度（pOk 动态范围 [guess, 1-slip]）。
    """
    m = np.zeros(N_SKILL)
    for s in range(N_SKILL):
        if py.random() < 0.30:      # 初始已掌握子集（DKT：学生已知部分技能）
            m[s] = float(rng.uniform(0.80, 0.95))
        else:
            m[s] = float(rng.uniform(0.0, 0.22))

    # 每学生独立参数（关键防泄漏：行唯一，无共享技能模板可记忆）
    guess = float(rng.uniform(0.18, 0.32))
    slip = float(rng.uniform(0.12, 0.24))
    learnp = float(rng.uniform(0.4, 0.7))

    s = py.randrange(N_SKILL)   # 起始技能：随机
    corrects, diffs, pOk = [], [], []
    for t in range(T):
        pm = guess + (1 - slip - guess) * m[s]
        p_ok = min(0.995, max(0.005, pm))
        ok = 1 if py.random() < p_ok else 0
        corrects.append(ok)
        diffs.append(py.randint(1, 5))
        pOk.append(float(round(p_ok, 4)))

        # 随机学习（答对且有概率推进，打破确定性复现）
        if ok == 1 and py.random() < 0.7:
            if len(deps[s]) > 0:
                pr_dep = float(np.mean([m[d] for d in deps[s]]))
                if pr_dep > theta:
                    m[s] = m[s] + learnp * (1.0 - m[s])
            else:
                m[s] = m[s] + learnp * (1.0 - m[s])

        # 技能推进：以 1-stay 概率沿晶格移动到 related 技能
        if py.random() < (1.0 - stay) and len(unlocks[s]) > 0:
            s = py.choice(unlocks[s])
    return corrects, diffs, pOk


def main():
    seed = 0
    rng = np.random.default_rng(seed)
    py = random.Random(seed)

    deps, unlocks = build_lattice()

    n_rows = 3840           # 与 p0_prereq / pybkt 同量级
    test_frac = 0.15        # row-split 固定留出（DKT-grid 学生轨迹 i.i.d.，学生级随机处理）

    stay = 0.80             # 留在当前技能概率（DKT 经典推进率）
    theta = 0.5             # 前置掌握阈值（链式解锁）

    rows = []
    for _ in range(n_rows):
        T = py.randint(8, 16)   # 与 pybkt 一致，体现学习曲线 + 防固定步长记忆
        corrects, diffs, pOk = gen_grid_student(
            deps, unlocks, stay, theta, T, rng, py)
        s0 = py.randrange(N_SKILL)   # 仅作记录，不必与序列对齐（Oracle 不依赖 topic）
        nxt = [-1] + corrects
        abil = [-1] + [1] * (T - 1)  # collate 占位，不参与 next 主评测
        rows.append({
            "topic": s0,
            "corrects": corrects,
            "diffs": diffs,
            "next": nxt,
            "abil": abil,
            "pOk": pOk,
        })

    with open(SAVE, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # row-split：固定随机 15% 为 test（与 p0_prereq 主管线一致）
    inds = list(range(n_rows))
    py.shuffle(inds)
    n_test = int(n_rows * test_frac)
    test_idx = sorted(inds[:n_test])
    train_idx = sorted(inds[n_test:])
    with open(SPLIT, "w", encoding="utf-8") as f:
        json.dump({"split_mode": "row-split", "seed": seed,
                   "test_index": test_idx, "train_index": train_idx}, f)

    print(f"[generate] {n_rows} rows (var len 8-16), 15-skill 3x5 lattice -> {os.path.abspath(SAVE)}")
    print(f"[split  ] row-split 15% -> {os.path.abspath(SPLIT)}  (train {len(train_idx)} / test {len(test_idx)})")

    # ---- Oracle Ⅰ：与 DKT 标签逐位一致 ----
    def auc(p, y):
        m = [yy >= 0 for yy in y]
        p = [x for x, mm in zip(p, m) if mm]
        y = [x for x, mm in zip(y, m) if mm]
        pos = [x for x, yy in zip(p, y) if yy == 1]
        neg = [x for x, yy in zip(p, y) if yy == 0]
        if not pos or not neg:
            return None, len(y)
        ranked = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg], key=lambda t: t[0])
        rp = sum(i + 1 for i, (_, cl) in enumerate(ranked) if cl == 1)
        np_, nn_ = len(pos), len(neg)
        a = rp / (np_ * nn_) - (np_ + 1) / (2 * nn_)
        return max(0.0, min(1.0, a)), len(y)

    test_rows = [rows[i] for i in test_idx]

    # 上帝视角：pOk[t+1] -> correct[t+1]
    bp, by = [], []
    for r in test_rows:
        for t in range(len(r["corrects"]) - 1):
            bp.append(r["pOk"][t + 1]); by.append(r["corrects"][t + 1])
    oracle, n = auc(bp, by)
    # 严格因果：pOk[t] -> correct[t+1]
    cp, cy = [], []
    for r in test_rows:
        for t in range(len(r["corrects"]) - 1):
            cp.append(r["pOk"][t]); cy.append(r["corrects"][t + 1])
    causal, n2 = auc(cp, cy)

    print(f"[Ⅰ.Oracle] god-view pOk[t+1]->correct[t+1]  auc={oracle:.4f}  n={n}")
    print(f"[Ⅰ.Causal] strict   pOk[t]  ->correct[t+1]  auc={causal:.4f}  n={n2}")

    B = 800
    vals = []
    for _ in range(B):
        s = rng.integers(0, n, n)
        v, _ = auc([bp[i] for i in s], [by[i] for i in s])
        vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    print(f"      god-view Oracle Bootstrap 95%CI=[{lo:.4f},{hi:.4f}] mean={np.mean(vals):.4f} n={n}")

    # 信息量诊断：pOk 是否有连续分布（非退化 0/1）
    allp = [x for r in test_rows for x in r["pOk"]]
    print(f"      pOk range=[{min(allp):.3f},{max(allp):.3f}]  std={np.std(allp):.3f}")


if __name__ == "__main__":
    main()