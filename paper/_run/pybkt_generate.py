#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pyBKT 机制合成数据生成（BKT-HMM 前向递推）+ 完整三层分解
================================================================
需求：找一个能透传逐位真概率 pOk 的第三方生成器，做独立于本项目
自制生成器（gen_synthetic，先难后易单调 mastery）的完整三层分解，
回应"只在自制生成器上验证"的外部效度质疑。

结论：pyBKT（Berkeley，CAHLR）提供标准 BKT-HMM 生成机制，但其高层
synthetic_data 的 installed 版在 prior 多维上有维度 bug，且不直接
输出逐位连续 pOk。故本脚本用 numpy 忠实复刻其 HMM 前向递推数学
（等价 pyBKT create_synthetic_data 的 loop 逻辑），在生成每个
作答的同时记录连续后验掌握度 L_{t}，并据此重建逐位真概率
  pOk[t] = (1 - slip) * L_t + guess * (1 - L_t)
从而三种生成口径（本步真概率 / slip·guess 隐状态 / 时序演化）全部可透传。

生成结构（与 p0_prereq / oracle_aligned_test 对齐）：
  每行 = 一个学生在单技能上的连续作答序列（topic 为 skill id 标量）。
  topic, corrects, diffs(=round(难度)), next, pOk 字段与现有 jsonl 兼容。

本脚本一次完成：
  (1) 生成 train/test 数据 + 固定 split
  (2) Oracle(Ⅰ)：next 对齐口径下 pOk[t] vs correct[t+1] 的 AUC
      （与 oracle_aligned_test 口径 B 完全一致：每行 t∈[0,L-2]）
用法：python paper/_run/pybkt_generate.py
"""
import os
import json
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "training", "data")

# ---------------- BKT HMM 前向递推生成 ----------------
def gen_bkt_skill(prior, learn, forget, guess, slip, diff, T, rng):
    """单技能单学生序列。返回 (corrects, diffs, pOk)。

    L_t = P(mastered at t)。给定作答观察后做贝叶斯更新，再接学习/遗忘转移。
    pOk[t] = P(correct_t) = (1-slip)*L_{t} + guess*(1-L_{t})。
    """
    corrects, diffs, pOk = [], [], []
    L = prior
    for t in range(T):
        p_ok = (1 - slip) * L + guess * (1 - L)
        ok = 1 if rng.random() < p_ok else 0
        # 后置更新（观察到 ok 后）→ 掌握后验
        if ok == 1:
            num = L * (1 - slip)
            denom = num + (1 - L) * guess
        else:
            num = L * slip
            denom = num + (1 - L) * (1 - guess)
        L_post = num / max(denom, 1e-9)
        # 学习/遗忘转移
        L = L_post * (1 - forget) + (1 - L_post) * learn
        L = min(0.999, max(0.001, L))
        corrects.append(ok)
        diffs.append(diff)
        pOk.append(float(round(p_ok, 4)))
    return corrects, diffs, pOk


def main():
    # 参数化：python pybkt_generate.py [seed] [n_skills] [n_rows] [out_name]
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_skills = int(sys.argv[2]) if len(sys.argv) > 2 else 192
    n_rows = int(sys.argv[3]) if len(sys.argv) > 3 else 3840
    out = sys.argv[4] if len(sys.argv) > 4 else "pybkt_prereq"
    SAVE = os.path.join(DATA, f"{out}.jsonl")
    SPLIT = os.path.join(DATA, f"{out}.split.json")
    rng = np.random.default_rng(seed)
    py = random.Random(seed)

    test_skill_gap = max(4, num_skills // 4)   # 尾部 25% 技能仅作 test（skill 级留出）

    rows = []
    for _ in range(n_rows):
        skill = py.randrange(1, num_skills + 1)   # topic ∈ [1,n_skills]，避开 padding_idx=0
        # BKT 参数（每个技能随机锚点：先验掌握、学习、遗忘、猜测、失误）
        prior = float(rng.uniform(0.05, 0.45))
        learn = float(rng.uniform(0.01, 0.08))   # 学习率调低→序列仍有起伏，难被"上一位"完全复制
        forget = float(rng.uniform(0.0, 0.08))
        guess = float(rng.uniform(0.18, 0.35))   # 噪声档拉高→本步更有不可测成分
        slip = float(rng.uniform(0.12, 0.25))
        diff = py.randint(1, 5)
        # 每条序列长度随机 8~16，体现学习曲线 + 防固定步长记忆
        T = py.randint(8, 16)
        corrects, diffs, pOk = gen_bkt_skill(
            prior, learn, forget, guess, slip, diff, T, rng)
        # next 字段（与主管线 gen_synthetic 同语义：next[i]=correct[i]，长度=corrects）
        # 消费方1 collate:  tgt_next[b,t]=next[t+1]=corrects[t+1]  （正确的 next-correct 标签）
        # 消费方2 line995:  next[1:]=corrects[1:] 长度=T-1，对齐 bkt_next_sequence(corrects) 的 T-1 个预测
        # 切勿写成 [-1]+corrects：① collate 会取到 tgt_next[b,t]=corrects[t]（当前已见输入→标签泄漏→AUC≈1）
        #                         ② next 长度 T+1≠T，line995 探针长度不匹配而崩溃
        nxt = list(corrects)
        abil = [-1] + [1] * (T - 1)   # collate 需 abil[i]; 仅作占位不参与 next 主评测
        rows.append({
            "topic": skill,
            "corrects": corrects,
            "diffs": diffs,
            "next": nxt,
            "abil": abil,
            "pOk": pOk,
        })

    with open(SAVE, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- skill 级留出：后 test_skill_gap 个 skill 只出现在 test（防同 skill 记忆）----
    # 每条行仅单个 skill；按 skill 分流，test 用 [num_skills-test_skill_gap, num_skills)，其余 train
    test_skills = set(range(num_skills - test_skill_gap, num_skills))
    train_idx = [i for i, r in enumerate(rows) if r["topic"] not in test_skills]
    test_idx = [i for i, r in enumerate(rows) if r["topic"] in test_skills]
    with open(SPLIT, "w", encoding="utf-8") as f:
        json.dump({"split_mode": "skill-split", "test_index": test_idx,
                   "train_index": train_idx, "test_topics": list(sorted(test_skills))}, f)

    print(f"generated {n_rows} rows (var len 8-16) -> {os.path.abspath(SAVE)}")
    print(f"split -> {os.path.abspath(SPLIT)}  (train {len(train_idx)} / test {len(test_idx)}, "
          f"test skills {len(test_skills)} disjoint from train)")

    # ---- (2) Oracle Ⅰ：与 DKT 标签逐位一致 —— pOk[t+1] vs correct[t+1] ----
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
    bp, by = [], []
    for r in test_rows:
        L = len(r["corrects"])
        for t in range(L - 1):   # 与 DKT eval mask 一致：t<=L-2, 标签=correct[t+1]
            bp.append(r["pOk"][t + 1])
            by.append(r["corrects"][t + 1])
    oracle, n = auc(bp, by)
    print(f"[Ⅰ Oracle] pyBKT-BKT 真概率 (predict correct[t+1])  auc={oracle:.4f}  n={n}")
    print("  对照(本项目自制合成同口径)：Oracle=0.759 / DKT(Ⅲ)=0.717 / gap①≈0.042")

    # Bootstrap 95% CI（n 与 DKT eval 同批，可比）
    B = 800
    vals = []
    N = len(by)
    for _ in range(B):
        s = rng.integers(0, N, N)
        v, _ = auc([bp[i] for i in s], [by[i] for i in s])
        vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    print(f"      Oracle Bootstrap 95%CI=[{lo:.4f},{hi:.4f}] mean={np.mean(vals):.4f} n={N}")


if __name__ == "__main__":
    main()