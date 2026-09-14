#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DINA/Q-matrix 认知诊断机制合成生成器 + Oracle(Ⅰ)
================================================================
机制：认知诊断（Cognitive Diagnosis），Q-matrix 决定每道题所需的最小技能集合，
学生具备"核对位掌握模式"（mastery pattern over required KCs），题目作答概率为
合取式（conjunctive）：
    mastered_k ∈ {0,1} 为学生对该知识点的精通隐变量；项 i 需要技能集 K_i。
    eta = prod_{k in K_i} mastered_k        （是否全部掌握，DINA 的 And-gate）
    pOk_i = guess + (1 - guess - slip) * eta
学生作答后按 DINA 后验更新各所需技能的掌握后验（Bayes），随正确/失误演化——
因此 pOk 是逐位变化的前向真概率，可直接透传作 Oracle Ⅰ。

与 pyBKT（单技能双参数 HMM）机制独立：DINA 的关键是非线性合取门
"任一所需技能未掌握即答对率塌回 guess"，而非 BKT 的线性 slip/guess 混合。
与 ktbd(IRT)/KS-ELO 亦异源。标量 topic + 单技能序列布局，与 p0/pyBKT/KS-ELO 共享，
纳入 matched 受控矩阵（同位 n_skills + 同样本量）。

each 行 = 单学生连续练习某技能（该技能的题目依赖一随机所需技能子集，含自身）。
next 字段语义与 pyBKT/KS-ELO 一致：next[i]=correct[i]（长度=corrects），
collate 处 tgt_next[b,t]=next[t+1]=corrects[t+1]（正确的 next-正确 标签）；
切勿写成 [-1]+corrects（会泄漏/index 崩溃）。

用法：python paper/_run/dina_generate.py [seed] [n_skills] [n_rows] [out_name]
"""
import os
import json
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "training", "data")


def gen_dina_skill(n_req, prior0, learn, forget, guess, slip, diff, T, rng):
    """单技能单学生 DINA-HMM 序列。返回 (corrects, diffs, pOk)。

    mastered_k = P(k 已掌握，全域技能共享的所需子集)。作答后按观察到对/错
    对所需各技能做 Bayes 更新；再接学习/遗忘转移。合取门在 pOk 中体现。
    """
    corrects, diffs, pOk = [], [], []
    m = np.full(n_req, prior0)          # 各所需技能的掌握后验
    for t in range(T):
        eta = float(np.prod(m))          # 合取 And-gate：全部所需技能已掌握
        p_ok = guess + (1 - guess - slip) * eta
        ok = 1 if rng.random() < p_ok else 0
        # 对所需各技能做 Bayes 掌握更新（忽略 "全掌握" 饱和后的极端收敛）
        for k in range(n_req):
            if ok == 1:
                num = m[k] * (1 - slip)
                denom = num + (1 - m[k]) * guess
            else:
                num = m[k] * slip
                denom = num + (1 - m[k]) * (1 - guess)
            m[k] = num / max(denom, 1e-9)
        # 学习/遗忘转移（对全部所需技能对称）
        m = np.clip(m * (1 - forget) + (1 - m) * learn, 0.001, 0.999)
        corrects.append(ok)
        diffs.append(diff)
        pOk.append(float(round(p_ok, 4)))
    return corrects, diffs, pOk


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


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n_skills = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    n_rows = int(sys.argv[3]) if len(sys.argv) > 3 else 3840
    out = sys.argv[4] if len(sys.argv) > 4 else "dina_prereq"
    SAVE = os.path.join(DATA, f"{out}.jsonl")
    SPLIT = os.path.join(DATA, f"{out}.split.json")

    rng = np.random.default_rng(seed)
    py = random.Random(seed)

    test_skill_gap = max(2, n_skills // 4)   # 后 25% 技能仅作 test（skill 级留出）

    rows = []
    for _ in range(n_rows):
        skill = py.randrange(1, n_skills + 1)          # topic ∈ [1,n_skills]
        tgt = rng.uniform(0.05, 0.45)                  # 自身先验掌握
        # 所需技能子集 = 自身 + 少量"前置"技能（认知诊断 Q-matrix 的合取需求）
        n_req = int(py.randint(1, 3))
        masked = np.where(np.arange(n_skills) != (skill - 1))[0]
        pre = list(rng.choice(masked, size=min(n_req - 1, len(masked)),
                              replace=False).tolist())
        n_req = len(pre) + 1                            # >=1（含自身）
        prior = float(tgt)                               # 自身
        learn = float(rng.uniform(0.01, 0.08))
        forget = float(rng.uniform(0.0, 0.08))
        guess = float(rng.uniform(0.18, 0.35))
        slip = float(rng.uniform(0.12, 0.25))
        diff = py.randint(1, 5)
        T = py.randint(8, 16)
        corrects, diffs, pOk = gen_dina_skill(
            n_req, prior, learn, forget, guess, slip, diff, T, rng)
        nxt = list(corrects)
        abil = [-1] + [1] * (T - 1)
        rows.append({"topic": skill, "corrects": corrects,
                     "diffs": diffs, "next": nxt, "abil": abil, "pOk": pOk})

    with open(SAVE, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- skill 级留出 split（同 pyBKT 口径，防同技能记忆）----
    test_skills = set(range(n_skills - test_skill_gap, n_skills))
    train_idx = [i for i, r in enumerate(rows) if r["topic"] not in test_skills]
    test_idx = [i for i, r in enumerate(rows) if r["topic"] in test_skills]
    with open(SPLIT, "w", encoding="utf-8") as f:
        json.dump({"split_mode": "skill-split", "test_index": test_idx,
                   "train_index": train_idx,
                   "test_topics": list(sorted(test_skills))}, f)

    # ---- Oracle Ⅰ：pOk[t+1] vs correct[t+1]，与 DKT 标签逐位一致 ----
    bp, by = [], []
    for i in test_idx:
        r = rows[i]
        L = len(r["corrects"])
        for t in range(L - 1):
            bp.append(r["pOk"][t + 1])
            by.append(r["corrects"][t + 1])
    oracle, n = auc(bp, by)
    B = 800
    vals = []
    for _ in range(B):
        s = rng.integers(0, n, n)
        v, _ = auc([bp[i] for i in s], [by[i] for i in s])
        vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    print(f"[Ⅰ Oracle] DINA(Q-matrix) 真概率  auc={oracle:.4f}  n={n}")
    print(f"      Bootstrap 95%CI=[{lo:.4f},{hi:.4f}] mean={np.mean(vals):.4f}")
    print(f"generated {n_rows} rows (n_skills={n_skills}) -> {os.path.abspath(SAVE)}")
    print(f"  test rows={len(test_idx)} / test skills={len(test_skills)} (disjoint)")


if __name__ == "__main__":
    main()