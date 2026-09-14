#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KS-discovery-for-ITS StudentModel 移植 —— 可行性验证
================================================================
目的：评估"ELO + 前置条件依赖 + 短/长程学习 + 遗忘"机制的第三方生成器
是否能把逐位真概率 pOk 透传进本项目三层分解管线（回应"只在自制生成器/pyBKT
上验证"的质控缺口）。

移植来源：sino7/KS-discovery-for-ITS 的 student_models/student_model.py
          过程动力学原文照搬（success_prediction / student_learning），
          仅把"同步多学生×多技能"放宽为"每学生固定单技能学习曲线"，
          以适配本项目每行=单技能序列的 jsonl 格式。

生成概率口径（忠实 README 演示）：pOk_t = 0.2 + 0.7 * proba_elo_t
  其中 proba_elo_t 由 ELO sigmoid + 前置条件 soft-min + 长短程掌握度求得；
  采样 success = Bern(pOk_t)。因此 pOk 即本步真实生成概率，可直接做 Oracle Ⅰ。

输出：training/data/ksgen_prereq.jsonl (+ .split.json)，并打印 Oracle Ⅰ。
用法：python paper/_run/ksgen_feasibility.py
"""
import os
import json
import random
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "training", "data")
LRNG_SCALE = 0.10   # 学习步长缩放：压制默认 ELO 过度更新→pOk 均值回归/倒挂，使逐位 pOk 良好校准

# ---------------- 移植自 utils.py（原样，仅内联） ----------------
def elo_proba(elo_a, elo_b):
    return torch.sigmoid(np.log(10.0) * (elo_a - elo_b) / 400.0)


def masked_softmin_elo(elos, mask):
    """soft-min over masked KCs：越弱的前置技能，下拉完成任务概率越强。"""
    batch_size, n_kc = elos.shape
    weights = mask * torch.exp(-np.log(10.0) * elos / 400.0)
    ws = weights / (1e-100 + torch.sum(weights, dim=-1).unsqueeze(-1))
    return torch.sum(ws * elos, dim=1)


# ---------------- 移植自 student_model.py（单学生版） ----------------
class SingleStudent:
    """单学生的全动力学：长时 kc_elo（lt）+ 短时 kc_elo（st）+ 前置条件软化 + 遗忘。"""

    def __init__(self, n_kc, ex_elo, kc_graph, rng):
        self.n_kc = n_kc
        self.ex_elo = ex_elo                       # (n_ex,) 每题 ELO
        self.kc_graph = kc_graph                   # (n_kc, n_kc) 前置条件邻接
        self.base_elo = 1000.0 + 200.0 * float(rng.normal(0, 1))
        self.init_kc = self.base_elo + 200.0 * rng.normal(0, 1, size=n_kc)
        self.log_lrng_speed = 3.0 + 0.3 * float(rng.normal(0, 1))
        self.log_success_reward = 4.0
        self.log_failure_reward = 2.0
        self.log_tau = 1.0
        self.epsilon = 0.5
        self.reset()

    def reset(self):
        self.st_kc = torch.tensor(self.init_kc.copy(), dtype=torch.float)
        self.lt_kc = torch.tensor(self.init_kc.copy(), dtype=torch.float)

    def _combined_kc(self):
        # 长短时 softmax 加权（原文 softmax over [lt, st]）
        stack = torch.stack([self.lt_kc, self.st_kc], dim=0)          # (2, n_kc)
        w = torch.softmax(stack, dim=0)
        return torch.sum(w * stack, dim=0)                            # (n_kc,)

    def success_prediction(self, ex):
        # 前置条件掩码：该题直接触达的 KC + 其所有祖先（二值化，soft-min 需 0/1 权重）
        direct = self.ex2kc[ex].float()
        contrib = direct + torch.mm(direct.unsqueeze(0), self.ancestor.T)  # k 直达 + k 的祖先
        mask = (contrib > 0).float()
        kc_elo = self._combined_kc().unsqueeze(0)                     # (1, n_kc)
        st = masked_softmin_elo(kc_elo, mask)                          # (1,)
        proba = elo_proba(st, torch.tensor([self.ex_elo[ex]], dtype=torch.float))
        return proba.item()

    def student_learning(self, ex, success):
        # 学习速度：成功 > 失败，×前置条件达成度 ×进度窗
        lrng = (torch.exp(torch.tensor(self.log_failure_reward)) +
                success * (torch.exp(torch.tensor(self.log_success_reward)) -
                           torch.exp(torch.tensor(self.log_failure_reward)))) * \
            torch.exp(torch.tensor(self.log_lrng_speed))
        parents = self.ex2kc[ex].float().unsqueeze(0)
        pr = masked_softmin_elo(self.lt_kc.unsqueeze(0), (parents > 0).float())
        pr = torch.nan_to_num(pr, nan=1.0)
        lrng = lrng * pr
        sp = self.success_prediction(ex)
        progress_area = 4.0 * sp * (1.0 - sp)
        kc_delta = (lrng * progress_area) * LRNG_SCALE * self.ex2kc[ex].float()
        # 长时
        self.lt_kc = self.lt_kc + self.epsilon * kc_delta * torch.exp((self.lt_kc - self.st_kc) / 100.0)
        self.lt_kc = torch.clamp(self.lt_kc, 500.0, 2500.0)
        # 短时 + 遗忘
        self.st_kc = self.st_kc + kc_delta
        self.st_kc = self.lt_kc + (self.st_kc - self.lt_kc) * torch.exp(-1.0 / torch.exp(torch.tensor(self.log_tau)))
        self.st_kc = torch.clamp(self.st_kc, 500.0, 2500.0)


def diff_from_elo(ex_elo):
    """把每题的 ELO 难度映射到 1-5（全局等宽分档），保证 diffs 字段可用。"""
    lo, hi = float(min(ex_elo)), float(max(ex_elo))
    span = max(hi - lo, 1e-6)
    return [int(max(1, min(5, round(1 + (float(e) - lo) / span * 4.0)))) for e in ex_elo]


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
    # 参数化：python ksgen_feasibility.py [seed] [n_kc] [n_rows] [out_name]
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n_kc = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    n_rows = int(sys.argv[3]) if len(sys.argv) > 3 else 3840
    out = sys.argv[4] if len(sys.argv) > 4 else "ksgen_prereq"
    SAVE = os.path.join(DATA, f"{out}.jsonl")
    SPLIT = os.path.join(DATA, f"{out}.split.json")
    rng = np.random.default_rng(seed)
    py = random.Random(seed)
    torch.manual_seed(seed)

    test_skill_gap = max(2, n_kc // 4)       # 尾部 25% KC 仅作 test（skill 级留出）

    # 前置条件图：Erdos-Renyi 期望度 2 → 祖先距离矩阵（anc[i,j]=i 到 j 的最短步数，0 表示不可达）
    A = np.zeros((n_kc, n_kc))
    for i in range(n_kc):
        for j in range(n_kc):
            if i != j and rng.random() < 2.0 / (n_kc - 1):
                A[i, j] = 1.0
    M = torch.tensor(A, dtype=torch.float)
    anc = torch.zeros(n_kc, n_kc)
    P = torch.eye(n_kc)
    for i in range(1, n_kc + 1):
        P = torch.mm(P, M)
        anc = torch.max(anc, (i * (P > 0)).float())   # 保留到达最近祖先的"步数×可达掩码"

    n_ex = n_kc
    ex_elo = 1500.0 + 300.0 * torch.randn(n_ex)
    EX2KC = (torch.eye(n_kc) + (anc > 0).float().T).clamp(max=1.0)   # EX2KC[j]=KC j + 其前置(祖先)，练 j 同时推进前置
    D = diff_from_elo(ex_elo.tolist())

    rows = []
    for _ in range(n_rows):
        # 每学生固定练一道题（单技能学习曲线）
        ex = py.randrange(n_ex)
        topic = ex + 1                # topic 偏移到 [1,n_kc]，避免 padding_idx=0 冻结主题嵌入
        stu = SingleStudent(n_kc, ex_elo.cpu().numpy(), anc, rng)
        stu.ex2kc = EX2KC
        stu.ancestor = anc
        T = py.randint(8, 16)
        corrects, pOk = [], []
        for t in range(T):
            proba = stu.success_prediction(ex)
            p_ok = 0.2 + 0.7 * proba            # README guess=0.2/slip 采样口径
            ok = 1 if rng.random() < p_ok else 0
            stu.student_learning(ex, float(ok))
            corrects.append(ok)
            pOk.append(float(round(min(0.999, max(0.001, p_ok)), 4)))
        # next 字段同主管线 gen_synthetic：next[i]=correct[i]（长度=corrects）。
        # collate tgt_next[b,t]=next[t+1]=corrects[t+1]（正确的 next-correct）；勿写 [-1]+corrects 否则泄漏/崩溃
        nxt = list(corrects)
        abil = [-1] + [1] * (T - 1)
        rows.append({"topic": topic, "corrects": corrects,
                     "diffs": [D[ex]] * T, "next": nxt, "abil": abil, "pOk": pOk})

    with open(SAVE, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- row 拆分（与主管线同口径：固定随机 85/15 test 集，Oracle 与 DKT 共用）----
    idx = list(range(len(rows)))
    py.shuffle(idx)
    sp = int(len(rows) * 0.85)
    train_idx, test_idx = idx[:sp], idx[sp:]
    with open(SPLIT, "w", encoding="utf-8") as f:
        json.dump({"split_mode": "row-split", "test_index": test_idx,
                   "train_index": train_idx, "test_topics": []}, f)

    # ---- Oracle Ⅰ：与 DKT 标签逐位一致 pOk[t+1] vs correct[t+1] ----
    bp, by = [], []
    for i in test_idx:
        r = rows[i]
        L = len(r["corrects"])
        for t in range(L - 1):
            bp.append(r["pOk"][t + 1]); by.append(r["corrects"][t + 1])
    oracle, n = auc(bp, by)
    B = 800
    vals = []
    for _ in range(B):
        s = rng.integers(0, n, n)
        v, _ = auc([bp[i] for i in s], [by[i] for i in s]); vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    print(f"[Ⅰ Oracle] KS-ELO 真概率 predict correct[t+1]  auc={oracle:.4f} n={n}")
    print(f"      Bootstrap 95%CI=[{lo:.4f},{hi:.4f}] mean={np.mean(vals):.4f}")
    print("  对照：本项目自制合成同口径 Oracle=0.759；pyBKT 口径见其脚本；随机=0.5")
    print(f"rows={n_rows} -> {os.path.abspath(SAVE)}")

    # ---- 可行性自洽诊断：pOk 是否真的随序列演化 / 逐位透传 ----
    def _auc(pp, yy):
        pos = [x for x, y in zip(pp, yy) if y == 1]; neg = [x for x, y in zip(pp, yy) if y == 0]
        r = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg], key=lambda t: t[0])
        rp = sum(i + 1 for i, (_, c) in enumerate(r) if c == 1)
        return rp / (len(pos) * len(neg)) - (len(pos) + 1) / (2 * len(neg))
    sp_p, sp_y = [], []
    rowr = []
    for rr in rows:
        rowr.append(max(rr["pOk"]) - min(rr["pOk"]))
        for t in range(len(rr["corrects"])):
            sp_p.append(rr["pOk"][t]); sp_y.append(rr["corrects"][t])
    same = _auc(sp_p, sp_y)
    flat = 100.0 * sum(1 for x in rowr if x < 0.05) / len(rowr)
    print(f"[诊断] 同一步 pOk[t]~correct[t] AUC={same:.4f}（>0.5 说明逐位真概率透传有效）")
    print(f"[诊断] 行内 pOk 极差 中位数={np.median(rowr):.4f}  平坦行(<0.05)占比={flat:.1f}%")


if __name__ == "__main__":
    main()