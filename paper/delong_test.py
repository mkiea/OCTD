# -*- coding: utf-8 -*-
"""
配对 DeLong 检验：Oracle 上界（名义上界，pOk[t+1]） vs DKT 主结果，在 DKT 评测
所用同一批 n=2880 个"下一题"测试样本上做配对显著性检验。

依据：
  - p0_prereq.jsonl 中 pOk 与 corrects 同下标对齐（dktDataBuilder.js）。
  - DKT collate 的 tgt_next[b,t]=corrects[t+1]，tgt_pok[b,t]=pOk[t+1]
    （train_dkt.py L155/L158），故同一样本的 oracle 配对分数= pOk[t+1]。
  - DeLong 1958（聚类相关两 AUC 之比较）：Z=(A1-A2)/sqrt(V10+V01-2C)。

运行前提：训练产生 paper/_run/pt_s1.pt。
用法： python paper/delong_test.py
"""
import json
import os
import sys
from math import erf, sqrt

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, ROOT)

import train_dkt as T

TT = 16


def phi(a, b):
    """DeLong 核：> 记 1，= 记 0.5，< 记 0（处理并列）。"""
    return 1.0 if a > b else (0.5 if a == b else 0.0)


def delong(A1, A2, f1, g1, f2, g2):
    """两相关 AUC 的 DeLong 检验。
    f_i = E_{neg}[phi(s_i, s_j)]（正样本侧），g_j = E_{pos}[phi(s_i, s_j)]（负样本侧）。
    返回 (A1, A2, diff, var_diff, z, p_two, p_one)。
    """
    n1 = len(f1)   # 正样本数
    n0 = len(g1)   # 负样本数
    mf1 = float(np.mean(f1)); mg1 = float(np.mean(g1))
    mf2 = float(np.mean(f2)); mg2 = float(np.mean(g2))
    V01 = float(np.sum((f1 - mf1) ** 2)) / (n1 - 1)   # A1 正侧方差(样本方差)
    V10 = float(np.sum((g1 - mg1) ** 2)) / (n0 - 1)   # A1 负侧方差
    V01b = float(np.sum((f2 - mf2) ** 2)) / (n1 - 1)  # A2 正侧方差
    V10b = float(np.sum((g2 - mg2) ** 2)) / (n0 - 1)  # A2 负侧方差
    COVf = float(np.sum((f1 - mf1) * (f2 - mf2))) / (n1 - 1)  # 正侧协方差
    COVg = float(np.sum((g1 - mg1) * (g2 - mg2))) / (n0 - 1)  # 负侧协方差
    var1 = V01 / n1 + V10 / n0
    var2 = V01b / n1 + V10b / n0
    cov = COVf / n1 + COVg / n0
    var_diff = var1 + var2 - 2 * cov
    se = sqrt(var_diff) if var_diff > 0 else 0.0
    z = (A1 - A2) / se if se > 0 else float("nan")
    p_one = 0.5 * (1 - erf(z / sqrt(2))) if not np.isnan(z) else float("nan")   # P(Z>z)：上界高于模型
    p_two = 2 * (0.5 * (1 - erf(abs(z) / sqrt(2)))) if not np.isnan(z) else float("nan")
    return A1, A2, A1 - A2, var_diff, z, p_two, p_one


def main():
    data_path = os.path.join(ROOT, "training", "data", "p0_prereq.jsonl")
    split_path = os.path.join(ROOT, "training", "data", "p0_prereq.split.json")
    rows = T.load_jsonl(data_path)
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)
    test_rows = [rows[i] for i in split["test_index"]]

    cp = os.path.join(ROOT, "paper", "_run", "pt_s1.pt")
    model = T.HybridKT(num_topics=192, max_len=TT, hidden=32, embed_dim=16)
    model.load_state_dict(torch.load(cp, map_location="cpu"), strict=False)
    model.eval()
    topic, correct, diff, lat, pre, mask, tgt_next, abil, mast, _w, tgt_pok, _reg = \
        T.collate(test_rows, model.num_topics, TT, torch.device("cpu"))
    with torch.no_grad():
        pn, _ms, _abil, _cal = model(topic, correct, diff)

    y, o, m = [], [], []
    for b in range(len(test_rows)):
        for t in range(TT):
            if mask[b, t].item() and int(tgt_next[b, t]) >= 0:
                oi = tgt_pok[b, t].item()
                if not np.isnan(oi) and 0.0 <= oi <= 1.0:
                    y.append(int(tgt_next[b, t]))
                    o.append(oi)
                    m.append(pn[b, t].item())
    y = np.asarray(y); o = np.asarray(o); m = np.asarray(m)
    n = len(y)
    pos = y == 1
    n1 = int(pos.sum()); n0 = n - n1
    print(f"n={n}  pos={n1}  neg={n0}")

    def auc_score(s, ylab):
        """秩和法 AUC（Mann–Whitney U / Wilcoxon），O(n log n) 替代 O(n²) 双循环。
        与 phi(a,b): >记1、=记0.5 的 AOC 语义一致：并列取平均秩。ylab 与 s 等长且逐位对齐。"""
        s = np.asarray(s); ylab = np.asarray(ylab)
        k = len(ylab)
        ppos = ylab == 1
        k1 = int(ppos.sum()); k0 = k - k1
        if k1 == 0 or k0 == 0:
            return None
        order = np.argsort(s, kind="mergesort")
        rank1 = np.empty(k)          # 1-based 平均秩（并列取平均）
        i = 0
        while i < k:
            j = i
            while j + 1 < k and s[order[j + 1]] == s[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            rank1[order[i:j + 1]] = avg
            i = j + 1
        r1 = float(rank1[ppos].sum())
        return (r1 - k1 * (k1 + 1) / 2.0) / (k1 * k0)

    A_o = auc_score(o, y)
    A_m = auc_score(m, y)
    # 逐样本 f（正侧）、g（负侧）※向量化
    def fg(s):
        s = np.asarray(s)
        f = np.empty(n1); g = np.empty(n0)
        sp = s[pos]; sn = s[~pos]
        for k, si in enumerate(sp):
            f[k] = float(np.mean(si > sn) + 0.5 * np.mean(si == sn))
        for k, sj in enumerate(sn):
            g[k] = float(np.mean(sp > sj) + 0.5 * np.mean(sp == sj))
        return f, g

    f1, g1 = fg(o)
    f2, g2 = fg(m)
    A1, A2, diff, vd, z, p2, p1 = delong(A_o, A_m, f1, g1, f2, g2)
    print("\n=== DeLong 配对检验（同一 n 测试样本） ===")
    print(f"Oracle(名义上界) AUC = {A1:.4f}")
    print(f"DKT(c0, seed1) AUC  = {A2:.4f}")
    print(f"gap = {diff:+.4f}   Var(diff) = {vd:.6f}")
    print(f"Z = {z:+.3f}   p_two_sided = {p2:.3e}   p_one_sided(A1>A2) = {p1:.3e}")

    # 配对 Bootstrap 差异 CI（稳健对照；重采样须同时重排分数与标签）
    rng = np.random.default_rng(42)
    B = 1000
    idx = np.arange(n)
    dvals = np.empty(B)
    for k in range(B):
        s = rng.choice(idx, n, replace=True)
        dvals[k] = auc_score(o[s], y[s]) - auc_score(m[s], y[s])
    lo, hi = np.percentile(dvals, [2.5, 97.5])
    print(f"配对 Bootstrap Δ 95%CI=[{lo:.4f},{hi:.4f}]  （不含 0 ⇒ gap 显著）")

    concl = ("显著：名义上界显著高于 DKT ⇒ 特征/结构侧存在可用提升空间"
             if p2 < 0.05 and np.signbit(lo) == np.signbit(hi) and lo > 0
             else "不显著：AUC 差异落入波动带，gap① 不能由本轮判据证伪")
    print("\n判定:", concl)


if __name__ == "__main__":
    main()