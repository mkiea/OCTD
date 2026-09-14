#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ktbd(DKT synthetic) 静态 IRT 拟合 -> 对齐 Oracle(Ⅰ) 读数
================================================================
ktbd synthetic 只含 [[id, correct]] 二值序列，无 pOk。方案A：假设其为静态
IRT(3PL) 生成，用二值作答拟合条目参数(a,b,c)，对 test 学生做 leave-one-out
能力估计得到 θ，据此闭式计算逐位 pOk，并按 DKT eval 的 next 对齐口径算 Oracle AUC。

对齐口径（与 train_dkt.py eval 完全一致）：
  DKT 在 test 每行 t∈[0,L-2] 评分，target = tgt_next[b,t] = next[t+1] = corrects[t+2]，
  样本数 n = 1999 × 48 = 95952。topic 固定为 [0..49]，故第 t 位预测的是 item (t+2)。
  静态 IRT 下 P(corrects[t+2]) = IRT(θ_s, item_{t+2})，与学生前缀历史无关。

Liftoff 判定：Oracle(IRT静态上界) 必须 ≥ DKT(c0)=0.824 才是合法 ceiling；
若 < 0.824 则说明生成器含更深的隐藏时序结构，静态 IRT 低估真上界（详见打印警告）。
用法：python paper/_run/ktbd_irt_oracle.py
"""
import json
import numpy as np

TRAIN = r"training/data/synthetic/synthetic_train.jsonl"
TEST = r"training/data/synthetic/synthetic_test.jsonl"
NUM_ITEMS = 50

# ---------------- 3PL IRT ----------------
def irt_p(theta, a, b, c):
    """P(correct) = c + (1-c) * sigmoid(a*(theta - b))"""
    return c + (1 - c) / (1.0 + np.exp(-a * (theta - b)))

# ---------------- load rows ----------------
def load_rows(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        rows.append({"topic": o["topic"], "corrects": o["corrects"]})
    return rows

def row_to_matrix(rows, num_items=NUM_ITEMS):
    R = np.zeros((len(rows), num_items))
    for i, r in enumerate(rows):
        for q, c in zip(r["topic"], r["corrects"]):
            R[i, q] = c
    return R

# ---------------- joint MLE on train ----------------
def fit_items_train(R, iters=3000, lr=0.05, lam=0.01):
    """联合最大似然拟合 train 的能力θ与条目参数(a,b,c), 返回条目参数。"""
    rng = np.random.default_rng(0)
    ns, nq = R.shape
    theta = rng.normal(0, 1, size=ns)
    a = np.full(nq, 1.5)
    b = rng.normal(0, 1, size=nq)
    logitc = np.full(nq, -1.4)  # -> c≈0.2

    def params():
        return theta, a, b, logitc

    def loss(params, R):
        th, a, b, lc = params
        c = 1 / (1 + np.exp(-lc))
        p = irt_p(th[:, None], a[None, :], b[None, :], c[None, :])
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.nansum(R * np.log(p) + (1 - R) * np.log(1 - p)), p

    # 简单随机梯度：逐个样本交替更新太慢, 用全量梯度+Adam
    m = {k: np.zeros_like(v) for k, v in dict(theta=theta, a=a, b=b, lc=logitc).items()}
    v = {k: np.zeros_like(kk) for k, kk in m.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    names = ["theta", "a", "b", "lc"]
    for it in range(iters):
        th, a_, b_, lc = theta, a, b, logitc
        c = 1 / (1 + np.exp(-lc))
        z = a_[None, :] * (th[:, None] - b_[None, :])          # [ns,nq]
        sig = 1 / (1 + np.exp(-z))
        p = c[None, :] + (1 - c[None, :]) * sig
        p = np.clip(p, 1e-6, 1 - 1e-6)
        denom = p * (1 - p)
        w = (p - R) / denom * (1 - c[None, :]) * sig * (1 - sig)  # h 因子
        g_theta = (w * a_[None, :]).sum(1)
        g_a = (w * (th[:, None] - b_[None, :])).sum(0)
        g_b = (-w * a_[None, :]).sum(0)
        g_c = ((p - R) / denom * (1 - sig)).sum(0)
        dc_dlc = c * (1 - c)
        g_lc = g_c * dc_dlc
        g_theta += lam * th
        g_a += lam * a_
        g_b += lam * b_

        grads = {"theta": g_theta, "a": g_a, "b": g_b, "lc": g_lc}
        for k in names:
            gg = np.clip(grads[k], -5, 5)
            m[k] = b1 * m[k] + (1 - b1) * gg
            v[k] = b2 * v[k] + (1 - b2) * gg * gg
            mh = m[k] / (1 - b1 ** (it + 1))
            vh = v[k] / (1 - b2 ** (it + 1))
            step = lr * mh / (np.sqrt(vh) + eps)
            if k == "theta":
                theta -= step
            elif k == "a":
                a = np.clip(a - step, 0.05, 5)
            elif k == "b":
                b -= step
            else:
                logitc = np.clip(logitc - step, -6, 0)
        if (it + 1) % 500 == 0:
            lossval, _ = loss(params(), R)
            print(f"[fit] iter {it+1} neg-ll={lossval:.1f}")
    c = 1 / (1 + np.exp(-logitc))
    return a, b, c

# ---------------- leave-one-out ability MLE ----------------
def ability_loo(a, b, c, Rtest, iters=6, lr=0.6):
    """对每个 test 学生做 leave-one-out 能力估计，返回 [ns, nq] pOk 矩阵。
    对给定学生剔除目标题 q 时，用其余 49 题响应估 θ^(−q)，再算 item q 的 pOk（防自预测泄漏）。"""
    ns, nq = Rtest.shape
    pok = np.zeros((ns, nq))
    base_theta = np.zeros(ns)
    # 全序列 θ 作粗起点
    for s in range(ns):
        obs = np.ones(nq, dtype=bool)
        th = 0.0
        for _ in range(iters):
            p = irt_p(th, a[obs], b[obs], c[obs])
            p = np.clip(p, 1e-6, 1 - 1e-6)
            g = np.sum((Rtest[s][obs] - p) * a[obs]) - 0.05 * th
            th = np.clip(th + lr * g, -6, 6)
        base_theta[s] = th
    # leave-one-out：对每 (s,q) 剔 q 估 θ，算 item q pOk
    rows_obs = np.ones((ns, nq), dtype=bool)
    for q in range(nq):
        excl = rows_obs.copy()
        excl[:, q] = False
        for s in range(ns):
            obs = excl[s]
            th = base_theta[s]
            for _ in range(iters):
                p = irt_p(th, a[obs], b[obs], c[obs])
                p = np.clip(p, 1e-6, 1 - 1e-6)
                g = np.sum((Rtest[s][obs] - p) * a[obs]) - 0.05 * th
                th = np.clip(th + lr * g, -6, 6)
            pok[s, q] = irt_p(th, a[q], b[q], c[q])
    return pok

def auc_from_rank(p, y):
    m = y >= 0
    p_, y_ = p[m], y[m]
    pos = p_[y_ == 1]
    neg = p_[y_ == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None, 0
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    rp = ranks[: len(pos)].sum()
    auc = rp / (len(pos) * len(neg)) - (len(pos) + 1) / (2 * len(neg))
    return float(np.clip(auc, 0, 1)), int(len(y_))

def main():
    tr = load_rows(TRAIN)
    te = load_rows(TEST)
    print(f"train rows={len(tr)} test rows={len(te)}")
    Rtr = row_to_matrix(tr)
    Rte = row_to_matrix(te)
    a, b, c = fit_items_train(Rtr)
    print("item a[0..4]=", a[:5].round(3), " b[0..4]=", b[:5].round(3), " cmean=", c.mean().round(3))
    # LOO 能力 -> pOk per (student,item)
    pok = ability_loo(a, b, c, Rte)
    # 对齐 DKT eval：每行 t=0..47, target=corrects[t+2], score=pOk[item=t+2]
    p_all, y_all = [], []
    for s, r in enumerate(te):
        L = len(r["corrects"])
        for t in range(L - 1):  # t = 0..47 (n=48 per row), 预测 item (t+2)
            if t + 2 <= L - 1:
                p_all.append(pok[s, r["topic"][t + 2]])
                y_all.append(r["corrects"][t + 2])
    auc, n = auc_from_rank(np.asarray(p_all), np.asarray(y_all))
    print("=" * 60)
    print(f"Oracle Ⅰ (静态IRT LOO, next对齐)  auc={auc:.4f}  samples={n}")
    print(f"对照: DKT(c0 ktbd)=0.824 ; oracle与0.824之差 = {auc-0.824:+.4f}")
    if auc < 0.824:
        print("警告: Oracle < DKT ⇒ 生成器隐藏结构更强, 静态IRT非真上界(低估)。POK路径对ktbd不成立。")
    else:
        print("结论: 静态IRT为kbd合法ceiling, 完整三层可行。")

if __name__ == "__main__":
    main()