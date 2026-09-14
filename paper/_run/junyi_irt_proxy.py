#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实 Junyi 代理 Oracle（Ⅰ'）：MML-EM 2PL IRT

真实数据 pOk 不可观测 ⇒ 恒可观测样本时无法直接构造真 Oracle。本文用"难度感知
2PL IRT"作代理下界参照：item=(topic, 难度档 1-5)；在 TRAIN 行上以 MML-EM
（Gauss-Hermite 求积，θ~N(0,1)）拟合每 item 的 (a,b)，对 test 用户用其 TRAIN
响应做 EAP 能力估计（防泄漏），再对 test 每行 t=0..L-2 计算
pOk = 2PL(θ_u, a_{item}, b_{item})，其中 item=(topic_b, diffs[t+1])，
target=corrects[t+1] —— 与 train_dkt.py 的 DKT 评估样本完全同口径。

口径忠告：
- 静态 IRT 无时序记忆，必然低估真可提取上界 → 仅作"下界参照/难度特征条件概率"，
  **不是**真 Oracle。与容量扫描可提取上界（Ⅱ'=0.795）区分开。
- 若 DKT(c0,0.7455) ≥ 代理 Ⅰ'：说明模型已超过静态难度/能力参照，待补的是
  时序结构（容量），与容量扫描 c0→c4 升到 0.795 一致。
- 若 DKT(c0) < 代理 Ⅰ'：难度+能力尚有余量未吃满。

用法: python paper/_run/junyi_irt_proxy.py
"""
import json, numpy as np

import os
_BASE = r"c:/Users/SolimPurmiss/Desktop/know-map-ai-learning-alpha-v0.3.5/know-map-ai-learning-alpha-v0.3.5"
ALIGN = os.path.join(_BASE, "training/data/junyi/junyi_aligned_fa2.jsonl")
SPLIT = os.path.join(_BASE, "training/data/junyi/fa2.split.json")
IRT_A = os.path.join(_BASE, "training/data/junyi/irt_a.npy")
IRT_B = os.path.join(_BASE, "training/data/junyi/irt_b.npy")
IRT_KEYS = os.path.join(_BASE, "training/data/junyi/irt_items.npy")

# --- Gauss-Hermite 2PL ---
def gh_nodes(K=41):
    # Gauss–Hermite 节点/权（θ~N(0,1)，EAP/MML 用）
    x, w = np.polynomial.hermite.hermgauss(K)
    return x.astype(float), (w / np.sqrt(np.pi)).astype(float)

XG, WG = gh_nodes(41)

def irt_p(a, b, x):
    """2PL: P = sigmoid(a*(x-b))，x=θ"""
    return 1.0 / (1.0 + np.exp(-a * (x - b)))

# --- 数据装配 ---
def build(limit=None):
    """读对齐行. 返回:
    rows: list of {user,topic,corrects,diffs,next}
    test_pos: sorted list of row indices (30000)
    """
    split = json.load(open(SPLIT, encoding="utf-8"))
    test = set(split["test_index"])
    rows = []
    n = 0
    for line in open(ALIGN, encoding="utf-8"):
        o = json.loads(line)
        n += 1
        if limit and n > limit:
            break
        rows.append({"user": o["user"], "topic": o["topic"],
                     "corrects": o["corrects"], "diffs": o["diffs"],
                     "next": o["next"], "in_test": (n - 1) in test})
    return rows, sorted(test)

def item_key(topic, diff):
    return (int(topic), int(diff))

def main():
    rows, test = build()
    print(f"rows={len(rows)} test_rows={len(test)}", flush=True)

    # item 编号（train 观测到的）
    idx = {}
    for r in rows:
        for d in r["diffs"]:
            k = item_key(r["topic"], d)
            idx.setdefault(k, len(idx))
    M = len(idx)
    print(f"items=(topic,diff) count={M}", flush=True)

    # 预分组：每 user 收集其所有 (item,correct)；同时每 item 收集所有 (user,correct) 加速 M-step
    train_by_user = {}
    train_by_item = [[] for _ in range(M)]  # [ [(user, r), ...], ... ]
    test_users = set()
    for r in rows:
        if r["in_test"]:
            test_users.add(r["user"])
        else:
            lst = train_by_user.setdefault(r["user"], [])
            for t in range(min(len(r["corrects"]), len(r["diffs"]))):
                k = idx[item_key(r["topic"], r["diffs"][t])]
                cr = int(r["corrects"][t])
                lst.append((k, cr))
                train_by_item[k].append((r["user"], cr))
    users = list(train_by_user.keys())
    print(f"train_users={len(users)} test_users={len(test_users)} items={M}", flush=True)
    avg_obs = sum(len(x) for x in train_by_item) / M
    print(f"  avg observations per item: {avg_obs:.1f}", flush=True)

    # ---- 两段式 EM（稳健、可解释，近似 MML）----
    # 段A：初始能力 θ0 = 归一化 logit(平均正确率)
    K = len(XG)
    from scipy.optimize import minimize as _min
    a = np.ones(M)
    b = np.zeros(M)
    th = {}
    for u, lst in train_by_user.items():
        s = sum(r for _, r in lst)
        th[u] = np.log((s + 0.5) / (len(lst) - s + 0.5))

    # step B: EM 迭代（M-step: 每 item 已预分组；E-step: EAP 更新能力）
    for it in range(30):
        # M-step: 每 item，用当前 θ 解 2PL (a,b) 最大似然
        changed = 0
        for j in range(M):
            obs = train_by_item[j]
            if len(obs) < 2:
                if a[j] != 1.0 or b[j] != 0.0:
                    a[j] = 1.0; b[j] = 0.0; changed += 1
                continue
            thetas = np.array([th[u] for u, _ in obs])
            rs = np.array([r for _, r in obs])
            def neg(lp):
                aa, bb = lp
                p = irt_p(aa, bb, thetas); p = np.clip(p, 1e-7, 1 - 1e-7)
                return -np.sum(rs * np.log(p) + (1 - rs) * np.log(1 - p))
            res = _min(neg, np.array([a[j], b[j]]), method="BFGS",
                       options={"maxiter": 35})
            aa_new, bb_new = np.clip(res.x[0], 0.05, 5.0), res.x[1]
            if abs(a[j] - aa_new) > 0.02 or abs(b[j] - bb_new) > 0.05:
                changed += 1
            a[j], b[j] = aa_new, bb_new
        # E-step: 更新每 user 能力 θ（EAP，用 all their train items）
        th_new = {}
        for u, lst in train_by_user.items():
            nodes = XG; pmat = np.empty((len(lst), K))
            for li, (j, r) in enumerate(lst):
                pmat[li] = irt_p(a[j], b[j], nodes)
            pm = np.clip(pmat, 1e-7, 1 - 1e-7)
            ll = np.sum(np.where(np.array([r for _, r in lst])[:, None] == 1,
                                 np.log(pm), np.log(1 - pm)), axis=0)
            post = WG * np.exp(ll - ll.max()); post /= post.sum()
            th_new[u] = float((nodes * post).sum())
        th = th_new
        if (it + 1) % 5 == 0:
            print(f"  [EM] iter {it+1} changed {changed}/{M} items", flush=True)
        if changed == 0:
            print(f"  [EM] early exit at iter {it+1} (no change)", flush=True)
            break
    # 保存训练 item 参数（供测试 user EAP 复用）
    np.save(IRT_A, a)
    np.save(IRT_B, b)
    np.save(IRT_KEYS, np.array(list(idx.keys())))
    print(f"[EM] item 拟合完成 items={M}", flush=True)

    # ---- 测试 user 能力 EAP（仅 train 响应）----
    def eap_user(u):
        lst = train_by_user.get(u)
        if not lst:
            return 0.0
        pmat = np.empty((len(lst), K))
        for li, (j, r) in enumerate(lst):
            pmat[li] = irt_p(a[j], b[j], XG)
        pm = np.clip(pmat, 1e-7, 1 - 1e-7)
        ll = np.sum(np.where(np.array([r for _, r in lst])[:, None] == 1,
                             np.log(pm), np.log(1 - pm)), axis=0)
        post = WG * np.exp(ll - ll.max()); post /= post.sum()
        return float((XG * post).sum())

    # ---- 代理 Oracle AUC（test 行，DKT 同口径）----
    p_all, y_all = [], []
    for pos in test:
        r = rows[pos]
        th_u = eap_user(r["user"])
        L = len(r["corrects"])
        for t in range(L - 1):            # t=0..L-2, target=corrects[t+1]
            k = idx.get(item_key(r["topic"], r["diffs"][t + 1]))
            if k is None:
                continue
            p_all.append(irt_p(a[k], b[k], th_u))
            y_all.append(int(r["next"][t + 1]))
    p_all = np.asarray(p_all); y_all = np.asarray(y_all)
    pos = p_all[y_all == 1]; neg = p_all[y_all == 0]
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    rp = ranks[: len(pos)].sum()
    auc = rp / (len(pos) * len(neg)) - (len(pos) + 1) / (2 * len(neg))
    auc = float(np.clip(auc, 0, 1))
    print("=" * 66)
    print(f"代理 Oracle Ⅰ' (2PL IRT, difficulty-aware)  auc={auc:.4f}  samples={len(p_all)}")
    print(f"对照: Ⅲ' DKT c0=0.7455 ; Ⅱ' 容量扫描上界 c4=0.795")
    print(f"  DKT - 代理 = {0.7455 - auc:+.4f}  (>0 ⇒ 模型已超静态难度/能力参照，补时序=容量杠杆)")
    return auc

if __name__ == "__main__":
    main()