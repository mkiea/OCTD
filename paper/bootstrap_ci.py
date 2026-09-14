# -*- coding: utf-8 -*-
"""
论文统计：Oracle 上界 与 DKT 主结果的 Bootstrap 95% CI（位置级重采样）
=====================================================================
说明：
  - 用真实数据计算（非臆造），B=1000 次按作答位置重采样。
  - Oracle：p0_prereq 全部含 pOk 的位置（23040 个），pOk vs 标签算 AUC。
  - DKT：固定测试集（p0_prereq.split.json 的 test_index），加载 re-train 的 seed-1
    checkpoint 得到逐位 next 预测，再在 2880 个测试位置上 bootstrap。
  - 两 CI 不重叠 ⇒ gap≈0.045 显著。

运行（先重训 seed1 生成 checkpoint）：
  python training/train_dkt.py --data training/data/p0_prereq.jsonl --num-topics 192 \
      --max-len 16 --hidden 32 --embed-dim 16 --epochs 40 --batch 64 --lr 1e-3 \
      --weight-decay 3e-3 --patience 8 --seed 1 --split-mode fixed \
      --split training/data/p0_prereq.split.json \
      --checkpoint paper/_run/pt_s1.pt --out paper/_run/onnx_s1.onnx
  python paper/bootstrap_ci.py
"""
import os
import sys
import json

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, ROOT)

import torch

import train_dkt as T

B = 1000
RNG = np.random.default_rng(42)


def auc_xy(p, y):
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    m = y >= 0
    p, y = p[m], y[m]
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan, len(y)
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    rp = ranks[: len(pos)].sum()
    auc = rp / (len(pos) * len(neg)) - (len(pos) + 1) / (2 * len(neg))
    return float(np.clip(auc, 0, 1)), len(y)


def bootstrap_ci_xy(p, y):
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    m = y >= 0
    p, y = p[m], y[m]
    n = len(y)
    idx = np.arange(n)
    vals = np.empty(B)
    for i in range(B):
        s = RNG.choice(idx, n, replace=True)
        v, _ = auc_xy(p[s], y[s])
        vals[i] = v
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.mean(vals)), float(lo), float(hi), n


def main():
    # ---- Oracle CI（注意：train_dkt.load_jsonl 丢弃 pOk，需读原始 jsonl）----
    data_path = os.path.join(ROOT, "training", "data", "p0_prereq.jsonl")
    rows = T.load_jsonl(data_path)
    all_p, all_y = [], []
    for line in open(data_path, "r", encoding="utf-8"):
        o = json.loads(line)
        if "pOk" in o and o["pOk"]:
            all_p.extend(float(x) for x in o["pOk"])
            all_y.extend(int(x) for x in o["corrects"])
    print("--- ORACLE (p0_prereq, all pOk positions) ---")
    mean, lo, hi, n = bootstrap_ci_xy(all_p, all_y)
    print(f"net-mean={mean:.4f} 95%CI=[{lo:.4f},{hi:.4f}] n={n}")

    # ---- DKT CI on fixed test ----
    split_path = os.path.join(ROOT, "training", "data", "p0_prereq.split.json")
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)
    test_index = split["test_index"]
    test_rows = [rows[i] for i in test_index]
    cp = os.path.join(ROOT, "paper", "_run", "pt_s1.pt")
    model = T.HybridKT(num_topics=192, max_len=16, hidden=32, embed_dim=16)
    model.load_state_dict(torch.load(cp, map_location="cpu"))
    model.eval()
    TT = 16
    topic, correct, diff, lat, pre, mask, tgt_next, tgt_abil, tgt_mast, _tw = T.collate(
        test_rows, model.num_topics, TT, torch.device("cpu"))
    with torch.no_grad():
        pn, ms, _ = model(topic, correct, diff)
    p_list, g_list = [], []
    for b in range(len(test_rows)):
        for t in range(TT):
            if mask[b, t].item() and int(tgt_next[b, t]) >= 0:
                p_list.append(pn[b, t].item())
                g_list.append(int(tgt_next[b, t]))
    print("--- DKT (c0, seed1, fixed test) ---")
    dkt_point, dn = auc_xy(p_list, g_list)
    mean2, lo2, hi2, _ = bootstrap_ci_xy(p_list, g_list)
    print(f"point-auc={dkt_point:.4f} 95%CI=[{lo2:.4f},{hi2:.4f}] n={dn}")


if __name__ == "__main__":
    main()