#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3-seed 扫描：固定数据与 60/40 切分（同口径原则），仅变模型训练 seed。
给 §5.7 通用性演示提供真实 mean±std，替代占位/单次读数。
运行: python oced_toy/oce_demo_toy_sweep.py
"""
import numpy as np
import torch

from oce_demo_toy import gen_sequences, SeqModel, train_eval, oracle_auc, set_seed

CFG = dict(n_seq=4000, T=60, rho=0.9, sigma_h=0.8, alpha=1.6, beta=0.9, c=0.6,
           obs_noise=0.6)

CONFIGS = [
    ("III 小模型 (GRU h=8, 仅可观察)", 8, 1, False),
    ("II 容量扫描 c0 (GRU h=16)", 16, 1, False),
    ("II 容量扫描 c4 (GRU h=128 x2)", 128, 2, False),
    ("III+z(s+noise) [加特征]", 8, 1, True),
]
EPOCHS = 12
BATCH = 256
LR = 3e-3
SEEDS = [7, 8, 9]
N_SEQ = CFG["n_seq"]


def main():
    # 数据/切分固定（用固定 data_seed，与模型训练 seed 解耦）
    data_seed = 7
    F, Z0, Y, pOk = gen_sequences(N_SEQ, CFG["T"], CFG["rho"], CFG["sigma_h"],
                                  CFG["alpha"], CFG["beta"], CFG["c"],
                                  with_z=False, obs_noise=CFG["obs_noise"], seed=data_seed)
    F = F.astype(np.float32)
    _, Z1, _, _ = gen_sequences(N_SEQ, CFG["T"], CFG["rho"], CFG["sigma_h"],
                                CFG["alpha"], CFG["beta"], CFG["c"],
                                with_z=True, obs_noise=CFG["obs_noise"], seed=data_seed)
    perm = np.random.RandomState(data_seed).permutation(N_SEQ)
    tr, te = perm[:int(N_SEQ * 0.6)], perm[int(N_SEQ * 0.6):]
    split_idx = (tr, te)

    print("=" * 80)
    print("非KT toy 3-seed 扫描：数据/切分固定(data_seed=%d)，仅变模型训练 seed %s"
          % (data_seed, SEEDS))
    print("=" * 80)
    o_n = int(len(Y[:, te].reshape(-1)))
    o = oracle_auc(pOk, Y, split_idx)[0]  # (auc, n) -> 取 auc
    print("Ⅰ Oracle(pOk)   mean=%.4f  (n=%d next-samples, 与seed无关)"
          % (o, o_n))

    base_d = F.shape[-1]
    rows = {}
    for name, hid, layers, ze_flag in CONFIGS:
        ze = Z1 if ze_flag else Z0
        fd = base_d + (1 if ze.shape[-1] else 0)
        vals = []
        for s in SEEDS:
            set_seed(s)
            torch.manual_seed(s); np.random.seed(s)
            model = SeqModel(hid, fd, layers)
            auc, m = train_eval(model, F, ze, Y, split_idx, EPOCHS, BATCH, LR, s)
            vals.append(auc)
            print("  %-34s seed=%d -> %.4f" % (name, s, auc))
        arr = np.array(vals)
        rows[name] = arr
        print("  %-34s mean=%.4f  std=%.4f" % (name, arr.mean(), arr.std()))
    print("-" * 80)

    oracle = o
    l3 = rows[CONFIGS[0][0]].mean()
    c0 = rows[CONFIGS[1][0]].mean()
    c4 = rows[CONFIGS[2][0]].mean()
    fix = rows[CONFIGS[3][0]].mean()
    cap = max(c0, c4)
    tau = 0.03
    g1 = oracle - cap; g2 = cap - l3
    lo = g2 + 1e-4; hi = g1 - 1e-4
    print("gap①(Ⅰ-Ⅱ)=%.4f   gap②(Ⅱ-Ⅲ)=%.4f   τ=%.3f" % (g1, g2, tau))
    print("容量扫描 c0->c4 = %.4f（饱和->非C）" % (c4 - c0))
    print("τ 稳健窗口=(%.4f,%.4f)，两级判定 => %s + %s"
          % (lo, hi, "B(特征暴露不足)" if g1 >= tau else "A(数据固有)",
             "C" if g2 >= tau else "非C(模型贴住可提取上界)"))
    print("加特征 z 收益 mean = %.4f" % (fix - l3))
    print("=" * 80)


if __name__ == "__main__":
    main()