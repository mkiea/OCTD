#!/usr/bin/env python3
"""
验证 Oracle 上界时序稳定性与pOk分布分位数分解
回答审稿人暗伤一：
  1.1 按序列位置t分层，确认Oracle AUC是否稳定
  1.2 pOk分布是否偏斜，极端值是否拉高Oracle AUC
"""
import json
import math
import numpy as np
from sklearn.metrics import roc_auc_score

def main():
    split_path = 'training/data/p0_prereq.split.json'
    data_path = 'training/data/p0_prereq.jsonl'

    with open(split_path, 'r', encoding='utf-8') as f:
        split_meta = json.load(f)
    test_index = set(split_meta['test_index'])

    t_bins = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, None)]
    t_groups = {b: [] for b in t_bins}
    p_bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    p_groups = {b: [] for b in p_bins}

    n_total = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if idx not in test_index:
                continue
            nexts = row['next']
            pOk = row['pOk']
            assert len(nexts) == len(pOk)
            for t, (y, p) in enumerate(zip(nexts, pOk)):
                if y < 0:
                    continue
                for (lo, hi) in t_bins:
                    if t >= lo and (hi is None or t <= hi):
                        t_groups[(lo, hi)].append((p, y))
                        break
                for (lo, hi) in p_bins:
                    if p > lo and p <= hi:
                        p_groups[(lo, hi)].append((p, y))
                        break
                n_total += 1

    print('=== Oracle 按序列位置t分层验证 ===')
    print(f'test set 总有效位置数: {n_total}\n')
    for (lo, hi) in t_bins:
        if not t_groups[(lo, hi)]:
            continue
        ps, ys = zip(*t_groups[(lo, hi)])
        n = len(ps)
        if n < 10:
            print(f't={lo}-{hi}: n={n} 太少跳过')
            continue
        auc = roc_auc_score(ys, ps)
        bin_name = f't={lo}-{hi}' if hi else f't>={lo}'
        print(f'{bin_name:>10} | n={n:<5d} | AUC={auc:.4f}')

    print('\n=== Oracle 按pOk分位数区间分解 ===')
    for (lo, hi) in p_bins:
        if not p_groups[(lo, hi)]:
            continue
        ps, ys = zip(*p_groups[(lo, hi)])
        n = len(ps)
        if n < 10:
            continue
        auc = roc_auc_score(ys, ps)
        count_percent = 100 * n / n_total
        print(f'pOk={lo:.1f}-{hi:.1f} | n={n:<4d} ({count_percent:.0f}%) | AUC={auc:.4f}')

    print('\n=== 全量test set Oracle AUC ===')
    all_ps, all_ys = [], []
    for g in t_groups.values():
        for p, y in g:
            all_ps.append(p)
            all_ys.append(y)
    auc_total = roc_auc_score(all_ys, all_ps)
    print(f'全量 n={len(all_ps)}  Oracle AUC={auc_total:.4f}')

    print('\n=== pOk分布直方图 ===')
    hist = [0] * 10
    for (p, y) in zip(all_ps, all_ys):
        b = int(math.floor(p * 10))
        if b >= 10:
            b = 9
        hist[b] += 1
    for i in range(10):
        lo = i / 10
        hi = (i + 1) / 10
        cnt = hist[i]
        pct = 100 * cnt / len(all_ps)
        print(f'{lo:.1f}-{hi:.1f}: {cnt:4d} ({pct:4.1f}%)')

if __name__ == '__main__':
    main()