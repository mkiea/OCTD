#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 junyi_Exercise_table.csv 构建 exercise → 难度档(1-5) 映射。
难度源：seconds_per_fast_problem（专家设定的快速完成秒数，0-60s）。
秒数越小 → 题越易 → 作答越快 → 难度档越低。
分箱：按有效秒数等频分位数切成 5 档（1 最易 ... 5 最难）。
空值/非数字 → 全员中位数箱（兜底档 3）。
输出：junyi_exercise_diff.json {exe_name: int(1-5)}
"""
import csv, json, sys, collections

SRC = r"c:/Users/SolimPurmiss/Desktop/know-map-ai-learning-alpha-v0.3.5/know-map-ai-learning-alpha-v0.3.5/training/data/junyi/srctable/junyi_Exercise_table.csv"
OUT = r"c:/Users/SolimPurmiss/Desktop/know-map-ai-learning-alpha-v0.3.5/know-map-ai-learning-alpha-v0.3.5/training/data/junyi/junyi_exercise_diff.json"

rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig")))
mapping = {}          # exe -> int diff
secs = []             # 有效秒数列表
valid = 0
for r in rows:
    name = r["name"].strip()
    raw = r["seconds_per_fast_problem"].strip()
    try:
        s = int(raw)
    except (ValueError, TypeError):
        s = None
    mapping[name] = s  # 临时存原始秒数，后续替换成档
    if s is not None:
        secs.append(s)
        valid += 1

print(f"total exercises {len(rows)}  valid sec {valid}  blank {len(rows)-valid}")

# 等频分箱：按秒数分位数切 5 档（quintiles）
secs_sorted = sorted(secs)
n = len(secs_sorted)
quantiles = [secs_sorted[int(n * q / 5)] for q in range(1, 5)]  # q1..q4 边界
print("quintile sec bounds:", quantiles)

def bin_sec(s):
    # s 越小越易 → 档越小
    if s <= quantiles[0]:
        return 1
    if s <= quantiles[1]:
        return 2
    if s <= quantiles[2]:
        return 3
    if s <= quantiles[3]:
        return 4
    return 5

# 空值兜底 = 中位数秒数对应档（其最常见的档）
median_sec = secs_sorted[n // 2]
fallback = bin_sec(median_sec)
print("median sec", median_sec, "-> fallback diff", fallback)

out = {}
dist = collections.Counter()
for name, sec in mapping.items():
    d = bin_sec(sec) if sec is not None else fallback
    out[name] = d
    dist[d] += 1

print("diff distribution:", dict(sorted(dist.items())))
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)
print("wrote", OUT, len(out), "exe")