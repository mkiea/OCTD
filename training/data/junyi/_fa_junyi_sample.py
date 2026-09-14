#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 junyi_rt_diff / junyi_rt_nodiff 采样同一组行号，输出逐行匹配的双版本样本。

保证两个版本的序列/顺序/行数完全一致，仅 diffs 字段不同 → 受控变量（难度）。
用法: python _fa_junyi_sample.py 200000 7
"""
import sys, os, json, random

BASE = "c:/Users/SolimPurmiss/Desktop/know-map-ai-learning-alpha-v0.3.5/"
BASE += "know-map-ai-learning-alpha-v0.3.5/training/data/junyi/"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7

SRC_DIFF, SRC_NODIFF = BASE + "junyi_rt_diff.jsonl", BASE + "junyi_rt_nodiff.jsonl"
OUT_DIFF, OUT_NODIFF = BASE + f"fa2_{N}_{SEED}_diff.jsonl", BASE + f"fa2_{N}_{SEED}_nodiff.jsonl"


def count_lines(path):
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def pick(n_lines, want, seed):
    random.seed(seed)
    idx = sorted(random.sample(range(n_lines), want))
    return set(idx)


def stream_select(src, out, keep):
    nw = 0
    with open(src, "r", encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i in keep:
                fout.write(line)
                nw += 1
    return nw


def main():
    nd = count_lines(SRC_DIFF)
    nn = count_lines(SRC_NODIFF)
    assert nd == nn, f"line mismatch {nd} vs {nn}"
    print(f"source lines diff={nd} nodiff={nn}", flush=True)
    want = min(N, nd)
    keep = pick(nd, want, SEED)
    print(f"sampled {len(keep)} rows seed={SEED}", flush=True)
    if os.path.exists(OUT_DIFF):
        os.remove(OUT_DIFF)
    if os.path.exists(OUT_NODIFF):
        os.remove(OUT_NODIFF)
    w1 = stream_select(SRC_DIFF, OUT_DIFF, keep)
    w2 = stream_select(SRC_NODIFF, OUT_NODIFF, keep)
    assert w1 == w2 == len(keep)
    print(f"written diff={w1} nodiff={w2}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()