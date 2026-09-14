#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第三方生成器（pyBKT / KS-ELO）三层分解驱动
================================================================
为每个数据集跑：
  Ⅰ Oracle 上界  —— 由对应 generate 脚本给出（真概率 pOk 对齐 correct[t+1]）
  Ⅲ 现实模型    —— GRU c0 (h32/e16)，3 seed → mean±std
  Ⅱ 可提取上界  —— 容量扫描 GRU c0(32/16)/c2(64/32)/c4(256/64)，各 3 seed → mean±std

共享固定测试集（--split-mode fixed 复用 .split.json 的 test_index），
与主管线/跨分布标定口径一致。seq: topic∈[1,..], 0=padding。

用法：
  python paper/_run/third_party_scan.py pybkt_prereq 193  3
  python paper/_run/third_party_scan.py ksgen_prereq 25   3
（每 seed 每档调用 train_dkt.py 训练 + ksgen_eval.py 评测）
"""
import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN = os.path.join(ROOT, "training", "train_dkt.py")
EVAL = os.path.join(ROOT, "paper", "_run", "ksgen_eval.py")
DATA = os.path.join(ROOT, "training", "data")

CAPS = {"c0": (32, 16), "c2": (64, 32), "c4": (256, 64)}
CKPT = os.path.join(ROOT, "paper", "_run", "ckpt_3p")


def run_train(data_name, num_topics, cap, seed):
    hidden, embed = CAPS[cap]
    ck = os.path.join(CKPT, f"{data_name}_{cap}_s{seed}.pt")
    cmd = [sys.executable, TRAIN,
           "--data", os.path.join(DATA, f"{data_name}.jsonl"),
           "--split-mode", "fixed",
           "--split", os.path.join(DATA, f"{data_name}.split.json"),
           "--num-topics", str(num_topics),
           "--arch", "gru", "--hidden", str(hidden), "--embed-dim", str(embed),
           "--seed", str(seed),
           "--checkpoint", ck,
           ]
    subprocess.run(cmd, check=True, capture_output=False, text=True)
    return ck


def run_eval(data_name, num_topics, cap, seed, ck):
    hidden, embed = CAPS[cap]
    cmd = [sys.executable, EVAL, "gru", str(hidden), str(embed), ck,
           str(num_topics), data_name]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    for line in out.strip().splitlines():
        if "[Ⅲ/Ⅱ]" in line:
            return line.strip()
    return "???"


def main():
    data_name, num_topics = sys.argv[1], int(sys.argv[2])
    seeds = [0, 1, 2] if len(sys.argv) < 4 or int(sys.argv[3]) > 1 else [0]
    results = {}
    for cap in ["c0", "c2", "c4"]:   # 顺序保证 c0 先出（Ⅲ 现实模型）
        for s in seeds:
            ck = run_train(data_name, num_topics, cap, s)
            line = run_eval(data_name, num_topics, cap, s, ck)
            results.setdefault(cap, []).append(line)
            print(line, flush=True)
    print("=" * 78)
    for cap in ["c0", "c2", "c4"]:
        for s, l in zip(seeds, results[cap]):
            print(f"[{data_name} {cap} seed{s}] {l}", flush=True)


if __name__ == "__main__":
    main()