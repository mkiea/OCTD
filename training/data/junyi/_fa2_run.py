#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Junyi 同序列双版本对照：跑满 6=2版本×3seed，抽取 test DKT AUC 汇总。
diff 与 nodiff 用同一 fa2.split.json（fixed 模式）→ 同测试集公平对比。
用法: python _fa2_run.py
"""
import subprocess, re, sys, os

T = "c:/Users/SolimPurmiss/Desktop/know-map-ai-learning-alpha-v0.3.5/"
T += "know-map-ai-learning-alpha-v0.3.5/training/"
DATA = T + "data/junyi/"
SPLIT = DATA + "fa2.split.json"
TRAIN = T + "train_dkt.py"

CONFIGS = []
for ver in ["diff", "nodiff"]:
    for seed in [0, 1, 2]:
        CONFIGS.append((ver, seed))

results = []
for ver, seed in CONFIGS:
    cmd = [
        sys.executable, TRAIN,
        "--data", DATA + f"fa2_200000_7_{ver}.jsonl",
        "--num-topics", "64", "--max-len", "16",
        "--epochs", "40", "--batch", "128", "--lr", "1e-3",
        "--seed", str(seed),
        "--split", SPLIT,
        "--split-mode", "fixed",
        "--checkpoint", DATA + f"_fa2_{ver}_s{seed}.pt",
        "--out", DATA + f"_fa2_{ver}_s{seed}.onnx",
        "--patience", "12",
    ]
    print(f"\n######## RUN {ver} seed={seed} ########", flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    m = re.search(r'DKT: (\{.*\})', out)
    line = m.group(1) if m else "NO_DKT_LINE"
    # 打印最末几行训练 + DKT 行
    for l in out.strip().splitlines():
        if l.startswith("epoch ") or l.startswith("best val") or l.startswith("DKT:") or l.startswith("early"):
            print("  " + l, flush=True)
    results.append((ver, seed, line))

print("\n===== SUMMARY (test DKT AUC) =====", flush=True)
for ver, seed, line in results:
    print(f"{ver} s{seed}: {line}", flush=True)