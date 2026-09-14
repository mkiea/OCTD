#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Junyi 同序列双版本 × 容量 c0/c2/c4：每个 (版本×容量×seed) 一次训练，
抽取 test DKT AUC 汇总。diff/nodiff 用同一 fa2.split.json（fixed）→ 同测试集公平对比。
用法: python _fa3_cap.py
"""
import subprocess, re, sys, os

T = "c:/Users/SolimPurmiss/Desktop/know-map-ai-learning-alpha-v0.3.5/"
T += "know-map-ai-learning-alpha-v0.3.5/training/"
DATA = T + "data/junyi/"
SPLIT = DATA + "fa2.split.json"
TRAIN = T + "train_dkt.py"

# 容量档: (标签, hidden, embed_dim, [额外flag])
CAPS = [
    ("c0", 32, 16, []),
    ("c2", 64, 32, []),
    ("c4", 256, 64, ["--use-attn"]),
]

CONFIGS = []
for ver in ["diff", "nodiff"]:
    for cap, hidden, embed, extra in CAPS:
        for seed in [0, 1, 2]:
            CONFIGS.append((ver, cap, hidden, embed, extra, seed))

results = []
for ver, cap, hidden, embed, extra, seed in CONFIGS:
    ckpt = DATA + f"_fa3_{ver}_{cap}_s{seed}.pt"
    if os.path.exists(ckpt) and os.path.getsize(ckpt) > 0:
        print(f"\nSKIP {ver} {cap} seed={seed} (checkpoint exists)", flush=True)
        results.append((ver, cap, seed, "SKIPPED"))
        continue
    cmd = [
        sys.executable, TRAIN,
        "--data", DATA + f"fa2_200000_7_{ver}.jsonl",
        "--num-topics", "64", "--max-len", "16",
        "--epochs", "40", "--batch", "128", "--lr", "1e-3",
        "--hidden", str(hidden), "--embed-dim", str(embed),
        "--patience", "12",
        "--seed", str(seed),
        "--split", SPLIT,
        "--split-mode", "fixed",
        "--checkpoint", ckpt,
        "--out", DATA + f"_fa3_{ver}_{cap}_s{seed}.onnx",
    ] + extra
    print(f"\n######## RUN {ver} {cap} seed={seed} ########", flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    m = re.search(r'DKT: (\{.*\})', out)
    line = m.group(1) if m else "NO_DKT_LINE"
    for l in out.strip().splitlines():
        if l.startswith("epoch ") or l.startswith("best val") or l.startswith("DKT:") or l.startswith("early"):
            print("  " + l, flush=True)
    results.append((ver, cap, seed, line))

print("\n===== SUMMARY (test DKT AUC) =====", flush=True)
for ver, cap, seed, line in results:
    print(f"{ver} {cap} s{seed}: {line}", flush=True)