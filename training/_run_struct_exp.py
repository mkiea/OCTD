#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨 seed 结构增益对比实验驱动（base / pres / struct × seed 0,1,2）

固定测试集（--split-mode fixed 复用 p0_prereq.split.json），公平择优。
"""
import subprocess, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "training", "train_dkt.py")
DATA  = os.path.join(ROOT, "training", "data", "p0_prereq.jsonl")
SPLIT = os.path.join(ROOT, "training", "data", "p0_prereq.split.json")
EDGES = os.path.join(ROOT, "training", "data", "p0_prereq.edges.json")
CKPT  = os.path.join(ROOT, "training", "checkpoints", "struct")
OUT   = os.path.join(ROOT, "training", "data")
os.makedirs(CKPT, exist_ok=True)

def run(group, seed):
    args = [sys.executable, TRAIN,
            "--data", DATA, "--split", SPLIT, "--split-mode", "fixed",
            "--num-topics", "192", "--max-len", "24",
            "--seed", str(seed), "--epochs", "20", "--batch", "64", "--patience", "8",
            "--checkpoint", os.path.join(CKPT, f"{group}_s{seed}.pt"),
            "--out", os.path.join(OUT, f"struct_{group}_s{seed}.onnx")]
    if group == "pres":
        args += ["--use-pres"]
    if group in ("struct", "sfwd"):
        args += ["--graph-edges", EDGES, "--graph-lambda", "1e-3", "--struct-lambda", "0.01"]
    if group == "sfwd":
        args += ["--struct-forward"]
    r = subprocess.run(args, capture_output=True, text=True)
    dkt = ""
    for line in r.stdout.splitlines():
        if line.startswith("DKT:"):
            dkt = line.strip()
    err = ""
    if r.returncode != 0:
        err = (r.stderr or r.stdout)[-800:]
    print(f"{group} seed{seed} -> {dkt}", flush=True)
    if err:
        print(f"  ERR({r.returncode}): {err}", flush=True)
    return dkt

def main():
    groups = sys.argv[1:] or ["base", "pres", "struct"]
    summary = {}
    for group in groups:
        aucs = []
        for seed in range(3):
            d = run(group, seed)
            # 提取 auc
            item = eval(d[len("DKT:"):]) if d.startswith("DKT:") else {}
            aucs.append(item.get("auc"))
        summary[group] = aucs
    print("\n=== 汇总（固定测试集 3-seed）===")
    for g, aucs in summary.items():
        mean = sum([x for x in aucs if x is not None]) / len([x for x in aucs if x is not None])
        print(f"{g:8s} AUC=" + " ".join(f"{x:.4f}" for x in aucs) + f"  mean={mean:.4f}")

if __name__ == "__main__":
    main()