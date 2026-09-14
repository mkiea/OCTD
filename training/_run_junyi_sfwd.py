#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Junyi 真实数据·方案A+ 前向结构验证（base / sfwd × seed 0,1,2）

数据: fa2_200000_7_diff.jsonl（20万行真实难度）; 固定测试集 fa2.split.json。
真实 topic 级先修边: topic_prereq_edges.id40.json（80 条，覆盖 39/40 topics）。
对比 baseline 与 struct-forward，量化可学习结构在真实结构信号下的增益。
"""
import subprocess, sys, os, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "training", "data", "junyi")
TRAIN = os.path.join(ROOT, "training", "train_dkt.py")
DATA  = os.path.join(BASE, "fa2_200000_7_diff.jsonl")
SPLIT = os.path.join(BASE, "fa2.split.json")
EDGES = os.path.join(BASE, "topic_prereq_edges.id40.json")
CKPT  = os.path.join(ROOT, "training", "checkpoints", "junyi_sfwd")
os.makedirs(CKPT, exist_ok=True)

def run(group, seed):
    args = [sys.executable, TRAIN,
            "--data", DATA, "--split", SPLIT, "--split-mode", "fixed",
            "--num-topics", "192", "--max-len", "16",
            "--seed", str(seed), "--epochs", "20", "--batch", "64", "--patience", "6",
            "--checkpoint", os.path.join(CKPT, f"{group}_s{seed}.pt"),
            "--out", os.path.join(BASE, f"junyi_{group}_s{seed}.onnx")]
    if group == "sfwd":
        args += ["--graph-edges", EDGES, "--graph-lambda", "1e-3", "--struct-lambda", "0.01",
                 "--struct-forward"]
    r = subprocess.run(args, capture_output=True, text=True)
    dkt = next((x.strip() for x in r.stdout.splitlines() if x.startswith("DKT:")), "")
    if r.returncode != 0:
        err = (r.stderr or r.stdout)[-600:]
        print(f"{group} seed{seed} ERR({r.returncode}): {err}", flush=True)
        return None
    print(f"{group} seed{seed} -> {dkt}", flush=True)
    item = eval(dkt[len("DKT:"):])
    return item.get("auc")

def main():
    groups = sys.argv[1:] or ["base", "sfwd"]
    summary = {}
    for g in groups:
        aucs = [run(g, s) for s in range(3)]
        ok = [x for x in aucs if x is not None]
        summary[g] = aucs
        print(f"{g:6s} AUC=" + " ".join(f"{x:.4f}" if x else "ERR" for x in aucs)
              + f"  mean={sum(ok)/len(ok):.4f}", flush=True)

if __name__ == "__main__":
    main()