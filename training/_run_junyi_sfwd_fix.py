#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Junyi·方案A+修复复跑：纯主任务驱动结构（gate 初值+1，--struct-lambda 0，protect struct_W）

对照上一篇 sfwd（struct_lambda 0.01 → struct_W 被稀疏/先验压回近零、Δ≈0）。
本次只让 struct_W 从主任务前向梯度学习，验证可学习邻接能否被真正学出并带来增益。

数据/口径与 base/sfwd 完全一致（fa2_200000_7_diff，固定测试集 fa2.split，n=421719）。
成功判据 = checkpoint 存在（导出崩溃可忽略，指标由 _eval_ckpt.py 复算）。
"""
import subprocess, sys, os, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "training", "data", "junyi")
TRAIN = os.path.join(ROOT, "training", "train_dkt.py")
EVAL  = os.path.join(ROOT, "training", "_eval_ckpt.py")
DATA  = os.path.join(BASE, "fa2_200000_7_diff.jsonl")
SPLIT = os.path.join(BASE, "fa2.split.json")
EDGES = os.path.join(BASE, "topic_prereq_edges.id40.json")
CKPT  = os.path.join(ROOT, "training", "checkpoints", "junyi_sfwd_fix")
os.makedirs(CKPT, exist_ok=True)

GROUP = "fix"

def run(seed):
    ckpt = os.path.join(CKPT, f"{GROUP}_s{seed}.pt")
    args = [sys.executable, TRAIN,
            "--data", DATA, "--split", SPLIT, "--split-mode", "fixed",
            "--num-topics", "192", "--max-len", "16",
            "--seed", str(seed), "--epochs", "20", "--batch", "64", "--patience", "6",
            "--checkpoint", ckpt,
            "--out", os.path.join(BASE, f"junyi_{GROUP}_s{seed}.onnx")]
    # 纯主任务驱动：结构仍接入前向，但放弃一切向 0 压制的正则（--struct-lambda 0）
    args += ["--graph-edges", EDGES, "--graph-lambda", "1e-3",
             "--struct-lambda", "0", "--struct-forward"]
    r = subprocess.run(args, capture_output=True, text=True)
    if not os.path.exists(ckpt):
        print(f"{GROUP} seed{seed} NO_CKPT rc={r.returncode} stdout={r.stdout[-300:]!r} stderr={r.stderr[-300:]!r}", flush=True)
        return None
    ev = subprocess.run([sys.executable, EVAL, "--data", DATA, "--split", SPLIT,
                         "--checkpoint", ckpt, "--num-topics", "192", "--max-len", "16",
                         "--struct-forward"], capture_output=True, text=True)
    line = next((x.strip() for x in ev.stdout.splitlines() if x.startswith("DKT:")), "")
    if not line:
        print(f"{GROUP} seed{seed} EVAL_FAIL stdout={ev.stdout[-300:]!r}", flush=True)
        return None
    item = eval(line[len("DKT:"):])
    auc = item.get("auc")
    print(f"{GROUP} seed{seed} -> DKT auc={auc} config=struct_lambda0_gate0", flush=True)
    return auc

def main():
    aucs = [run(s) for s in range(3)]
    ok = [x for x in aucs if x is not None]
    print(f"{GROUP}  AUC=" + " ".join(f"{x:.4f}" if x else "ERR" for x in aucs)
          + f"  mean={sum(ok)/len(ok):.4f}  n_valid={len(ok)}", flush=True)

if __name__ == "__main__":
    main()