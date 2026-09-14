#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAKT 结构消融 + 收敛性评估（只读已训练 checkpoint，不再训练）
================================================================
对 training/data/sakt_{nopos,1h,2l,long}_s{0,1,2}.pt 在固定 test 集上重放
train_dkt.py main() 的评测段（collate → model forward → split_eval），
得到与训练时相同口径的 test AUC，供写入 sakt_cross_arch_report.txt。

构造逐 checkpoint 的模型配置（与训练时命令行一一对应）：
  nopos : --no-pos            → use_pos=False, heads=4, layers=1
  1h    : --heads 1           → use_pos=True,  heads=1, layers=1
  2l    : --layers 2          → use_pos=True,  heads=4, layers=2
  long  : 收敛性复核(高epoch) → use_pos=True,  heads=4, layers=1
共同：num_topics=192, max_len=16, hidden=32, embed_dim=16, 无校准头。
"""
import argparse
import json
import torch

from train_dkt import load_jsonl, collate, split_eval, SaktKT

DATA = "data/p0_prereq.jsonl"
SPLIT = "data/p0_prereq.split.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="nopos,1h,2l,long")
    args = ap.parse_args()

    rows = load_jsonl(DATA)
    with open(SPLIT, "r", encoding="utf-8") as f:
        spec = json.load(f)
    test_idx = spec["test_index"]
    test_rows = [rows[i] for i in test_idx]

    num_topics, max_len, hidden, embed_dim = 192, 16, 32, 16

    CFG = {
        "base":  dict(heads=4, layers=1, use_pos=True),
        "nopos": dict(heads=4, layers=1, use_pos=False),
        "1h":    dict(heads=1, layers=1, use_pos=True),
        "2l":    dict(heads=4, layers=2, use_pos=True),
        "long":  dict(heads=4, layers=1, use_pos=True),
    }

    for tag in [t.strip() for t in args.configs.split(",") if t.strip()]:
        cfg = CFG[tag]
        aucs = []
        for s in range(3):
            import os
            ckpt = f"data/sakt_{tag}_s{s}.pt"
            if not os.path.exists(ckpt):
                continue
            model = SaktKT(num_topics=num_topics, max_len=max_len,
                           hidden=hidden, embed_dim=embed_dim,
                           n_heads=cfg["heads"], n_layers=cfg["layers"],
                           use_pok_calib=False, use_pos=cfg["use_pos"])
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            model.eval()
            T = max_len
            topic, correct, diff, lat, pre, mask, tgt_next, _a, _m, _w, _p, _r = collate(
                test_rows, model.num_topics, T, torch.device("cpu"))
            with torch.no_grad():
                pn, ms, _, cal = model(topic, correct, diff)
            p_next, gt = [], []
            for b in range(len(test_rows)):
                for t in range(T):
                    if mask[b, t].item() and int(tgt_next[b, t]) >= 0:
                        p_next.append(pn[b, t].item())
                        gt.append(int(tgt_next[b, t]))
            res = split_eval(p_next, gt)
            aucs.append(res["auc"])
            print(f"SAKT[{tag}] seed{s}: auc={res['auc']:.4f}")
        mean = sum(aucs) / len(aucs)
        std = (sum((a - mean) ** 2 for a in aucs) / len(aucs)) ** 0.5
        print(f"SAKT[{tag}] mean±std = {mean:.4f}±{std:.4f}")


if __name__ == "__main__":
    main()