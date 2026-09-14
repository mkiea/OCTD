#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""仅评估：从已保存 checkpoint 计算 DKT 测试集 AUC（不经 ONNX 导出）。

为何存在：torch.onnx.export 偶尔在原生层段错误，崩溃发生在 DKT 评测打印前，
会丢失缓冲 stdout 让 runner 误判失败——但训练早已完成、checkpoint 有效。
本脚本绕过导出，直接从 checkpoint + 固定测试集(fixed split)复算 DKT 指标，
用于补全 crashed seed 的实验点。
"""
import argparse, json, os, random
import numpy as np
import torch

from train_dkt import HybridKT, collate, split_eval, load_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-topics", type=int, default=192)
    ap.add_argument("--max-len", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--embed-dim", type=int, default=16)
    ap.add_argument("--struct-forward", action="store_true")
    args = ap.parse_args()

    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    rows = load_jsonl(args.data)
    with open(args.split, "r", encoding="utf-8") as f:
        spec = json.load(f)
    train_idx, test_idx = spec["train_index"], spec["test_index"]
    test_rows = [rows[i] for i in test_idx]

    model = HybridKT(num_topics=args.num_topics, max_len=args.max_len,
                     hidden=args.hidden, embed_dim=args.embed_dim,
                     struct_forward=args.struct_forward)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    T = args.max_len
    topic, correct, diff, _lat, _pre, mask, tgt_next, *_ = collate(
        test_rows, model.num_topics, T, torch.device("cpu"))
    with torch.no_grad():
        pn, _ms, _abil, _cal = model(topic, correct, diff)
    dkt_p_next, dkt_gt = [], []
    for b in range(len(test_rows)):
        for t in range(T):
            if mask[b, t].item() and int(tgt_next[b, t]) >= 0:
                dkt_p_next.append(pn[b, t].item())
                dkt_gt.append(int(tgt_next[b, t]))
    res = split_eval(dkt_p_next, dkt_gt)
    print("DKT:", json.dumps(res))
    print("EVAL_OK")


if __name__ == "__main__":
    main()