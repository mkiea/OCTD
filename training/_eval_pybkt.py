#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对 pybkt 数据评估已训练 GRU checkpoint 的 DKT 读数（复用 train_dkt.collate/split_eval）。"""
import os, sys, json, torch
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # training/
sys.path.insert(0, base)
import train_dkt as DKT

DATA = os.path.join(base, "training", "data")

def main():
    arch, hidden, embed, ckpt = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    rows = DKT.load_jsonl(os.path.join(DATA, "pybkt_prereq.jsonl"))
    sp = json.load(open(os.path.join(DATA, "pybkt_prereq.split.json")))
    test_rows = [rows[i] for i in sp["test_index"]]
    num_topics = 192
    mlen = 6
    model = _make(arch, hidden, embed, num_topics, mlen)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        topic, correct, diff, lat, pre, mask, tgt_next, *_ = DKT.collate(test_rows, num_topics, mlen, torch.device("cpu"))
        pn, *_ = model(topic, correct, diff)
    p, y = [], []
    for b in range(len(test_rows)):
        for t in range(mlen):
            if mask[b, t].item() and int(tgt_next[b, t]) >= 0:
                p.append(pn[b, t].item()); y.append(int(tgt_next[b, t]))
    res = DKT.split_eval(p, y)
    print(f"{arch} h{hidden}/e{embed}  DKT  test_auc={res['auc']}  n={res['n']}")

def _make(arch, hidden, embed, num_topics, max_len):
    if arch == "gru":
        return DKT.HybridKT(num_topics=num_topics, max_len=max_len, hidden=hidden, embed_dim=embed)
    if arch == "sakt":
        return DKT.SaktKT(num_topics=num_topics, max_len=max_len, hidden=hidden, embed_dim=embed)
    raise ValueError(arch)

if __name__ == "__main__":
    main()