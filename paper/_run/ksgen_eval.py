#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 ksgen_prereq 上评估已训练模型（Ⅲ 现实模型 / Ⅱ 容量扫描）。"""
import sys, os, json
import torch
_TRAIN = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "training"))
if _TRAIN not in sys.path:
    sys.path.insert(0, _TRAIN)
import train_dkt as DKT

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "training", "data")


def main():
    arch, hidden, embed, ckpt = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    num_topics = int(sys.argv[5]) if len(sys.argv) > 5 else 9   # 模型嵌入容量（topic ∈ [0, num_topics-1]，0=padding）
    data_name = sys.argv[6] if len(sys.argv) > 6 else "ksgen_prereq"   # 数据集 basename（默认 ksgen_prereq）
    rows = DKT.load_jsonl(os.path.join(DATA, f"{data_name}.jsonl"))
    sp = json.load(open(os.path.join(DATA, f"{data_name}.split.json")))
    test_rows = [rows[i] for i in sp["test_index"]]
    mlen = 16
    if arch == "gru":
        model = DKT.HybridKT(num_topics=num_topics, max_len=mlen, hidden=hidden, embed_dim=embed)
    elif arch == "sakt":
        model = DKT.SaktKT(num_topics=num_topics, max_len=mlen, hidden=hidden, embed_dim=embed)
    else:
        raise ValueError(arch)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        fields = DKT.collate(test_rows, num_topics, mlen, torch.device("cpu"))
        topic, correct, diff = fields[0], fields[1], fields[2]
        mask, tgt = fields[5], fields[6]   # 注意：collate 里 mask=idx5，tgt_next=idx6
        pn, *_ = model(topic, correct, diff)
    p, y = [], []
    for b in range(len(test_rows)):
        for t in range(mlen):
            if mask[b, t].item() and int(tgt[b, t]) >= 0:
                p.append(pn[b, t].item()); y.append(int(tgt[b, t]))
    res = DKT.split_eval(p, y)
    print(f"[Ⅲ/Ⅱ] {arch} h{hidden}/e{embed}  DKT(test)  auc={res['auc']}  acc={res['acc']}  brier={res['brier']}  n={res['n']}")


if __name__ == "__main__":
    main()