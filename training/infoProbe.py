#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信息量探针（投钱前检查）：计算数据的理论可预测性上界（oracle）
================================================================
合成数据生成时用的是真实概率 pOk（给定当前历史后"本步答对"的真概率）。
把 pOk 透传进训练行后，用它对实际结果算 AUC = 该数据的**理论可预测性上限**
（Aleatoric 上界）：任何模型都只能逼近它，无法超过它。

用法：python training/infoProbe.py --data training/data/<rows>.jsonl --oracle

数据需含 pOk 列（由 dktDataBuilder opts.pOk=true 生成，见 genBridgeProbe.js）。
对照口径：
  oracle(AUC)      = 含隐藏状态的绝对上界（0.76 级），模型不可达；
  DKT/容量扫描(AUC)= 可观察可提取的实测水平（≈0.71），模型可及。
  oracle 与 DKT 之差 ≈ 隐藏 slip/guess 内部状态信息（不可从二进制作答恢复）。
"""
import argparse
import json

import numpy as np


def load_rows(path):
    rows = []
    for line in open(path, "r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        rows.append({
            "corrects": [int(x) for x in o["corrects"]],
            "pOk": [float(x) if x is not None else None for x in o["pOk"]] if "pOk" in o else None,
        })
    return rows


def auc_from_rank(p, y):
    m = y >= 0
    p_, y_ = p[m], y[m]
    pos = p_[y_ == 1]
    neg = p_[y_ == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None, 0
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    rp = ranks[: len(pos)].sum()
    auc = rp / (len(pos) * len(neg)) - (len(pos) + 1) / (2 * len(neg))
    return float(np.clip(auc, 0, 1)), int(len(y_))


def main():
    ap = argparse.ArgumentParser(description="数据理论可预测性上界（oracle 信息量探针）")
    ap.add_argument("--data", required=True)
    args = ap.parse_args()

    rows = load_rows(args.data)
    all_p, all_y = [], []
    for r in rows:
        if not r["pOk"]:
            continue
        all_p.extend(r["pOk"])
        all_y.extend(r["corrects"])
    if not all_p:
        print("该数据无 pOk 列（需由 dktDataBuilder opts.pOk=true 生成），无可算 oracle。")
        return
    p = np.asarray(all_p, dtype=np.float64)
    y = np.asarray(all_y, dtype=np.int64)
    auc, m = auc_from_rank(p, y)
    print(f"ORACLE 真实生成概率  auc={auc:.4f}  samples={m}  (理论可预测性上界；配比 DKT/容量扫描即可判可观察可提取)")
    print("对照: 若 DKT(或容量扫描) 已逼近该上界 ⇒ 可观察可提取到头, 破它必须换/加熵数据;"
          " 差值为隐藏状态不可及信息。")


if __name__ == "__main__":
    main()