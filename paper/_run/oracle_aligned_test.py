# -*- coding: utf-8 -*-
"""同口径验证：在 DKT 评测所用的同一批 2880 个"下一题"测试样本上，
重新计算 Oracle 上界（pOk vs 标签），与"全部 23040 位置"口径的 0.762 对比。

若同口径 oracle 仍 ≈0.762 ⇒ gap① ≈ 0.045 非采样口径伪差，审稿人 P0#1 被证伪；
若明显回落（接近 0.717+）⇒ gap① 部分来自口径差异，需重写主结论。
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "training", "data")


def load_rows(path):
    rows = []
    for line in open(path, "r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        rows.append({"corrects": [int(x) for x in o["corrects"]],
                     "pok": [float(x) for x in o["pOk"]]})
    return rows


def auc(p, y):
    m = [yy >= 0 for yy in y]
    p = [x for x, mm in zip(p, m) if mm]
    y = [x for x, mm in zip(y, m) if mm]
    pos = [x for x, yy in zip(p, y) if yy == 1]
    neg = [x for x, yy in zip(p, y) if yy == 0]
    if not pos or not neg:
        return None, len(y)
    ranked = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg],
                    key=lambda t: t[0])
    rp = sum(i + 1 for i, (_, cl) in enumerate(ranked) if cl == 1)
    np_ = len(pos); nn_ = len(neg)
    a = rp / (np_ * nn_) - (np_ + 1) / (2 * nn_)
    return max(0.0, min(1.0, a)), len(y)


def main():
    rows = load_rows(os.path.join(DATA, "p0_prereq.jsonl"))
    with open(os.path.join(DATA, "p0_prereq.split.json"), "r", encoding="utf-8") as f:
        split = json.load(f)
    test_idx = split["test_index"]
    test_rows = [rows[i] for i in test_idx]

    # ---- 口径 A：全部 23040 位置（复现 paper 0.762） ----
    ap, ay = [], []
    for r in rows:
        ap.extend(r["pok"])
        ay.extend(r["corrects"])
    a_oracle, an = auc(ap, ay)
    print(f"[A] Oracle · 全部 pOk 位置         : auc={a_oracle:.4f}  n={an}")

    # ---- 口径 B：DKT 同款 2880 下一题测试样本（t in [0, L-2]，标签=corrects[t+1]） ----
    # 名义上界 Oracle 与 DKT 任务同语义：用"下一步"真实内部概率 pOk[t+1] 预测"下一步"标签 a[t+1]
    # （即 collate 的 tgt_next=corrects[t+1]、tgt_pok=pOk[t+1]）。旧版误用 pok[t] vs corrects[t]（错 1 位）。
    bp, by = [], []
    for r in test_rows:
        L = len(r["corrects"])
        for t in range(L - 1):  # 与 DKT eval mask: t<=L-2, tgt_next>=0 完全一致
            bp.append(r["pok"][t + 1])
            by.append(r["corrects"][t + 1])
    b_oracle, bn = auc(bp, by)
    print(f"[B] Oracle · 同口径 2880 测试样本   : auc={b_oracle:.4f}  n={bn}")

    # ---- 口径 B 的 Bootstrap 95% CI（n=2880，与 DKT 同 n）----
    rng = __import__("numpy").random.default_rng(42)
    B = 1000
    vals = []
    p_arr = [x for x, yy in zip(bp, by) if yy >= 0]
    y_arr = [x for x, yy in zip(by, by) if x >= 0]
    N = len(y_arr)
    import numpy as np
    for _ in range(B):
        s = rng.integers(0, N, N)
        v, _ = auc([p_arr[i] for i in s], [y_arr[i] for i in s])
        vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    print(f"[B·CI] Oracle 同口径 Bootstrap 95%CI=[{lo:.4f},{hi:.4f}] mean={np.mean(vals):.4f} n={N}")
    # DKT 端（paper 已有）：0.7174 CI=[0.6984,0.7359] n=2880
    print("        DKT(Ⅲ) 端 n=2880：point≈0.717，CI≈[0.698,0.736] ⇒ 两口径现均为 n=2880，可比")

    # ---- 口径 C：全区全部位置但仅训练区 vs 测试区拆分提示 ----
    tp, ty = [], []
    train_set = set(range(len(rows))) - set(test_idx)
    for i in sorted(train_set):
        r = rows[i]
        L = len(r["corrects"])
        for t in range(L):
            tp.append(r["pok"][t]); ty.append(r["corrects"][t])
    c_oracle, cn = auc(tp, ty)
    print(f"[C] Oracle · 训练区全部 {cn} 位置      : auc={c_oracle:.4f}")

    # 主结论
    print("\n判定: 口径B(同2880) 与 口径A(全23040) 之差 "
          f"= {abs(b_oracle - a_oracle):.4f}"
          " → " + ("同口径后 oracle 未明显回落 ⇒ gap 非采样伪差"
                    if a_oracle - b_oracle < 0.01 else
                    "同口径后 oracle 回落 ⇒ gap 存在口径伪差，需重写主结论"))


if __name__ == "__main__":
    main()