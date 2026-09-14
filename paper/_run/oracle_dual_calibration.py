#!/usr/bin/env python3
"""
方案3：Oracle 双口径对照（严格因果 vs 上帝视角）
验证暗伤"Oracle 与 DKT 任务存在因果边界错位"。

结论（同一 p0_prereq 固定测试集，同一 n=2880）：
  A 上帝视角  pOk[t]->next[t]              n=2880  AUC=0.7596   （等价 B；next[t]=corrects[t]，逐项即 pOk[t] 对放置 corrects[t]）
  B 上帝视角  pOk[t+1] -> corrects[t+1]    n=2880  AUC=0.7596   （与 DKT 同目标 a_{t+1}，但泄漏下一步内部态）
  C 严格因果  pOk[t]  -> corrects[t+1]     n=2880  AUC=0.5485   （模型可达，几乎无预测力）
论文正式判定：主读数取 B（上帝视角 0.759），A*_causal=C（0.549）作数据固有可测性对照。
关键点：A、B 等价（索引平移），都代表上帝视角；真正区别是 C —— 不泄漏下一步内部态时本步概率对下一步几乎无预测力。
"""
import json
from sklearn.metrics import roc_auc_score


def main():
    data_path = 'training/data/p0_prereq.jsonl'
    split_path = 'training/data/p0_prereq.split.json'
    with open(data_path, 'r', encoding='utf-8') as f:
        rows = [json.loads(l) for l in f if l.strip()]
    spec = json.load(open(split_path, 'r', encoding='utf-8'))
    test = set(spec['test_index'])

    A_ps, A_ys, nA = [], [], 0
    B_ps, B_ys, nB = [], [], 0
    C_ps, C_ys, nC = [], [], 0
    for idx in test:
        r = rows[idx]
        n = len(r['corrects'])
        for t in range(n):
            # A: 本步自证 pOk[t]->next[t](=corrects[t])，next[0]=-1 被 skip
            nt = r['next'][t]
            if nt >= 0:
                A_ps.append(r['pOk'][t]); A_ys.append(r['corrects'][t]); nA += 1
        for t in range(n - 1):
            # B: pOk[t+1] -> corrects[t+1]
            B_ps.append(r['pOk'][t + 1]); B_ys.append(r['corrects'][t + 1]); nB += 1
            # C: pOk[t] -> corrects[t+1]
            C_ps.append(r['pOk'][t]); C_ys.append(r['corrects'][t + 1]); nC += 1

    print(f'A 本步自证    pOk[t]->corrects[t]     n={nA}  AUC={roc_auc_score(A_ys, A_ps):.4f}')
    print(f'B 上帝视角    pOk[t+1]->corrects[t+1] n={nB}  AUC={roc_auc_score(B_ys, B_ps):.4f}')
    print(f'C 严格因果    pOk[t]->corrects[t+1]   n={nC}  AUC={roc_auc_score(C_ys, C_ps):.4f}')


if __name__ == '__main__':
    main()