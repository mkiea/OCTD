#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BKT + DKT 混合知识追踪模型 —— 训练器（方案 B：Python(PyTorch) → ONNX → onnxruntime-node）
======================================================================================
「混合」的含义（与 brainscore 侧 core/knowledgeTracker.js 的 BKT 兼容、可回退）：
  1. COLD-START / 概念先验（BKT 侧）：
     初始隐藏态 h0 由首个知识点的嵌入经 init_proj 生成 → 网络在"零作答/短交互"时
     的输出自动贴近该知识点的先验掌握水平，等价于把 BKT 的 P(L0) 烘焙进 DKT。
  2. 长程依赖 / 概念转移（DKT 侧）：
     GRU 把 (知识点, 对错, 难度) 交互序列编码为隐状态，建模概念间转移与长期记忆。
  3. 难度感知（DAS3H 思想）：
     难度嵌入参与输入编码；next-correct 头拼接难度嵌入，修正"同一掌握度下不同难度
     答对概率不同"。
  4. 多任务头（对齐刘美含论文 / 现有 tracing 契约）：
       mastery_score  → p_mastery(0-100，除以 100 即 p_mastery)
       p_next_correct → DKT 学到的下一题答对概率代理
       ability_level  → Bloom 认知层次分类（0-3，落 user_ability_profile）
  5. 推理侧融合（Node 端 core/dkt.js）：
     以 DKT 的 mastery_score 为稳定掌握概率，再用 BKT 解析层 predictNextCorrect /
     recommendDifficulty 在给定难度下推导下一题答对概率与推荐难度 → 输出契约与
     /api/knowledge/tracing 完全一致，BKT 引擎可作为模型缺失时的回退。

依赖隔离：仅本 training/ 目录使用 torch/numpy/onnx，不进 Node 运行期。
数据来源：优先读 JSONL 数据（--data，由 buildTrainingSet 导出），否则合成生成。
导出：固定 I/O 名 seq_topic/seq_correct/seq_diff → p_next_correct/mastery_score/ability_level。
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# 1. 数据
# --------------------------------------------------------------------------- #

def gen_synthetic(num_topics=32, max_len=16, per_topic=40, seed=0):
    """生成"先难后易/学习进步"的可信作答序列（含难度）与逐位置标签。

    每个知识点有一个真实掌握轨迹：从 low→high 学习；答对概率随掌握度与难度变化；
    加入 slip/guess 噪声。标签：
      next_correct[t] = 第 t+1 次是否答对(最后一位无标签)
      ability[t]      = 由局部掌握度映射 0-3（对应薄弱/一般/良好/优秀）
    """
    rng = random.Random(seed)
    rows = []  # {topic:int, corrects:[0/1], diffs:[1-5], next:[0/1 or -1], abil:[0-3], mastery_tgt:[0-1]}
    for topic in range(num_topics):
        prior = rng.uniform(0.15, 0.5)          # 初始掌握度
        learn_rate = rng.uniform(0.02, 0.08)    # 每次作答后的学习增量
        final = min(0.98, prior + learn_rate * (per_topic + rng.uniform(5, 20)))
        mastery = prior
        step = (final - prior) / max(per_topic, 1)
        corrects, diffs, nexts, abils, mast_tgt = [], [], [], [], []
        for i in range(per_topic):
            diff = rng.randint(1, 5)
            slip = 0.05 + diff * 0.04      # 越难失误越高
            guess = 0.35 - diff * 0.05     # 越难蒙对越低
            p_ok = min(0.999, max(0.001, mastery * (1 - slip) + (1 - mastery) * guess))
            ok = 1 if rng.random() < p_ok else 0
            corrects.append(ok)
            diffs.append(diff)
            mast_tgt.append(round(float(min(mastery, 0.999)), 4))  # 真值掌握度（单调学习轨迹）
            if i > 0:
                nexts.append(ok)
                # 局部掌握度 → 0-3 档（条图口径：优秀≥0.8/良好≥0.6/一般≥0.4/薄弱<0.4）
                m = sum(corrects[i - 4:i]) / 4.0 if i >= 4 else mastery
                abils.append(0 if m < 0.4 else (3 if m >= 0.8 else (2 if m >= 0.6 else 1)))
            else:
                nexts.append(-1)
                abils.append(-1)
            mastery = min(0.99, mastery + step)
        # 截短到 max_len（保留尾部，反映最近状态）
        rows.append({
            "topic": topic,
            "corrects": corrects[-max_len:],
            "diffs": diffs[-max_len:],
            "next": nexts[-max_len:],
            "abil": abils[-max_len:],
            "mastery_tgt": mast_tgt[-max_len:],
        })
    return rows


def load_jsonl(path):
    rows = []
    for line in open(path, "r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        t = obj["topic"]
        # topic：标量(单知识点序列，合成数据) 或 数组(每条交互一个知识点，如 Junyi 真实数据)。
        # collate 的 torch.tensor(topic) 对两者都能兼容赋值(标量广播/数组逐位)。
        topic = [int(x) for x in t] if isinstance(t, list) else int(t)
        rows.append({
            "topic": topic,
            "corrects": [int(x) for x in obj["corrects"]],
            "diffs": [int(x) for x in obj["diffs"]],
            "next": [int(x) for x in obj.get("next", [])],
            "abil": [int(x) for x in obj.get("abil", [])],
            "mastery_tgt": [float(x) for x in obj["mastery_tgt"]] if "mastery_tgt" in obj else None,
            "latencys": [float(x) for x in obj["latencys"]] if "latencys" in obj else None,
            "weights": [float(x) for x in obj["weights"]] if "weights" in obj else None,  # 样本加权[方向B]
            "pres": [float(x) for x in obj["pres"]] if "pres" in obj else None,  # 先修掌握度[结构入输入]
            "pok": [float(x) for x in obj["pOk"]] if "pOk" in obj else None,  # 真生成概率 pOk（校准辅助头监督）
            "regimes": [int(x) for x in obj["regimes"]] if "regimes" in obj else None,  # 真实认知机制(探针④d输入特征)
        })
    return rows


def collate(rows, num_topics, max_len, device):
    """填充为 [B, T]，并构建逐位置标签。返回 (topic,correct,diff,lat,pres,mask,tgt_next,tgt_abil,tgt_mast,tgt_w)。

    tgt_w[B,T]：next 监督的每样本权重。对齐语义 tgt_next[b,t]=corrects[t+1]，故用 weights[t+1]；
    行无 weights（未加权口径）→ 全 1，退回原无加权 BCE。
    pres[B,T]：先修掌握度特征（结构入输入，[0,1]）；行无 pres → 全 0（不注入，等价无先修信息）。
    """
    B = len(rows)
    T = max_len
    topic = torch.zeros((B, T), dtype=torch.long, device=device)
    correct = torch.zeros((B, T), dtype=torch.long, device=device)
    diff = torch.zeros((B, T), dtype=torch.long, device=device)
    lat = torch.zeros((B, T), dtype=torch.float32, device=device)  # 作答耗时特征（归一化[0,1]；缺省 0）
    pre = torch.zeros((B, T), dtype=torch.float32, device=device)  # 先修掌握度（归一化[0,1]；缺省 0）
    tgt_next = torch.full((B, T), -1, dtype=torch.long, device=device)
    tgt_abil = torch.full((B, T), -1, dtype=torch.long, device=device)
    tgt_mast = torch.full((B, T), float("nan"), dtype=torch.float32, device=device)
    tgt_pok = torch.full((B, T), float("nan"), dtype=torch.float32, device=device)  # 校准辅助头目标（真概率 pOk[t+1]）
    tgt_w = torch.ones((B, T), dtype=torch.float32, device=device)
    mask = torch.zeros((B, T), dtype=torch.long, device=device)  # 非 padding 位置
    regime_feat = torch.zeros((B, T), dtype=torch.long, device=device)  # 真实认知机制输入特征(know=0..unk=3；缺省0)
    for b, r in enumerate(rows):
        L = len(r["corrects"])
        topic[b, :L] = torch.tensor(r["topic"], dtype=torch.long)
        correct[b, :L] = torch.tensor(r["corrects"], dtype=torch.long)
        diff[b, :L] = torch.tensor(r["diffs"], dtype=torch.long)
        if r.get("latencys"):
            lat[b, :L] = torch.tensor(r["latencys"][:L], dtype=torch.float32)
        if r.get("pres"):
            pre[b, :L] = torch.tensor(r["pres"][:L], dtype=torch.float32)
        mask[b, :L] = 1
        if r.get("mastery_tgt"):
            tgt_mast[b, :L] = torch.tensor(r["mastery_tgt"], dtype=torch.float32)
        if r.get("weights"):
            tgt_w[b, :L] = torch.tensor(r["weights"][:L], dtype=torch.float32)
        if r.get("regimes"):
            regime_feat[b, :L] = torch.tensor(r["regimes"][:L], dtype=torch.long)
        for t in range(L):
            if t + 1 < L:
                tgt_next[b, t] = r["next"][t + 1]
                tgt_abil[b, t] = r["abil"][t + 1]
                if r.get("pok") is not None:
                    tgt_pok[b, t] = r["pok"][t + 1]  # 校准目标：第 t 步预测的正确[t+1] 的真概率
    return topic, correct, diff, lat, pre, mask, tgt_next, tgt_abil, tgt_mast, tgt_w, tgt_pok, regime_feat


# --------------------------------------------------------------------------- #
# 2. 模型
# --------------------------------------------------------------------------- #

class DifferentiableBKT(nn.Module):
    """可微 BKT 先验分支：BKT 后验掌握概率作为独立输出，与 DKT 的 mastery 头加权融合。

    与纯 Python 基线(bkt_next_sequence)的关键区别：prior/learn/slip/guess 均为可学习参数
    （经 sigmoid 约束在 (0,1)），slip/guess 按难度分区学习；前向按 BKT 的 autoregressive
    后验更新保留动力学（强归纳偏置），同时整条路径可被反向梯度训练（融合后由 mastery MSE 监督）。
    推理侧 BKT 参数仍可单独导出使用 → 维持"模型缺失时回退纯 BKT"的可回退契约。
    """
    def __init__(self, n_diff=6):
        super().__init__()
        self.logit_prior = nn.Parameter(torch.tensor(-0.62, dtype=torch.float32))  # sigmoid≈0.35
        self.logit_learn = nn.Parameter(torch.tensor(-1.99, dtype=torch.float32))  # sigmoid≈0.12
        self.logit_slip = nn.Parameter(torch.full((n_diff,), -2.20, dtype=torch.float32))   # 基准≈0.10
        self.logit_guess = nn.Parameter(torch.full((n_diff,), -1.10, dtype=torch.float32))  # 基准≈0.25
        self.eps = 1e-6

    def _posterior_sequence(self, correct, diff):
        """返回 (L_before[B,T], L_after[B,T])：每位作答前/后的 BKT 掌握后验概率(0-1)。"""
        B, T = correct.shape
        slip = torch.sigmoid(self.logit_slip[diff])    # [B,T] 难度耦合 slip
        guess = torch.sigmoid(self.logit_guess[diff])  # [B,T]
        prior = torch.sigmoid(self.logit_prior)
        learn = torch.sigmoid(self.logit_learn)
        # 初始后验 = 可学习先验 P(L0)，以张量广播保保持梯度（勿转 float，否则切断 logit_prior 的梯度）
        L = prior.unsqueeze(0).expand(B)
        before, after = [], []
        for t in range(T):
            before.append(L)  # 第 t 位作答前的后验（即更新前）
            cor = correct[:, t]
            s, g = slip[:, t], guess[:, t]
            # p(对)=L*(1-s)+(1-L)*g；观察到答案后做贝叶斯后验更新
            num = torch.where(cor == 1, L * (1 - s), L * s)
            denom = num + torch.where(cor == 1, (1 - L) * g, (1 - L) * (1 - g))
            L = torch.clamp(num / (denom + self.eps), 0.0, 1.0)
            L = torch.clamp(L + (1 - L) * learn, 0.0, 1.0)  # 学习转移
            after.append(L)  # 观察到 answer[t] 后的后验，可作为 GRU 输入特征
        return torch.stack(before, dim=1), torch.stack(after, dim=1)

    def forward(self, correct, diff, return_after=False):
        """correct/diff: [B,T] long。
        return_after=False → L_before[B,T]（第 t 位作答前的掌握先验，供 mastery 头融合，向后兼容）；
        return_after=True  → (L_before, L_after)[B,T]（L_after 供作为 GRU 输入特征，走 next 头 → 影响 auc）。"""
        before, after = self._posterior_sequence(correct, diff)
        if return_after:
            return before, after
        return before


class CausalSelfAttn(nn.Module):
    """单头因果自注意力（方向 D）：对 GRU 隐态做带掩码的自注意力编码，
    让 next 头能按历史上下文加权组合隐态，而非仅取当前位 h_t。
    只改内部表示，输入/输出 ONNX 契约保持 3 入 3 出不变；掩码保证不引入未来信息。"""
    def __init__(self, hidden):
        super().__init__()
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.gate = nn.Linear(hidden, hidden)
        self.scale = hidden ** 0.5

    def forward(self, h, mask):
        # h[B,T,H], mask[B,T] 非 padding(0/1)
        B, T, H = h.shape
        q = self.q(h); k = self.k(h); v = self.v(h)      # [B,T,H]
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale  # [B,T,T]
        # 因果：位置 i 只看 ≤i
        causal = torch.triu(torch.ones(T, T, device=scores.device), diagonal=1).bool()
        scores = scores.masked_fill(causal.unsqueeze(0), -1e9)
        # 填充：padding 位置不参与被 attend
        pad = ~mask.bool().unsqueeze(1)                  # [B,1,T]
        scores = scores.masked_fill(pad, -1e9)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v)                      # [B,T,H]
        return torch.tanh(self.gate(torch.tanh(h + ctx)))  # residual + gate


class SaktKT(nn.Module):
    """非循环替换架构：SAKT（Self-Attentive Knowledge Tracing, Pandey & Karypis 2019）。

    目的：跨架构验证"0.72 平台是否为数据上限"——若 GRU 只是归纳偏置不对，换纯因果
    多头自注意力（近年 KT 主流、无循环、每位置直接 attend 全部历史）应能吃掉 gap。
    forward/输出契约与 HybridKT 完全一致（p_next/mastery/abil + 可选 calib），
    因此 training / eval / 固定 split 流程可无缝复用，与 GRU 在相同数据上直接可比。

    与 GRU 族的关键差异：无循环结构、无顺序归纳偏置，靠位置编码 + 因果自注意力
    显式建模"下一题条件概率 = f(历史对错, 难度, 位置)"。
    """
    def __init__(self, num_topics=32, max_len=16, hidden=32, embed_dim=16,
                 n_heads=4, n_layers=1, use_pok_calib=False, use_pos=True):
        super().__init__()
        self.num_topics = num_topics
        self.max_len = max_len
        self.hidden = hidden                    # FFN 隐藏维
        self.embed_dim = embed_dim              # 注意力的 d_model（各子空间可共享）
        self.n_heads = max(1, n_heads)
        self.n_layers = max(1, n_layers)
        self.topic_emb = nn.Embedding(num_topics, embed_dim, padding_idx=0)
        self.diff_emb = nn.Embedding(6, embed_dim, padding_idx=0)
        self.correct_emb = nn.Embedding(2, embed_dim)
        self.lat_lin = nn.Linear(1, embed_dim)
        self.pres_lin = nn.Linear(1, embed_dim)
        self.regime_emb = nn.Embedding(4, embed_dim, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, max_len, embed_dim))  # 可学习位置编码
        self.use_pos = use_pos
        if self.use_pos:
            nn.init.normal_(self.pos, mean=0.0, std=0.02)
        self.in_proj = nn.Linear(embed_dim, embed_dim)
        # 堆叠因果多头自注意力 + FFN（残差 + LayerNorm）
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, self.n_heads, batch_first=True) for _ in range(self.n_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim, hidden), nn.ReLU(), nn.Linear(hidden, embed_dim))
            for _ in range(self.n_layers)
        ])
        self.norm1 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(self.n_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(self.n_layers)])
        # 输出头（与 GRU 同契约：p_next 拼接下一题难度；mastery 取当前位）
        self.next_head = nn.Linear(embed_dim + embed_dim, 1)
        self.mastery_head = nn.Linear(embed_dim, 1)
        self.abil_head = nn.Linear(embed_dim, 4)
        self.calib_net = nn.Sequential(
            nn.Linear(embed_dim + embed_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        ) if use_pok_calib else None
        self.abil_w = 0.3

    def forward(self, topic, correct, diff, latency=None, pres=None, regime_in=None, next_diff=None, return_hidden=False):
        B, T = topic.shape
        x = self.topic_emb(topic) + self.diff_emb(diff) + self.correct_emb(correct)
        if latency is not None:
            x = x + self.lat_lin(latency.unsqueeze(-1))
        if pres is not None:
            x = x + self.pres_lin(pres.unsqueeze(-1))
        if regime_in is not None:
            x = x + self.regime_emb(regime_in)
        if self.use_pos:
            x = x + self.pos[:, :T, :]
        x = torch.tanh(self.in_proj(x))
        pad_mask = (topic == 0)  # [B,T] bool, True=需被 attend 屏蔽
        for i in range(self.n_layers):
            # 因果：位置 t 只看 ≤t（用 attn_mask 防未来泄漏）
            causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            h = self.attn_layers[i](x, x, x, key_padding_mask=pad_mask, attn_mask=causal)[0]
            x = self.norm1[i](x + h)
            f = self.ffn_layers[i](x)
            x = self.norm2[i](x + f)
        h = x  # [B,T,embed_dim]
        mastery = torch.sigmoid(self.mastery_head(h))[..., 0]
        if next_diff is None:
            nd = torch.cat([diff[:, 1:], diff[:, -1:]], dim=1)
        else:
            nd = next_diff
        nd_emb = self.diff_emb(nd)
        p_next = torch.sigmoid(self.next_head(torch.cat([h, nd_emb], dim=-1)))[..., 0]
        abil = self.abil_head(h)
        calib = (torch.sigmoid(self.calib_net(torch.cat([h, nd_emb], dim=-1)))[..., 0]
                 if self.calib_net is not None else None)
        if return_hidden:
            return p_next, mastery, abil, calib, h
        return p_next, mastery, abil, calib


class SaintKT(nn.Module):
    """Transformer 基替换架构：SAINT（Separate-Attention Intervention Network, Choi et al. 2020）。

    跨架构验证用：把 SAKT 的"单塔自注意力"拆成**双塔**——一个编码器处理
    (题目 topic embedding + 难度 diff) 序列，另一个处理 (作答 correct) 序列，
    两塔各自堆叠因果自注意力后再交叉融合，最后再接下一题难度作预测。
    这是 SAINT 的核心结构差异（习题嵌入与作答嵌入分离、encoder-decoder 型分离注意），
    其余契约（forward 4 输出、training/eval/fixed split 复用）与 SaktKT 完全一致，
    故可与 GRU/SAKT/GKT/DKVMN 在相同数据上直接可比。

    语义对齐：pn[:, t] = P(correct_{t+1})，用第 t 次交互的融合表示读取 topic[t+1]。"""
    def __init__(self, num_topics=32, max_len=16, hidden=32, embed_dim=16,
                 n_heads=4, n_layers=1, use_pok_calib=False, use_pos=True):
        super().__init__()
        self.num_topics = num_topics
        self.max_len = max_len
        self.hidden = hidden
        self.embed_dim = embed_dim
        self.n_heads = max(1, n_heads)
        self.n_layers = max(1, n_layers)
        # 双塔输入：题目/难度 塔 与 作答 塔 各自独立嵌入
        self.topic_emb = nn.Embedding(num_topics, embed_dim, padding_idx=0)
        self.diff_emb = nn.Embedding(6, embed_dim, padding_idx=0)
        self.correct_emb = nn.Embedding(2, embed_dim)
        self.lat_lin = nn.Linear(1, embed_dim)
        self.pres_lin = nn.Linear(1, embed_dim)
        self.regime_emb = nn.Embedding(4, embed_dim, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.use_pos = use_pos
        if self.use_pos:
            nn.init.normal_(self.pos, mean=0.0, std=0.02)
        self.in_proj_quest = nn.Linear(embed_dim, embed_dim)    # 题目塔输入投影
        self.in_proj_resp = nn.Linear(embed_dim, embed_dim)     # 作答塔输入投影
        # 双塔各自堆叠因果多头自注意力 + FFN
        self.attn_quest = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, self.n_heads, batch_first=True) for _ in range(self.n_layers)
        ])
        self.attn_resp = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, self.n_heads, batch_first=True) for _ in range(self.n_layers)
        ])
        self.ffn_q = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim, hidden), nn.ReLU(), nn.Linear(hidden, embed_dim)) for _ in range(self.n_layers)])
        self.ffn_r = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim, hidden), nn.ReLU(), nn.Linear(hidden, embed_dim)) for _ in range(self.n_layers)])
        self.norm1_q = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(self.n_layers)])
        self.norm2_q = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(self.n_layers)])
        self.norm1_r = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(self.n_layers)])
        self.norm2_r = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(self.n_layers)])
        # 两塔融合投影
        self.fuse = nn.Linear(embed_dim + embed_dim, embed_dim)
        self.next_head = nn.Linear(embed_dim + embed_dim, 1)
        self.mastery_head = nn.Linear(embed_dim, 1)
        self.abil_head = nn.Linear(embed_dim, 4)
        self.calib_net = (nn.Sequential(
            nn.Linear(embed_dim + embed_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        ) if use_pok_calib else None)
        self.abil_w = 0.3

    def _stack(self, x, pad_mask, T, attns, ffns, n1, n2):
        for i in range(self.n_layers):
            causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            h = attns[i](x, x, x, key_padding_mask=pad_mask, attn_mask=causal)[0]
            x = n1[i](x + h)
            x = n2[i](x + ffns[i](x))
        return x

    def forward(self, topic, correct, diff, latency=None, pres=None, regime_in=None, next_diff=None, return_hidden=False):
        B, T = topic.shape
        x_quest = self.topic_emb(topic) + self.diff_emb(diff)
        x_resp = self.correct_emb(correct)
        if latency is not None:
            x_resp = x_resp + self.lat_lin(latency.unsqueeze(-1))
        if pres is not None:
            x_resp = x_resp + self.pres_lin(pres.unsqueeze(-1))
        if regime_in is not None:
            x_resp = x_resp + self.regime_emb(regime_in)
        if self.use_pos:
            x_quest = x_quest + self.pos[:, :T, :]
            x_resp = x_resp + self.pos[:, :T, :]
        x_quest = torch.tanh(self.in_proj_quest(x_quest))
        x_resp = torch.tanh(self.in_proj_resp(x_resp))
        pad_mask = (topic == 0)
        hq = self._stack(x_quest, pad_mask, T, self.attn_quest, self.ffn_q, self.norm1_q, self.norm2_q)
        hr = self._stack(x_resp, pad_mask, T, self.attn_resp, self.ffn_r, self.norm1_r, self.norm2_r)
        h = torch.tanh(self.fuse(torch.cat([hq, hr], dim=-1)))   # 双塔融合表示
        mastery = torch.sigmoid(self.mastery_head(h))[..., 0]
        if next_diff is None:
            nd = torch.cat([diff[:, 1:], diff[:, -1:]], dim=1)
        else:
            nd = next_diff
        nd_emb = self.diff_emb(nd)
        p_next = torch.sigmoid(self.next_head(torch.cat([h, nd_emb], dim=-1)))[..., 0]
        abil = self.abil_head(h)
        calib = (torch.sigmoid(self.calib_net(torch.cat([h, nd_emb], dim=-1)))[..., 0]
                 if self.calib_net is not None else None)
        if return_hidden:
            return p_next, mastery, abil, calib, h
        return p_next, mastery, abil, calib


class AktKT(nn.Module):
    """Transformer 基替换架构：AKT（Attention-based Knowledge Tracing, Pandey et al. 2021）。

    跨架构验证用：在因果自注意力基座上引入 AKT 的**Rasch 型难度系数**——对每个 (topic, 难度档)
    维护可学习的一维"题目难度"标量，与按学生的能力交互，对该题预测 logit 施加 Rasch 偏置；
    同时用对该题难度嵌入的**跨注意力**把历史信息按与下一题的 Rasch 相关度加权读取。
    这是 AKT"以幂函数情境化（Rasch 化）注意力显式建模习得顺序"的核心结构元素的最小实现，
    其余契约与 SaktKT 完全一致，可在相同数据上与其余架构直接可比。

    语义对齐：pn[:, t] = P(correct_{t+1})，用第 t 次交互后的表示对 topic[t+1] 作 Rasch 偏置预测。"""
    def __init__(self, num_topics=32, max_len=16, hidden=32, embed_dim=16,
                 n_heads=4, n_layers=1, use_pok_calib=False, use_pos=True):
        super().__init__()
        self.num_topics = num_topics
        self.max_len = max_len
        self.hidden = hidden
        self.embed_dim = embed_dim
        self.n_heads = max(1, n_heads)
        self.n_layers = max(1, n_layers)
        self.topic_emb = nn.Embedding(num_topics, embed_dim, padding_idx=0)
        self.diff_emb = nn.Embedding(6, embed_dim, padding_idx=0)
        self.correct_emb = nn.Embedding(2, embed_dim)
        self.lat_lin = nn.Linear(1, embed_dim)
        self.pres_lin = nn.Linear(1, embed_dim)
        self.regime_emb = nn.Embedding(4, embed_dim, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.use_pos = use_pos
        if self.use_pos:
            nn.init.normal_(self.pos, mean=0.0, std=0.02)
        self.in_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, self.n_heads, batch_first=True) for _ in range(self.n_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim, hidden), nn.ReLU(), nn.Linear(hidden, embed_dim))
            for _ in range(self.n_layers)
        ])
        self.norm1 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(self.n_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(self.n_layers)])
        # Rasch 偏置：每 (topic, 难度档) 一个可学习难度标量 b_it，与能力交互作用于 logit
        self.rasch_b = nn.Embedding(num_topics * 6, 1)
        nn.init.zeros_(self.rasch_b.weight)
        self.ability = nn.Linear(embed_dim, 1)               # 由历史表示估计学生能力分
        self.mix = nn.Linear(embed_dim + embed_dim, hidden)  # 跨注意力后的融合
        self.next_head = nn.Linear(hidden + embed_dim, 1)
        self.mastery_head = nn.Linear(hidden, 1)
        self.abil_head = nn.Linear(hidden, 4)
        self.calib_net = (nn.Sequential(
            nn.Linear(hidden + embed_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        ) if use_pok_calib else None)
        self.abil_w = 0.3

    def forward(self, topic, correct, diff, latency=None, pres=None, regime_in=None, next_diff=None, return_hidden=False):
        B, T = topic.shape
        x = self.topic_emb(topic) + self.diff_emb(diff) + self.correct_emb(correct)
        if latency is not None:
            x = x + self.lat_lin(latency.unsqueeze(-1))
        if pres is not None:
            x = x + self.pres_lin(pres.unsqueeze(-1))
        if regime_in is not None:
            x = x + self.regime_emb(regime_in)
        if self.use_pos:
            x = x + self.pos[:, :T, :]
        x = torch.tanh(self.in_proj(x))
        pad_mask = (topic == 0)
        for i in range(self.n_layers):
            causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            h = self.attn_layers[i](x, x, x, key_padding_mask=pad_mask, attn_mask=causal)[0]
            x = self.norm1[i](x + h)
            x = self.norm2[i](x + self.ffn_layers[i](x))
        h = x
        # 跨注意力：query 用下一题题目嵌入，key/value 用历史表示 → 与下一题 Rasch 相关度加权读取
        q_next = self.topic_emb(topic)
        scores = torch.matmul(q_next, h.transpose(-1, -2)) / (self.embed_dim ** 0.5)
        causal = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal.unsqueeze(0), -1e9)
        # 填充：padding 位置不参与被 attend（sakt 同款，mask 填 padding 位）
        scores = scores.masked_fill(pad_mask.unsqueeze(1), -1e9)
        ctx = torch.matmul(torch.softmax(scores, dim=-1), h)   # [B,T,embed_dim]
        m = torch.tanh(self.mix(torch.cat([h, ctx], dim=-1)))  # [B,T,hidden]
        mastery = torch.sigmoid(self.mastery_head(m))[..., 0]
        if next_diff is None:
            nd = torch.cat([diff[:, 1:], diff[:, -1:]], dim=1)
        else:
            nd = next_diff
        nd_emb = self.diff_emb(nd)
        ability = self.ability(h)[..., 0]                      # 学生能力分 [B,T]
        # Rasch 偏置：b[(topic, diff)] 作用于该题 logit（pred = σ(MLP(·) + b·ability 交互））
        bidx = (topic * 6 + diff).clamp(0, self.rasch_b.weight.shape[0] - 1)
        b_it = self.rasch_b(bidx)[..., 0]                      # [B,T]
        p_next_emb = self.next_head(torch.cat([m, nd_emb], dim=-1))[..., 0]
        p_next = torch.sigmoid(p_next_emb + b_it * ability)    # Rasch 交互偏置
        abil = self.abil_head(m)
        calib = (torch.sigmoid(self.calib_net(torch.cat([m, nd_emb], dim=-1)))[..., 0]
                 if self.calib_net is not None else None)
        if return_hidden:
            return p_next, mastery, abil, calib, m
        return p_next, mastery, abil, calib


class DkvmnKT(nn.Module):
    """非循环替换架构：DKVMN（Dynamic Key-Value Memory Network, Zhang et al. 2017）。

    跨架构验证用：静态 key-memory（每知识点一块概念表征）+ 动态 value-memory（掌握度状态），
    逐位 read/update；输出契约与 HybridKT 完全一致（p_next/mastery/abil），使同一
    training / 固定 split 流程无缝复用，与 GRU/SAKT 在相同数据上直接可比。

    语义对齐：pn[:, t] = P(correct_{t+1}=1 | 历史至 t)——用"第 t 次交互更新记忆"后读取
    下一题（topic[t+1]）的记忆打分，与 collate 的 tgt_next[b,t]=corrects[t+1] 精确对应。
    """
    def __init__(self, num_topics=32, max_len=16, hidden=32, embed_dim=16,
                 dk=16, dv=16, use_pok_calib=False):
        super().__init__()
        self.num_topics = num_topics
        self.max_len = max_len
        self.hidden = hidden
        self.embed_dim = embed_dim
        self.dk = dk                          # key 维（静态 key 记忆）
        self.dv = dv                          # value 维（动态掌握度状态）
        # 静态 key 记忆 M^k（概念 × key）；问题嵌入 q 与之算相关性
        self.key_concepts = nn.Parameter(torch.zeros(num_topics, dk))
        nn.init.normal_(self.key_concepts, std=0.02)
        self.topic_emb = nn.Embedding(num_topics, dk, padding_idx=0)  # q_k（兼作图正则 weight）
        self.correct_emb = nn.Embedding(2, dv, padding_idx=0)
        self.diff_emb = nn.Embedding(6, dv, padding_idx=0)
        self.in_dim = dk + dv + dv
        self.erase_net = nn.Linear(self.in_dim, dv)   # erase（遗忘）
        self.add_net = nn.Linear(self.in_dim, dv)     # add（写入门控）
        # 读取 → f（含下一题难度）
        self.read_net = nn.Sequential(nn.Linear(dv + dv, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
        self.next_head = nn.Linear(hidden, 1)
        self.mastery_head = nn.Linear(hidden, 1)
        self.abil_head = nn.Linear(hidden, 4)
        self.abil_w = 0.3

    @staticmethod
    def corr(q, mk, dk):
        return torch.softmax(q @ mk.t() / (dk ** 0.5), dim=-1)

    def forward(self, topic, correct, diff, latency=None, pres=None, regime_in=None, next_diff=None, return_hidden=False):
        B, T = topic.shape
        v_mem = torch.zeros(B, self.num_topics, self.dv, device=topic.device)
        pn = torch.zeros(B, T, device=topic.device)
        hs = torch.zeros(B, T, self.hidden, device=topic.device)
        for t in range(1, T):
            idx = t - 1
            qk = self.topic_emb(topic[:, idx])
            inter = torch.cat([qk, self.correct_emb(correct[:, idx]), self.diff_emb(diff[:, idx])], dim=-1)
            w = self.corr(qk, self.key_concepts, self.dk)          # [B, num_topics]
            e = torch.sigmoid(self.erase_net(inter))               # [B, dv]
            a = torch.tanh(self.add_net(inter))                    # [B, dv]
            w3 = w.unsqueeze(-1)                                   # [B, num_topics, 1]
            v_mem = v_mem * (1 - w3 * e.unsqueeze(1)) + w3 * a.unsqueeze(1)
            # 用更新后的记忆读取下一题（t）→ 预测 correct[t]
            w_t = self.corr(self.topic_emb(topic[:, t]), self.key_concepts, self.dk)
            r = torch.einsum("bn,bnd->bd", w_t, v_mem)             # [B, dv]
            f = torch.tanh(self.read_net(torch.cat([r, self.diff_emb(diff[:, t])], dim=-1)))
            hs[:, idx] = f
            pn[:, idx] = torch.sigmoid(self.next_head(f)).squeeze(-1)
        mastery = torch.sigmoid(self.mastery_head(hs)).squeeze(-1)
        abil = self.abil_head(hs)
        calib = (torch.sigmoid(self.calib_net(hs)).squeeze(-1)
                 if getattr(self, "calib_net", None) is not None else None)
        if return_hidden:
            return pn, mastery, abil, calib, hs
        return pn, mastery, abil, calib


class GktKT(nn.Module):
    """非循环替换架构：GKT（Graph-based Knowledge Tracing, Nakagawa et al. 2019）。

    跨架构验证用：每个知识点一个隐态节点，学习邻接矩阵做消息传递，门控更新全部节点，
    对查询知识点读隐态打分；输出契约与 HybridKT 一致（p_next/mastery/abil），同一流程可复用。
    消息传递用全连接可学习邻接（本数据知识点无真实图结构，作为"可随数据学习的关系先验"）。

    语义对齐同 DKVMN/SAKT：pn[:, t] = P(correct_{t+1})，用第 t 次交互更新全部节点后读取 topic[t+1]。
    """
    def __init__(self, num_topics=32, max_len=16, hidden=32, embed_dim=16, use_pok_calib=False):
        super().__init__()
        self.num_topics = num_topics
        self.max_len = max_len
        self.hidden = hidden
        self.embed_dim = embed_dim
        # 可学习邻接（topic×topic 消息系数），初始近单位对角+小扰动增量
        self.adj = nn.Parameter(torch.zeros(num_topics, num_topics))
        nn.init.normal_(self.adj, std=0.02)
        self.topic_emb = nn.Embedding(num_topics, hidden, padding_idx=0)  # 兼作图正则 weight
        self.correct_emb = nn.Embedding(2, hidden, padding_idx=0)
        self.diff_emb = nn.Embedding(6, hidden, padding_idx=0)
        gate_in = hidden + hidden + hidden          # [agg, 对错, 难度]
        self.forget = nn.Linear(gate_in, hidden)    # 遗忘门 f
        self.input_g = nn.Linear(gate_in, hidden)   # 输入门 i
        self.cand_emb = nn.Linear(gate_in, hidden)  # 候选更新
        self.next_head = nn.Linear(hidden + hidden, 1)
        self.mastery_head = nn.Linear(hidden, 1)
        self.abil_head = nn.Linear(hidden, 4)
        self.abil_w = 0.3

    def forward(self, topic, correct, diff, latency=None, pres=None, regime_in=None, next_diff=None, return_hidden=False):
        B, T = topic.shape
        h = torch.zeros(B, self.num_topics, self.hidden, device=topic.device)
        pn = torch.zeros(B, T, device=topic.device)
        hs = torch.zeros(B, T, self.hidden, device=topic.device)
        for t in range(1, T):
            idx = t - 1
            agg = torch.einsum("ij,bjd->bid", self.adj, h)          # [B, num_topics, hidden]
            cb = self.correct_emb(correct[:, idx]).unsqueeze(1).expand(B, self.num_topics, self.hidden)
            db = self.diff_emb(diff[:, idx]).unsqueeze(1).expand(B, self.num_topics, self.hidden)
            gate_in = torch.cat([agg, cb, db], dim=-1)              # [B, num_topics, 3*hidden]
            fg = torch.sigmoid(self.forget(gate_in))
            ig = torch.sigmoid(self.input_g(gate_in))
            cand = torch.tanh(self.cand_emb(gate_in))
            h = (1 - ig) * (fg * h) + ig * cand
            # 查询知识点 topic[t] 的隐态（逐 batch 按 topic 取值 → [B, hidden]）
            gidx = topic[:, t].unsqueeze(-1).unsqueeze(-1).expand(B, 1, self.hidden)
            q = torch.gather(h, 1, gidx).squeeze(1)
            pn[:, idx] = torch.sigmoid(self.next_head(
                torch.cat([q, self.diff_emb(diff[:, t])], dim=-1))).squeeze(-1)
            hs[:, idx] = q
        mastery = torch.sigmoid(self.mastery_head(hs)).squeeze(-1)
        abil = self.abil_head(hs)
        calib = (torch.sigmoid(self.calib_net(hs)).squeeze(-1)
                 if getattr(self, "calib_net", None) is not None else None)
        if return_hidden:
            return pn, mastery, abil, calib, hs
        return pn, mastery, abil, calib


class HybridKT(nn.Module):
    def __init__(self, num_topics=32, max_len=16, hidden=32, embed_dim=16, use_bkt_prior=False, use_bkt_feat=False, use_attn=False, use_pok_calib=False, struct_forward=False):
        super().__init__()
        self.num_topics = num_topics
        self.max_len = max_len
        self.hidden = hidden
        self.embed_dim = embed_dim
        self.topic_emb = nn.Embedding(num_topics, embed_dim, padding_idx=0)
        self.diff_emb = nn.Embedding(6, embed_dim, padding_idx=0)   # 0 为 pad，1-5 难度
        self.correct_emb = nn.Embedding(2, embed_dim)
        self.lat_lin = nn.Linear(1, embed_dim)   # 可选作答耗时特征（seq_latency）；无时整段不注入
        self.pres_lin = nn.Linear(1, embed_dim)  # 可选先修掌握度特征（结构入输入）；无时整段不注入
        self.regime_emb = nn.Embedding(4, embed_dim, padding_idx=0)  # 可选真实认知机制特征(know/slip/guess/unk)；无注入
        self.input_proj = nn.Linear(embed_dim, embed_dim)
        # BKT 冷启动：用首知识点嵌入生成初始隐态（把先验 P(L0) 烘焙进去）
        self.init_proj = nn.Linear(embed_dim, hidden)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True)
        # 多任务头
        self.mastery_head = nn.Linear(hidden, 1)
        self.next_head = nn.Linear(hidden + embed_dim, 1)  # 拼接目标难度嵌入
        self.abil_head = nn.Linear(hidden, 4)
        # 校准辅助头（探针④c）：为隔离"读出头表达力"vs"隐态表示信息量"，用两层 MLP 而非单线性，
        # 监督其直接拟合真概率 pOk[t+1]。若 MLP 仍到不了 oracle，则瓶颈在共享隐态 h_t 本身。
        self.calib_net = nn.Sequential(
            nn.Linear(hidden + embed_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        ) if use_pok_calib else None
        self.abil_w = 0.3  # 能力头的损失权重（辅助监督）
        # 可微 BKT 分支（两用途共用同一模块，参数同一套，可任选/可叠加）：
        #   ⑤   use_bkt_prior：后验掌握作为独立分支与 mastery 头加权融合（α 可学习）
        #   ⑤b  use_bkt_feat ：把 BKT 后验 L_after 作为 GRU 输入特征喂入（走 next 头 → 影响 auc）
        self.bkt = DifferentiableBKT() if (use_bkt_prior or use_bkt_feat) else None
        self.fuse_logit = nn.Parameter(torch.tensor(0.0)) if use_bkt_prior else None  # α=sigmoid(0)=0.5 平衡起步
        self.bkt_in_lin = nn.Linear(1, embed_dim) if use_bkt_feat else None
        self.use_bkt_feat = use_bkt_feat
        self.use_attn = use_attn
        self.attn = CausalSelfAttn(hidden) if use_attn else None
        # 动态可学习知识结构（方案A）：struct_W 为每知识点的"结构向量"，
        # 动态邻接 A[i,j] = σ(struct_W[i]·struct_W[j])——结构强弱由模型从作答数据端到端学出，
        # 取代"固定 graph_edges + topic_emb L2"的静态先验。struct_init_W 用于把现有图正则先验
        # 固化进结构（见 struct_adj 的残差正则），train_dkt 支持全架构，本模块不连 forward，仅入 loss。
        # 初始化为小随机非零：struct_adj 在 W=0 处处于鞍点（dA/dW∝(Wi+Wj)=0），
        # 全零初始化会令结构梯度恒为 0、永远无法可学；用小随机扰动脱离原点。
        self.struct_W = nn.Parameter(torch.randn(num_topics, embed_dim) * 0.1)
        self.struct_init_W = None  # 先验结构向量（由外部 graph_edges 构造），非参数仅存托管
        # 动态结构前向（方案A+）：把可学习邻接接入前向，使 struct_W 直接从主任务获得梯度，
        # 而非仅作 loss 正则。struct_feat = struct_lin( A[k,:] @ topic_emb.weight ) 为当前主题 k
        # 的"先修/邻居上下文"聚合；经 tanh(struct_gate) 门控叠加到 GRU 隐态 h（门初值 +1 → 前向梯度
        # 一路有效；需要时模型可自适应把门调回，实现"是否利用结构"的可学习权衡）。再供 mastery/next/abil
        # 三头消费。causality-safe（只聚合当前主题邻居的静态嵌入，不引入未来作答信息）、并行化（无逐时间步扫描）。
        # 默认关闭不改变部署契约。
        self.struct_lin = nn.Linear(embed_dim, hidden) if struct_forward else None
        # gate 初始 +1 → tanh≈0.76，让结构前向梯度从第 0 步就有效（此前 0 → tanh(0)=0 门控关死，
        # struct_W 只受 loss 稀疏/先验压制而塌缩近零，见 2026-09-10 诊断）。配合 --struct-lambda 0 用纯主任务驱动。
        self.struct_gate = nn.Parameter(torch.tensor(1.0)) if struct_forward else None  # sigmoid/tanh 门采点
        self.use_struct_fwd = bool(struct_forward)

    def struct_adj(self):
        """动态可学习邻接 A[K,K]：A[i,j] = σ(struct_W[i]·struct_W[j])。

        结构强弱端到端可学；对角（自环）强制 0，padding id0 行列隔离恒为 0。
        """
        A = torch.sigmoid(torch.matmul(self.struct_W, self.struct_W.t()))
        idm = torch.arange(self.num_topics, device=A.device)
        A = A * (idm[:, None] != idm[None, :]).float()   # 去自环
        A[0, :].zero_()
        A[:, 0].zero_()                                   # padding 隔离
        return A

    def struct_reg(self, graph_edges):
        """结构正则：让可学习邻接 A 与"预测语义嵌入"对齐，同时向 graph_edges 先验靠拢。

        - 语义对齐项（数据信号）：A 逐步向 topic_emb 内积 sigmoid 对齐，使结构学到与
          预测一致的语义关系（端到端信号源，无图边时该项仍可学）。
        - 先验项（graph_edges 先验初始化）：沿先修边对 struct_W 做 L2 拉近，作为先验烘焙。
        - 稀疏项：非边邻接强度抑制，避免退化为全连接稠密结构。
        返回 (reg张量, 对齐张量)，供调用方加权重汇入 total loss。
        """
        A = self.struct_adj()                                       # [K,K] 可学习邻接
        # 语义对齐：topic_emb 内积的 sigmoid 作为目标（邻接语义应与预测嵌入一致）
        emb = self.topic_emb.weight                                # [K, embed_dim]
        emb_prob = torch.sigmoid(torch.matmul(emb, emb.t()))
        idm = torch.arange(self.num_topics, device=emb.device)
        emb_prob = emb_prob * (idm[:, None] != idm[None, :]).float()
        align = (A - emb_prob).pow(2).mean()
        # 稀疏项：非自环、非 padding 处邻接尽量小（结构应稀疏）
        ones = torch.ones_like(A)
        nonedge = ones * (idm[:, None] != idm[None, :]).float()
        nonedge[0, :].zero_(); nonedge[:, 0].zero_()
        sparsity = (A * nonedge).mean()
        # 先验项：沿 graph_edges 边拉近 struct_W（先验烘焙）
        prior = torch.tensor(0.0, device=A.device)
        if graph_edges is not None and graph_edges.numel():
            e0, e1 = graph_edges[:, 0], graph_edges[:, 1]
            prior = (self.struct_W[e0] - self.struct_W[e1]).pow(2).mean()
        return align, sparsity, prior

    def forward(self, topic, correct, diff, latency=None, pres=None, regime_in=None, next_diff=None, return_hidden=False):
        """
        topic/correct/diff: [B,T] long, 左对齐, 0 为 padding。
        latency: [B,T] float（可选）作答耗时特征，已归一化[0,1]；None 时不注入（等价旧模型）。
        pres: [B,T] float（可选）先修掌握度特征（结构入输入），已归一化[0,1]；None 时不注入（等价旧模型）。
        regime_in: [B,T] long（可选）真实认知机制 know=0/slip/guess/unk（探针④d输入特征）；None 时不注入。
        next_diff: [B,T] long → 第 t 位的下一题难度（用于 next-correct 头；缺省用 diff 移位）。
        返回 p_next[B,T], mastery[B,T], abil_logits[B,T,4]；可返回所有隐态。
        """
        B, T = topic.shape
        x = self.topic_emb(topic) + self.diff_emb(diff) + self.correct_emb(correct)
        if latency is not None:
            x = x + self.lat_lin(latency.unsqueeze(-1))
        # 探针④d·机制注入：真实认知机制(know/slip/guess/unk)作为时序特征注入 GRU 输入。
        # 本质是"未压缩的真实机制标签"直给输入——若 gap① 可由机制信号本身复现，则特征不足即被证伪。
        if regime_in is not None:
            x = x + self.regime_emb(regime_in)
        # 论文方向·结构入输入：把"本主题当前最弱先修掌握度"(pres∈[0,1])作为动态时序特征注入 GRU。
        # 与难度/耗时同级参与 input_proj 前的叠加 → 结构信号进入 h → next 头的动态学习路径，
        # 使 GRU 能学到"先修不牢→后续学得差"的跨知识点依赖（区别于仅约束 embedding 静态距离的图正则）。
        if pres is not None:
            x = x + self.pres_lin(pres.unsqueeze(-1))
        # 方向⑤b：把 BKT 后验 L_after（观察到 answer[t] 后的掌握）作为输入特征注入 GRU，
        # 使 BKT 动力学沿 h → next 头进入 p_next 计算（可影响评估 auc），BKT 参数随 next 头梯度训练。
        if self.bkt is not None and self.use_bkt_feat:
            _, L_after = self.bkt(correct, diff, return_after=True)
            x = x + self.bkt_in_lin(L_after.unsqueeze(-1))
        x = torch.tanh(self.input_proj(x))
        # 冷启动：从首知识点嵌入生成 h0（mask 掉全 padding 的批次 → 0 隐态）
        first_emb = self.topic_emb(topic[:, 0])
        h0 = torch.tanh(self.init_proj(first_emb)).unsqueeze(0)  # [1,B,H]
        h, _ = self.gru(x, h0)  # [B,T,H]
        # 方向 D：因果自注意力对 GRU 隐态重新加权组合（历史上下文），掩码防信息泄漏；契约不变。
        if self.attn is not None:
            h = self.attn(h, (topic != 0).float())
        # 动态结构前向（方案A+）：把可学习邻接聚合的"先修/邻居上下文"叠加进 GRU 隐态，
        # 使 struct_W 直接进入主任务(p_next/mastery)梯度路径（而非仅受 loss 正则间接影响）。
        # 门控 tanh(struct_gate) 初始 0 → 结构分支零输出，等价旧模型；随训练门自动开启。
        if self.use_struct_fwd:
            A = self.struct_adj()                                          # [K,K] 可学习邻接
            nbr = A[topic] @ self.topic_emb.weight                         # [B,T,K]×[K,E]→[B,T,E] 邻居语义聚合
            struct_feat = self.struct_lin(nbr)                             # [B,T,H]
            h = h + torch.tanh(self.struct_gate) * struct_feat             # 门控叠加
        mastery = torch.sigmoid(self.mastery_head(h))[..., 0]  # [B,T] GRU 掌握的掌握概率
        # 可微 BKT 先验分支加权融合：α 可学习，α∈(0,1)，融合掉 BKT 后验掌握（强归纳偏置）
        # 注意守卫用 fuse_logit（仅 ⑤ use_bkt_prior 才有），避免仅 ⑤b 时 bkt 存在但无融合权重
        if self.fuse_logit is not None:
            bkt_L = self.bkt(correct, diff)          # [B,T] 可微 BKT 后验掌握概率(作答前)
            alpha = torch.sigmoid(self.fuse_logit)   # 学习融合权重
            mastery = alpha * mastery + (1 - alpha) * bkt_L
        # next 头：难度感知（默认取 diff[t+1]，末尾无则用 diff[t]）
        if next_diff is None:
            nd = torch.cat([diff[:, 1:], diff[:, -1:]], dim=1)
        else:
            nd = next_diff
        nd_emb = self.diff_emb(nd)
        p_next = torch.sigmoid(self.next_head(torch.cat([h, nd_emb], dim=-1)))[..., 0]
        abil = self.abil_head(h)  # [B,T,4]
        # 校准辅助头（探针④c）：与 next 头同输入(h_i, 难度 i+1)，直接回归真概率 pOk[i+1]。
        # 若该头 AUC≈oracle，则证 0.72→0.76 的 gap 是"监督/校准"而非"表示/信息"缺失。
        calib = (torch.sigmoid(self.calib_net(torch.cat([h, nd_emb], dim=-1)))[..., 0]
                 if self.calib_net is not None else None)
        if return_hidden:
            return p_next, mastery, abil, calib, h
        return p_next, mastery, abil, calib

    def export_onnx(self, path, use_latency=False):
        """导出 ONNX。

        use_latency=False（默认，部署路径）：3 输入 seq_topic/seq_correct/seq_diff，不含 latency 分支，
        与现有运行期 core/dkt.js 的 3 输入推理契约完全一致 → 可正常替换部署。
        use_latency=True（实验路径）：4 输入，额外 seq_latency；仅供训练实验，不部署到运行期。
        统一输出"单一文件"（外部权重内联），便于打包分发（onnxruntime-node 无需外部权重文件）。
        """
        import onnx as _onnx
        self.eval()
        B, T = 1, self.max_len
        topic = torch.zeros((B, T), dtype=torch.long)
        correct = torch.zeros((B, T), dtype=torch.long)
        diff = torch.zeros((B, T), dtype=torch.long)
        latency = torch.zeros((B, T), dtype=torch.float32)
        tmp = path + ".tmp"
        # 直接导出模块 forward；校准辅助头默认禁用(calib_net=None → 第4输出为 None，被 ONNX 丢弃)，
        # 故 ONNX 仍是既有 3 输出部署契约。校准辅助头仅训练实验用，不进入部署。
        if use_latency:
            torch.onnx.export(
                self, (topic, correct, diff, latency), tmp,
                input_names=["seq_topic", "seq_correct", "seq_diff", "seq_latency"],
                output_names=["p_next_correct", "mastery_score", "ability_level"],
                opset_version=17, dynamic_axes=None)
        else:
            torch.onnx.export(
                self, (topic, correct, diff), tmp,
                input_names=["seq_topic", "seq_correct", "seq_diff"],
                output_names=["p_next_correct", "mastery_score", "ability_level"],
                opset_version=17, dynamic_axes=None)
        m = _onnx.load(tmp)
        _onnx.save_model(m, path, save_as_external_data=False)
        # 清理临时导出产物（.tmp 及可能的 .tmp.data）
        for cand in (tmp, tmp + ".data"):
            if os.path.exists(cand):
                os.remove(cand)


# --------------------------------------------------------------------------- #
# 3. 训练 / 评测
# --------------------------------------------------------------------------- #

def train(model, rows, device, args, max_len):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = len(rows)
    use_lat = bool(getattr(args, "use_latency", False))
    use_w = bool(getattr(args, "use_weight", False))
    use_pres = bool(getattr(args, "use_pres", False))
    use_calib = bool(getattr(args, "use_pok_calib", False))
    use_regime = bool(getattr(args, "use_regime", False))
    X, Y, D, LAT, PRE, M, TN, TA, TMA, TW, TPO, REG = collate(rows, model.num_topics, max_len, torch.device("cpu"))
    if not use_w:
        TW = torch.ones_like(TW)  # 未加权口径：退回普通 BCE（数据仍含 weights，仅不用）
    # 移到 GPU/cpu（此处统一用 device）
    CACHE = (X, Y, D, LAT, PRE, M, TN, TA, TMA, TW, TPO, REG)
    X, Y, D, LAT, PRE, M, TN, TA, TMA, TW, TPO, REG = [t.to(device) for t in CACHE]
    # 先修图正则（论文方向 P0+）：读整数边 [src,dst]，对 topic_emb 相邻嵌入做 L2 拉近，
    #    让图谱结构进入模型表征（而非只改数据分布）。无图(req edges 空)时该项为 0，完全不影响旧训练。
    graph_edges = None
    graph_lambda = float(getattr(args, "graph_lambda", 1e-3) or 0)
    # 动态可学习知识结构（方案A）：--struct-lambda>0 时启用可学习邻接 struct_W。
    # 复用 graph_edges 读出的先验边作为结构先验烘焙；lambda=0（默认）完全关闭，向后兼容。
    struct_lambda = float(getattr(args, "struct_lambda", 0) or 0)
    if getattr(args, "graph_edges", None):
        try:
            import json as _json
            with open(args.graph_edges, "r", encoding="utf-8") as _f:
                g = _json.load(_f)
            g = [[int(a), int(b)] for a, b in g if int(a) != int(b)]
            if g:
                ge = torch.tensor(g, dtype=torch.long, device=device)
                # 排除 padding id 0（topic_emb 的 padding 行不参与正则，保持其恒为 0 的意义）
                ge = ge[(ge[:, 0] > 0) & (ge[:, 1] > 0)]
                if ge.numel():
                    graph_edges = ge
        except Exception:
            graph_edges = None
    valid_ratio = 0.15
    nv = max(1, int(n * valid_ratio))
    best = float("inf")
    best_epoch = 0
    patience = max(0, int(getattr(args, "patience", 8) or 0))
    # 课程学习（方向⑦-6）：增量难度教学。计算每训练行的行均难度（真实位置 diff>0），
    #   随 epoch 从易到难抬升阈值，仅取难度 <= 阈值的行参与当轮训练。只影响样本呈现顺序，
    #   不改模型/验证集，保留 shuffle 以隔离课程与批次噪声。curriculum=0 即回退全量训练。
    cur = float(getattr(args, "curriculum", 0) or 0)
    cur_min = float(getattr(args, "curriculum_min", 1.0) or 1.0)
    row_diff = torch.full((n,), float("nan"), dtype=torch.float32, device=device)
    if cur > 0:
        for i in range(n):
            d = D[i][(D[i] > 0) & (M[i] > 0)]
            row_diff[i] = d.float().mean().item() if d.numel() else cur_min
        cur_pool = list(range(n - nv))  # 全量训练行池（课程仍是全池子集）
    import time as _time
    _t0 = _time.time()          # 训练总起点（wall clock）
    for epoch in range(args.epochs):
        _te = _time.time()      # 本 epoch 起点
        model.train()
        # 课程阈值：p=epoch/epochs，阈值=cur_min + (5-cur_min)*(p^tau)，tau=cur（越大越先集中在最易行）
        if cur > 0:
            p = (epoch + 1) / max(args.epochs, 1)
            tau = max(cur, 1e-3)
            thr = cur_min + (5 - cur_min) * (p ** (1.0 / tau))
            in_cur = [i for i in cur_pool if row_diff[i].item() <= thr + 1e-6]
            if not in_cur:
                in_cur = list(cur_pool)
            idx = in_cur
        else:
            idx = list(range(n - nv))
        random.shuffle(idx)
        total = 0.0
        nstep = 0
        for s in range(0, len(idx), args.batch):
            bi = idx[s:s + args.batch]
            bX, bY, bD, bLAT, bPRE, bM, bTN, bTA, bTMA, bTW, bTPO, bREG = X[bi], Y[bi], D[bi], LAT[bi], PRE[bi], M[bi], TN[bi], TA[bi], TMA[bi], TW[bi], TPO[bi], REG[bi]
            call = (bX, bY, bD, bLAT if use_lat else None, bPRE if use_pres else None, bREG if use_regime else None)
            pn, ms, abil, cal = model(*call)
            loss = 0.0
            # next 监督（先掩码再 BCE，避免 padding=-1 触发校验错误）
            # use_weight=true：按 bTW(蒙对/失误降权、掌握/不懂满权) 对每题样本加权，降 slip/guess 噪声。
            m_next = (bTN >= 0)
            if m_next.any():
                w = bTW[m_next].clamp(1e-2, 1e2) if m_next.any() else 1.0
                loss = loss + torch.nn.functional.binary_cross_entropy(
                    pn[m_next].float().clamp(1e-6, 1 - 1e-6),
                    bTN[m_next].float(), weight=w)
            # mastery 监督：真值掌握度回归（校准 mastery 头，避免失准）
            m_mast = torch.isfinite(bTMA)
            if m_mast.any():
                loss = loss + torch.nn.functional.mse_loss(ms[m_mast], bTMA[m_mast])
            # ability 辅助监督
            m_ab = (bTA >= 0)
            if m_ab.any():
                loss = loss + model.abil_w * torch.nn.functional.cross_entropy(
                    abil[m_ab], bTA[m_ab])
            # 校准辅助头监督（探针④c）：对共享隐态施加"直接拟合真概率 pOk[t+1]"的校准信号，
            # 检验 gap① 是否可由校准监督收敛到 oracle。
            if use_calib and cal is not None:
                m_pk = torch.isfinite(bTPO)
                if m_pk.any():
                    loss = loss + torch.nn.functional.binary_cross_entropy(
                        cal[m_pk].clamp(1e-6, 1 - 1e-6), bTPO[m_pk].clamp(0, 1))
            # 先修图正则（结构进模型）：相邻 topic 嵌入 L2 拉近。batch 无关，直接对整表算一次。
            if graph_edges is not None and graph_lambda > 0:
                e = model.topic_emb.weight                      # [num_topics, embed_dim]
                reg = (e[graph_edges[:, 0]] - e[graph_edges[:, 1]]).pow(2).mean()
                loss = loss + graph_lambda * reg
            # 动态可学习知识结构（方案A）：--struct-lambda>0 时启用可学习邻接。
            # 结构正则 = α·语义对齐 + β·稀疏 + γ·先验；均走 model.struct_reg，batch 无关，整表算一次。
            if struct_lambda > 0:
                align, sparsity, prior = model.struct_reg(graph_edges)
                # 三项权重：对齐为主(1.0)，稀疏抑制 0.5，先验烘焙 0.5（无图边时 prior 恒 0）
                loss = loss + struct_lambda * (align + 0.5 * sparsity + 0.5 * prior)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            nstep += 1
        # 验证
        model.eval()
        vpn, vms, vabil, _vc = model(X[n - nv:], Y[n - nv:], D[n - nv:], LAT[n - nv:] if use_lat else None, PRE[n - nv:] if use_pres else None, REG[n - nv:] if use_regime else None)
        vmean = None
        for w, tgt, mask, key in (
            (vpn, TN[n - nv:], M[n - nv:], "next"),
        ):
            m = (tgt >= 0) & mask.bool()
            if m.any():
                vmean = torch.nn.functional.binary_cross_entropy(
                    torch.clamp(vpn[m], 1e-6, 1 - 1e-6), tgt[m].float()).item()
        if vmean is not None and vmean < best - 1e-4:
            best = vmean
            best_epoch = epoch
            torch.save(model.state_dict(), args.checkpoint)
        if args.epochs <= 40 or epoch % 10 == 0:
            _et = _time.time() - _te
            _mt = _time.time() - _t0
            print(f"epoch {epoch+1}/{args.epochs} loss={total/max(nstep,1):.4f} val_next_bce={vmean if vmean is not None else float('nan'):.4f} [ec={_et:.1f}s tot={_mt:.1f}s]")
        # 早停：连续 patience 轮 val 未改善即停止，抑制 GRU 拟合噪声
        if patience and epoch - best_epoch >= patience:
            print(f"early stop at epoch {epoch+1} (best val {best:.4f} @ epoch {best_epoch+1})")
            break
    print(f"best val next BCE: {best:.4f}")


def split_eval(pn, gt, clip=True):
    """ACC / AUC / Brier，与 Node core 的 evaluate() 口径一致。"""
    p = np.clip(np.asarray(pn, dtype=np.float64), 0, 1) if clip else np.asarray(pn, dtype=np.float64)
    y = np.asarray(gt, dtype=np.float64)
    m = y >= 0
    p, y = p[m], y[m]
    if len(y) == 0:
        return {"acc": 0.0, "auc": None, "brier": None, "n": 0}
    acc = float(np.mean((p >= 0.5).astype(int) == y.astype(int)))
    brier = float(np.mean((p - y) ** 2))
    pos = p[y == 1]
    neg = p[y == 0]
    auc = None
    if len(pos) > 0 and len(neg) > 0:
        ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
        rp = ranks[: len(pos)].sum()
        auc = float(rp / (len(pos) * len(neg)) - (len(pos) + 1) / (2 * len(neg)))
        auc = float(np.clip(auc, 0, 1))
    return {"acc": round(acc, 4), "auc": None if auc is None else round(auc, 4),
            "brier": round(brier, 4), "n": int(len(y))}


def bkt_next_sequence(corrects, diffs):
    """纯 BKT 逐位 next 概率（与 core/knowledgeTracker 同步）：掌握概率后验 + 下一题难度。"""
    prior, learn = 0.35, 0.12
    L = prior
    p_nexts = []
    for i in range(len(corrects) - 1):
        cor = corrects[i]
        d = diffs[i + 1] if i + 1 < len(diffs) else 3
        slip = min(0.5, max(0.01, 0.10 + (d - 3) * 0.05))
        guess = min(0.5, max(0.01, 0.25 - (d - 3) * 0.05))
        p_corr_now = L * (1 - slip) + (1 - L) * guess
        if cor == 1:
            num = L * (1 - slip)
            L = num / (num + (1 - L) * guess)
        else:
            num = L * slip
            L = num / (num + (1 - L) * (1 - guess))
        L = L + (1 - L) * learn
        # 对 t 步，next 概率用 i+1 难度
        slip_nt = min(0.5, max(0.01, 0.10 + (d - 3) * 0.05))
        guess_nt = min(0.5, max(0.01, 0.25 - (d - 3) * 0.05))
        p_nexts.append(L * (1 - slip_nt) + (1 - L) * guess_nt)
    return p_nexts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="JSONL 数据（可选；缺省合成）")
    ap.add_argument("--num-topics", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--embed-dim", type=int, default=16)
    ap.add_argument("--arch", choices=["gru", "sakt", "saint", "akt", "dkvmn", "gkt"], default="gru",
                    help="gru=GRU 族（HybridKT，默认）；sakt=单塔纯因果多头自注意力；saint=双塔分离注意（题目/作答各自编码后融合）；akt=因果自注意力+Rasch 型难度系数+跨注意力；三者均为非循环替换架构，用于跨架构验证数据上限")
    ap.add_argument("--heads", type=int, default=4, help="SAKT 注意力头数")
    ap.add_argument("--layers", type=int, default=1, help="SAKT 自注意力+FFN 堆叠层数")
    ap.add_argument("--no-pos", action="store_true", help="SAKT 消融：关闭可学习位置编码（验证顺序信息是否必需）")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8, help="早停：连续 N 轮 val BCE 未改善即停止（0=禁用）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="training/data/ai_train.split.json")
    ap.add_argument("--split-mode", choices=["row", "fixed"], default="row",
                    help="row=每次随机行级留出并写 split 文件（默认）；fixed=读既有 split 文件的 train/test 索引复用，不改动（用于多 seed 在同一测试集上公平择优）")
    ap.add_argument("--checkpoint", default="training/data/dkt_best.pt")
    ap.add_argument("--out", default="app/server/models/dkt.onnx")
    ap.add_argument("--use-latency", action="store_true", help="启用作答耗时(seq_latency)特征；关闭时导出3输入模型(部署兼容)")
    ap.add_argument("--use-pres", action="store_true",
                    help="论文方向·结构入输入：启用作答时的先修掌握度(seq_pres)动态特征进 GRU 输入；\
数据须含 pres 字段(由 dktDataBuilder opts.pres=true 生成)。仅训练实验用，导出仍为3输入模型(部署兼容)。")
    ap.add_argument("--use-regime", action="store_true",
                    help="探针④d：把真实认知机制(know/slip/guess/unk)作为时序输入特征注入 GRU。\
数据须含 regimes 字段(由 dktDataBuilder opts.regimeInput=true 生成)。仅研究探针用，不进入部署(导出仍3输入)。")
    ap.add_argument("--use-weight", action="store_true", help="对 next 监督做样本加权；数据须含 weights 字段（方向B），关闭则回退普通 BCE")
    ap.add_argument("--use-bkt-prior", action="store_true", help="方向⑤：可微 BKT 后验与 GRU mastery 头加权融合（不改 next 头/auc）；关闭即纯 DKT")
    ap.add_argument("--use-bkt-feat", action="store_true", help="方向⑤b：把 BKT 后验 L_after 作为 GRU 输入特征喂入（经 next 头影响 auc）；可与 --use-bkt-prior 叠加")
    ap.add_argument("--use-pok-calib", action="store_true",
                    help="探针④c：加校准辅助头，直接监督其拟合真概率 pOk[t+1]（数据须含 pOk 列）。\
仅训练实验用，不改变部署 ONNX 的 3 输出契约。")
    ap.add_argument("--use-attn", action="store_true", help="方向D：GRU 后加因果自注意力编码历史上下文（契约仍3入3出）；关闭即纯GRU")
    ap.add_argument("--curriculum", type=float, default=0.0,
                    help="课程学习：按 epoch 递增的行均难度阈值（从易到难）。0=关闭；1=线性 1→5；>1=温和曲线(阈值=1+4*(p^tau)，tau=该值)。")
    ap.add_argument("--curriculum-min", type=float, default=1.0, help="课程学习起始难度阈值（默认 1，即最早只喂最易行）")
    ap.add_argument("--graph-edges", default=None, help="先修图正则边文件（JSON：[[src_id,dst_id],...]，src=先修→dst=本主题）。\
供对 topic_emb 施加先修结构正则（论文方向 P0+，让图谱结构进模型，ONNX 契约定长不变）")
    ap.add_argument("--graph-lambda", type=float, default=1e-3, help="先修图正则权重（对相邻 topic 嵌入做 L2 拉近）")

    ap.add_argument("--struct-lambda", type=float, default=0.0,
                    help="动态可学习知识结构（方案A）权重：>0 时启用可学习邻接 struct_W（A[i,j]=σ(struct_W[i]·struct_W[j])，\
与预测语义嵌入对齐 + 稀疏抑制 + graph_edges 先验烘焙）。0=关闭（默认），完全向后兼容。导出的 ONNX 仍为 3 输入契约")
    ap.add_argument("--struct-forward", action="store_true",
                    help="方案A+：把可学习邻接接入前向——struct_feat=struct_lin(A[k,:]@topic_emb) 经 tanh(struct_gate) \
门控叠加进 GRU 隐态，使 struct_W 直接从主任务获得梯度（而非仅 loss 正则）。需配合 --struct-lambda>0 启用结构。\
导出仍为 3 输入契约（struct_W 为内部参数）")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_jsonl(args.data) if args.data else gen_synthetic(
        num_topics=args.num_topics, max_len=args.max_len, seed=args.seed)

    # 训练集 / 测试集切分（行级随机留出 15%）。
    # 知识点在本数据中是相互独立的 hash 桶，无共亨规律可迁移，"按知识点留出"会让
    # 模型面对从未见过也无共性的知识点 → auc 天然≈随机(≈0.5)，无法反映真实能力。
    # 实际部署场景是"对已知知识点预测新序列"→ 用行级随机留出最符合用途；
    # 且测试划分写入 --split 文件，Node 端部署后测试复用同一 test 集，保证口径一致。
    n = len(rows)
    split_mode = args.split_mode or "row"
    if split_mode == "fixed" and os.path.exists(args.split):
        # 复用既有 split 索引 → 多 seed 在同一测试集上对比，择优基线一致
        with open(args.split, "r", encoding="utf-8") as f:
            spec = json.load(f)
        train_idx, test_idx = spec["train_index"], spec["test_index"]
    else:
        idx = list(range(n))
        random.shuffle(idx)
        split = int(n * 0.85)
        train_idx, test_idx = idx[:split], idx[split:]
        with open(args.split, "w", encoding="utf-8") as f:
            json.dump({"split_mode": "row-split", "test_index": test_idx,
                       "train_index": train_idx, "test_topics": []}, f)
    train_rows = [rows[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]

    # Embedding 固定为声明的 num_topics（与桥/运行时一致），而非按数据最大桶自动缩容。
    # 理由：stableTopicId(name, num_topics) 会把任意新知识点哈希进 [1, num_topics-1]，
    # 若模型按 "数据中出现过的最大桶+1" 建表，运行时遇到训练里没出现过的桶号会越界。
    # 固定 num_topics 后，空桶照样分配 Embedding 行，推理永不越界（未训练桶只是学习稀疏）。
    model = HybridKT(num_topics=args.num_topics, max_len=args.max_len,
                     hidden=args.hidden, embed_dim=args.embed_dim,
                     use_bkt_prior=args.use_bkt_prior, use_bkt_feat=args.use_bkt_feat,
                     use_attn=args.use_attn, use_pok_calib=args.use_pok_calib,
                     struct_forward=args.struct_forward)
    if args.arch == "sakt":
        model = SaktKT(num_topics=args.num_topics, max_len=args.max_len,
                       hidden=args.hidden, embed_dim=args.embed_dim,
                       n_heads=args.heads, n_layers=args.layers,
                       use_pok_calib=args.use_pok_calib, use_pos=not args.no_pos)
    elif args.arch == "saint":
        model = SaintKT(num_topics=args.num_topics, max_len=args.max_len,
                        hidden=args.hidden, embed_dim=args.embed_dim,
                        n_heads=args.heads, n_layers=args.layers,
                        use_pok_calib=args.use_pok_calib, use_pos=not args.no_pos)
    elif args.arch == "akt":
        model = AktKT(num_topics=args.num_topics, max_len=args.max_len,
                      hidden=args.hidden, embed_dim=args.embed_dim,
                      n_heads=args.heads, n_layers=args.layers,
                      use_pok_calib=args.use_pok_calib, use_pos=not args.no_pos)
    elif args.arch == "dkvmn":
        # dk（key 维）/ dv（value 维）沿用隐藏规模；固定 split 同口径与 GRU/SAKT 可比
        model = DkvmnKT(num_topics=args.num_topics, max_len=args.max_len,
                        hidden=args.hidden, embed_dim=args.embed_dim,
                        dk=args.embed_dim, dv=args.embed_dim,
                        use_pok_calib=args.use_pok_calib)
    elif args.arch == "gkt":
        model = GktKT(num_topics=args.num_topics, max_len=args.max_len,
                      hidden=args.hidden, embed_dim=args.embed_dim,
                      use_pok_calib=args.use_pok_calib)
    train(model, train_rows, device, args, args.max_len)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval().to("cpu")
    if not args.use_pok_calib and args.arch == "gru":
        model.export_onnx(args.out, use_latency=args.use_latency)
        print(f"exported: {os.path.abspath(args.out)}")
    else:
        print("skip onnx export（校准辅助头/SAKT 探针不进入部署路径） checkpoint=", os.path.abspath(args.checkpoint))

    # 评测：DKT vs BKT（测试集上逐位 next 概率）
    T = args.max_len
    topic, correct, diff, lat, pre, mask, tgt_next, tgt_abil, tgt_mast, _tw, tgt_pok, _reg = collate(test_rows, model.num_topics, T, torch.device("cpu"))
    with torch.no_grad():
        pn, ms, _, cal = model(topic, correct, diff,
                               latency=(lat if args.use_latency else None),
                               pres=(pre if args.use_pres else None),
                               regime_in=(_reg if args.use_regime else None))
    dkt_p_next, dkt_gt = [], []
    cal_p_next = []  # 校准辅助头输出（拟合真概率 pOk[t+1]）
    for b in range(len(test_rows)):
        for t in range(T):
            if mask[b, t].item() and int(tgt_next[b, t]) >= 0:
                dkt_p_next.append(pn[b, t].item())
                dkt_gt.append(int(tgt_next[b, t]))
                if cal is not None:
                    cal_p_next.append(cal[b, t].item())
    res = split_eval(dkt_p_next, dkt_gt)
    if cal is not None and cal_p_next:
        res_calib = split_eval(cal_p_next, dkt_gt)
        print("CALIB_HEAD(test):", json.dumps(res_calib))
        print("  → 校准头 AUC 对比 next-head:", res_calib["auc"], "vs", res["auc"], "；oracle=0.7619")

    bkt_p_next = []
    bkt_gt = []
    for r in test_rows:
        bkt_p_next.extend(bkt_next_sequence(r["corrects"], r["diffs"]))
        bkt_gt.extend(r["next"][1:])
    res_bkt = split_eval(bkt_p_next, bkt_gt)

    print("=== DKT(hybrid) vs BKT 评测门禁 (test) ===")
    print("DKT:", json.dumps(res))
    print("BKT:", json.dumps(res_bkt))


def rows_max_topic(rows):
    return max(r["topic"] for r in rows) if rows else 0


if __name__ == "__main__":
    main()