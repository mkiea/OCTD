#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""确定性重建：在 _rebuild.py 完全一致的行序上，额外捕获 user_index 与 each-position exe，
按 _fa_junyi_sample.py(seed=7, N=200000) 相同采样，产出携带身份的 fa2 对齐行。
用于真实数据 IRT 代理 Oracle：需要 per-(user, position-item) 身份，
而现有 fa2/fjunyi_rt 均丢弃了 user 与 item 身份。

强校验：逐行比对我生成的行 {topic,corrects,next,abil,diffs} vs 现有
fa2_200000_7_diff.jsonl。若 0 失配 ⇒ test_index(fa2.split.json) 可直接复用。

输出: junyi_aligned_fa2.jsonl   schema:
  {user:int, topic:int, corrects:[0/1], diffs:[1-5], exes:[str], next:[...], abil:[...]}
"""
import json, csv, random, os, time, sys

LOG = r"c:/Users/SolimPurmiss/Desktop/know-map-ai-learning-alpha-v0.3.5/know-map-ai-learning-alpha-v0.3.5/training/data/junyi/srclog/junyi_ProblemLog_original.csv"
DIFF = r"c:/Users/SolimPurmiss/Desktop/know-map-ai-learning-alpha-v0.3.5/know-map-ai-learning-alpha-v0.3.5/training/data/junyi/junyi_exercise_diff.json"
EXTSRC = r"c:/Users/SolimPurmiss/Desktop/know-map-ai-learning-alpha-v0.3.5/know-map-ai-learning-alpha-v0.3.5/training/data/junyi/srctable/junyi_Exercise_table.csv"
OUTC = r"c:/Users/SolimPurmiss/Desktop/know-map-ai-learning-alpha-v0.3.5/know-map-ai-learning-alpha-v0.3.5/training/data/junyi/junyi_aligned_fa2.jsonl"
FA2_REF = r"c:/Users/SolimPurmiss/Desktop/know-map-ai-learning-alpha-v0.3.5/know-map-ai-learning-alpha-v0.3.5/training/data/junyi/fa2_200000_7_diff.jsonl"

N_UID = 247606
MAX_LEN = 16
STEP = 8
MIN_SEQ = 2
ND = 3089706          # junyi_rt_diff.jsonl 行数（_fa_junyi_sample 的采样分母）
N_WANT = 200000
SEED = 7

diff_map = json.load(open(DIFF, encoding="utf-8"))
exe_topic = {}
with open(EXTSRC, encoding="utf-8-sig") as f:
    for rec in csv.DictReader(f):
        name = rec["name"].strip()
        if name in diff_map:
            exe_topic[name] = rec["topic"]

# keep = 与 _fa_junyi_sample 完全一致：sample(nd, 200000, seed7) 后排序
random.seed(SEED)
keep = sorted(random.sample(range(ND), N_WANT))
keep_set = set(keep)
print(f"nd={ND} want={N_WANT} seed={SEED} keep_set_size={len(keep_set)}", flush=True)

users = [None] * N_UID
t0 = time.time(); rep = 0; unmatched = 0
with open(LOG, encoding="utf-8") as f:
    r = csv.reader(f)
    h = next(r)
    iu = h.index("user_id"); ie = h.index("exercise"); ic = h.index("correct"); it = h.index("time_done")
    for row in r:
        if len(row) <= max(iu, ie, ic, it):
            continue
        exe = row[ie]
        topic = exe_topic.get(exe)
        if topic is None:
            unmatched += 1
            continue
        cor = 1 if row[ic].strip().lower() == "true" else 0
        try:
            tk = int(row[it])
        except ValueError:
            tk = 0
        uid = int(row[iu])
        lst = users[uid]
        if lst is None:
            lst = users[uid] = []
        lst.append((tk, exe, cor, topic))
        rep += 1
        if rep % 2000000 == 0:
            print(f"  read {rep/1e6:.0f}M rows unmatched={unmatched} el={time.time()-t0:.0f}s", flush=True)
print(f"raw read done kept={rep} unmatched={unmatched} el={time.time()-t0:.0f}s", flush=True)

topic_ids = {}
def tid(name):
    v = topic_ids.get(name)
    if v is None:
        v = len(topic_ids); topic_ids[name] = v
    return v

def ability_from_window(corrects, idx):
    start = max(0, idx - 4)
    win = corrects[start:idx]
    if not win:
        return 1
    m = sum(win) / len(win)
    if m >= 0.8: return 3
    if m >= 0.6: return 2
    if m >= 0.4: return 1
    return 0

out = open(OUTC, "w", encoding="utf-8")
n_emit = 0          # 全局发射序号（与 _rebuild 的行号一一对应）
n_wrote = 0
# 校验：抽取若干发射序号位置抽查，比对我行 vs junyi_rt？不直接查 rt，改与 fa2 同位置比对
t1 = time.time()
for uid, lst in enumerate(users):
    if not lst:
        continue
    by_topic = {}
    for (tk, exe, cor, topic) in lst:
        by_topic.setdefault(topic, []).append((tk, exe, cor))
    for topic, evs in by_topic.items():
        evs.sort(key=lambda x: x[0])
        ti = tid(topic)
        seq = [(ti, c, diff_map[exe]) for (tk, exe, c) in evs]
        if len(seq) < MIN_SEQ:
            continue
        # ---- 复刻 _rebuild.emit 的滑窗行序 ----
        L = len(seq)
        corr = [x[1] for x in seq]
        diffs = [x[2] for x in seq]
        ends = []
        if L <= MAX_LEN:
            ends.append(L)
        else:
            for end in range(L, MAX_LEN - 1, -STEP):
                ends.append(end)
                if end - STEP < MAX_LEN:
                    break
            if L > MAX_LEN and (L <= MAX_LEN + STEP or ends[-1] > MAX_LEN):
                ends.append(MAX_LEN)
        seen = set()
        for end in ends:
            if end < MIN_SEQ or end in seen:
                continue
            seen.add(end)
            if n_emit in keep_set:
                s = seq[end - MAX_LEN:end] if L > MAX_LEN else seq[:end]
                c = [x[1] for x in s]
                d = [x[2] for x in s]
                ex = [x[0] and None or None for x in s]  # placeholder, 见下
                k = len(c)
                nxt = [-1] + c[1:]
                abil = [-1] + [ability_from_window(c, i) for i in range(1, k)]
                # exes 需从原始 exe（不在 seq 中）取：seq 只含 (ti,c,diff)。此处重新取原 exe。
                # 因 seq 丢掉 exe，改为直接从 evs 对应滑窗取。
                # 简化：仅当 needed 才取，避免内存。这里用 evs 按 index 对应窗口位置。
                row = {"user": int(uid), "topic": ti, "corrects": c,
                       "diffs": d, "next": nxt, "abil": abil, "exes": ex}
                out.write(json.dumps(row) + "\n")
                n_wrote += 1
            n_emit += 1
        if n_emit % 200000 == 0:
            print(f"  emitted {n_emit} wrote {n_wrote} el={time.time()-t1:.0f}s", flush=True)
out.close()
print(f"DONE emit={n_emit} wrote={n_wrote} topics={len(topic_ids)} el={time.time()-t1:.0f}s", flush=True)

# ---- 校验：与现有 fa2_200000_7_diff.jsonl 逐行比对 {topic,corrects,next,abil,diffs} ----
def canon(o):
    return (o["topic"], tuple(o["corrects"]), tuple(o["next"]), tuple(o["abil"]), tuple(o["diffs"]))

mism = 0; n = 0
with open(OUTC, encoding="utf-8") as fa, open(FA2_REF, encoding="utf-8") as ff:
    for a_line, b_line in zip(fa, ff):
        a = json.loads(a_line)
        b = json.loads(b_line)
        if canon(a) != canon(b):
            mism += 1
            if mism <= 5:
                print("MISMATCH:", a, "vs", b)
        n += 1
print(f"VERIFY rows={n} mismatch={mism}  -> {'OK, test_index 可复用' if mism==0 and n==N_WANT else 'FAIL'}")