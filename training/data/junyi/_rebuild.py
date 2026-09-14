#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重建带 exercise 粒度的 Junyi 序列，输出「同序列双版本」对照数据。

分组：(user_id, topic) 每对构成一条原始作答轨迹，按 time_done 排序。
每条交互挂其 exercise 的真实难度档(1-5)；topic 用 Exercise_table 的 topic 名哈希为整数 id。
窗口：与 train_dkt 兼容 (max_len=16, 自尾部滑窗步长=8, min_seq_len=2)。

输出两个版本（同分组同序列，仅 diffs 不同）：
  junyi_rt_diff.jsonl        diffs = 真实难度档
  junyi_rt_nodiff.jsonl      diffs = 恒 3（无难度对照）
schema: {topic:int(0..N), corrects:[0/1], diffs:[1-5], next:[0/1 or -1], abil:[0-3 or -1]}
"""
import json, csv, os, time

LOG = r"c:/Users/SolimPurmiss/Desktop/know-map-ai-learning-alpha-v0.3.5/know-map-ai-learning-alpha-v0.3.5/training/data/junyi/srclog/junyi_ProblemLog_original.csv"
DIFF = r"c:/Users/SolimPurmiss/Desktop/know-map-ai-learning-alpha-v0.3.5/know-map-ai-learning-alpha-v0.3.5/training/data/junyi/junyi_exercise_diff.json"
EXTSRC = r"c:/Users/SolimPurmiss/Desktop/know-map-ai-learning-alpha-v0.3.5/know-map-ai-learning-alpha-v0.3.5/training/data/junyi/srctable/junyi_Exercise_table.csv"
OUTD = r"c:/Users/SolimPurmiss/Desktop/know-map-ai-learning-alpha-v0.3.5/know-map-ai-learning-alpha-v0.3.5/training/data/junyi"

MAX_LEN = 16
STEP = 8
MIN_SEQ = 2

diff_map = json.load(open(DIFF, encoding="utf-8"))

# execute -> topic 名 ; 仅覆盖 Exercise_table 内且有难度的 exercise
exe_topic = {}
with open(EXTSRC, encoding="utf-8-sig") as f:
    r = csv.DictReader(f)
    for rec in r:
        name = rec["name"].strip()
        if name in diff_map:
            exe_topic[name] = rec["topic"]

# 固定 uid 数量 247606；用数组按行存每用户事件列表 (time, exe, correct)
N_UID = 247606
users = [None] * N_UID

t0 = time.time()
rows_seen = 0
unmatched = 0
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
            continue  # 该 exercise 无难度映射，剔除
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
        rows_seen += 1
        if rows_seen % 2000000 == 0:
            print(f"  read {rows_seen/1e6:.0f}M rows, unmatched={unmatched}, elapsed={time.time()-t0:.0f}s", flush=True)

print(f"read done. kept={rows_seen} unmatched(no-diff)={unmatched} elapsed={time.time()-t0:.0f}s", flush=True)

# topic 名 -> 稳定整数 id
topic_ids = {}
def tid(name):
    if name not in topic_ids:
        topic_ids[name] = len(topic_ids)
    return topic_ids[name]

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

def emit(seq, f_diff, f_nodiff):
    """seq: list of (topic_id, correct, diff). 输出该轨迹的滑窗双版本行。"""
    L = len(seq)
    corr = [x[1] for x in seq]
    diffs = [x[2] for x in seq]
    topicid = seq[0][0]
    # 滑窗
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
        s = seq[end - MAX_LEN:end] if L > MAX_LEN else seq[:end]
        c = [x[1] for x in s]
        d = [x[2] for x in s]
        k = len(c)
        nxt = [-1] + c[1:]
        abil = [-1] + [ability_from_window(c, i) for i in range(1, k)]
        row = {"topic": topicid, "corrects": c, "next": nxt, "abil": abil}
        rdiff = dict(row); rdiff["diffs"] = d
        rnod = dict(row); rnod["diffs"] = [3] * k
        f_diff.write(json.dumps(rdiff) + "\n")
        f_nodiff.write(json.dumps(rnod) + "\n")

f_diff = open(os.path.join(OUTD, "junyi_rt_diff.jsonl"), "w", encoding="utf-8")
f_nodiff = open(os.path.join(OUTD, "junyi_rt_nodiff.jsonl"), "w", encoding="utf-8")

n_traj = 0
n_row_diff = 0
n_row_nodiff = 0
t1 = time.time()
for uid, lst in enumerate(users):
    if not lst:
        continue
    # 组内按 (topic) 再按时间分组
    by_topic = {}
    for (tk, exe, cor, topic) in lst:
        by_topic.setdefault(topic, []).append((tk, exe, cor))
    for topic, evs in by_topic.items():
        evs.sort(key=lambda x: x[0])  # 按时间排序
        ti = tid(topic)
        seq = [(ti, c, diff_map[exe]) for (tk, exe, c) in evs]
        if len(seq) < MIN_SEQ:
            continue
        n_traj += 1
        emit(seq, f_diff, f_nodiff)
        n_row_diff += 1
        n_row_nodiff += 1
    if uid % 50000 == 0:
        print(f"  users {uid}/{N_UID} traj {n_traj} elapsed {time.time()-t1:.0f}s", flush=True)

f_diff.close(); f_nodiff.close()
print(f"DONE traj={n_traj} rows(each) ~ {n_row_diff} topics={len(topic_ids)} elapsed={time.time()-t1:.0f}s", flush=True)
sizes = {f: os.path.getsize(os.path.join(OUTD, f)) for f in ["junyi_rt_diff.jsonl", "junyi_rt_nodiff.jsonl"]}
print("output sizes:", sizes, flush=True)