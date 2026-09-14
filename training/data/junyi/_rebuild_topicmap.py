#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重建 junyi_rt_diff 的 topic 名 -> 整数 id 映射（复刻 _rebuild.py 的 tid() 分配顺序）

_rebuild.py 中 tid() 在遍历 users(按 uid) 时，对每个 user 的 by_topic(dict) 键(插入序=topic名在
该用户内首次出现序)分配 0,1,2...。要拿到与数据完全一致的 id，必须按相同顺序扫 LOG。
本脚本只存 name->id 映射（不产出数据），供对齐先修边到 40-topic 空间。
"""
import json, csv

LOG = r"c:/Users/SolimPurmiss/Desktop/know-map-ai-learning-alpha-v0.3.5/know-map-ai-learning-alpha-v0.3.5/training/data/junyi/srclog/junyi_ProblemLog_original.csv"
DIFF = r"c:/Users/SolimPurmiss/Desktop/know-map-ai-learning-alpha-v0.3.5/know-map-ai-learning-alpha-v0.3.5/training/data/junyi/junyi_exercise_diff.json"
EXTSRC = r"c:/Users/SolimPurmiss/Desktop/know-map-ai-learning-alpha-v0.3.5/know-map-ai-learning-alpha-v0.3.5/training/data/junyi/srctable/junyi_Exercise_table.csv"
OUT = r"c:/Users/SolimPurmiss/Desktop/know-map-ai-learning-alpha-v0.3.5/know-map-ai-learning-alpha-v0.3.5/training/data/junyi/topic_name2id.json"

N_UID = 247606
diff_map = json.load(open(DIFF, encoding="utf-8"))

# exercise -> topic 名（仅 diff_map 覆盖且有 topic）
exe_topic = {}
with open(EXTSRC, encoding="utf-8-sig") as f:
    for rec in csv.DictReader(f):
        name = rec["name"].strip()
        if name in diff_map:
            exe_topic[name] = rec["topic"].strip()

users = [None] * N_UID
with open(LOG, encoding="utf-8") as f:
    r = csv.reader(f)
    h = next(r)
    iu = h.index("user_id"); ie = h.index("exercise"); it = h.index("time_done")
    for row in r:
        if len(row) <= max(iu, ie):
            continue
        exe = row[ie]
        topic = exe_topic.get(exe)
        if topic is None:
            continue
        try:
            uid = int(row[iu])
        except ValueError:
            continue
        try:
            tk = int(row[it])
        except ValueError:
            tk = 0
        if users[uid] is None:
            users[uid] = []
        users[uid].append((tk, exe, topic))

topic_ids = {}
tmap_list = []  # 记录 id 对应顺序，便于自检
def tid(name):
    if name not in topic_ids:
        topic_ids[name] = len(topic_ids)
    return topic_ids[name]

for uid, lst in enumerate(users):
    if not lst:
        continue
    by_topic = {}
    for (tk, exe, topic) in lst:
        by_topic.setdefault(topic, []).append((tk, exe))
    for topic, evs in by_topic.items():
        tid(topic)  # 分配 id

print("topics assigned:", len(topic_ids))
json.dump(topic_ids, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved:", OUT)