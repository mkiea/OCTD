import os, sys, time
from concurrent.futures import ThreadPoolExecutor
import requests, urllib3
urllib3.disable_warnings()

URL = "https://api.github.com/repos/rfarid/LSTM_DKT/contents/dkt_xes3g5m_90_10/dataset.txt?ref=main"
TOTAL = 62618618
DST = r"c:\Users\SolimPurmiss\Desktop\know-map-ai-learning-alpha-v0.3.5\know-map-ai-learning-alpha-v0.3.5\training\data\xes3g5m_dkt_dataset.txt"
parth = DST + ".part"
WORKERS = 6
MPS = 1 << 20

nsh = (TOTAL + MPS - 1) // MPS
os.makedirs(os.path.dirname(parth), exist_ok=True)

def bounds(i):
    s = i * MPS
    e = min(TOTAL - 1, (i + 1) * MPS - 1)
    return s, e

def fetch(i):
    path = parth + f".{i:04d}"
    # 已存在且大小正确则跳过
    if os.path.exists(path) and os.path.getsize(path) == min(MPS, TOTAL - i * MPS):
        return True
    want = min(MPS, TOTAL - i * MPS)
    for attempt in range(10):  # 限重试，避免耗尽 API 配额
        try:
            s, e = bounds(i)
            with requests.get(URL,
                              headers={"accept": "application/vnd.github.raw",
                                       "range": f"bytes={s}-{e}",
                                       "User-Agent": "t"},
                              timeout=(30, 60), verify=False) as r:
                if r.status_code != 206 or len(r.content) != want:
                    time.sleep(0.5); continue
            with open(path, "wb") as f:
                f.write(r.content)
            return True
        except Exception:
            time.sleep(0.5)
    return False

t0 = time.time()
bad = []
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    for i, ok in zip(range(nsh), ex.map(fetch, range(nsh))):
        if not ok:
            bad.append(i)

done = nsh - len(bad)
print(f"pass done {done}/{nsh} failed {len(bad)} elapsed {time.time()-t0:.0f}s", flush=True)
if bad:
    print("failed:", bad[:30], flush=True)
else:
    print("merging...", flush=True)
    with open(DST, "wb") as out:
        for i in range(nsh):
            p = parth + f".{i:04d}"
            with open(p, "rb") as f:
                while True:
                    b = f.read(1 << 20)
                    if not b:
                        break
                    out.write(b)
            os.remove(p)
    print(f"merged size {os.path.getsize(DST)} expect {TOTAL}", flush=True)
    print("ALL DONE", flush=True)