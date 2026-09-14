import os, sys, time
from concurrent.futures import ThreadPoolExecutor
import requests, urllib3
urllib3.disable_warnings()

URL = "https://api.github.com/repos/rfarid/LSTM_DKT/contents/dkt_xes3g5m_90_10/dataset.txt?ref=main"
TOTAL = 62618618
DST = r"c:\Users\SolimPurmiss\Desktop\know-map-ai-learning-alpha-v0.3.5\know-map-ai-learning-alpha-v0.3.5\training\data\xes3g5m_dkt_dataset.txt"
parth = DST + ".part"
WORKERS = 16
MPS = 1 << 20  # ~1MB per shard

nsh = (TOTAL + MPS - 1) // MPS
os.makedirs(os.path.dirname(parth), exist_ok=True)

def bounds(i):
    s = i * MPS
    e = min(TOTAL - 1, (i + 1) * MPS - 1)
    return s, e

def fetch(i):
    path = parth + f".{i:04d}"
    want = min(MPS, TOTAL - i * MPS)
    for attempt in range(60):
        try:
            s, e = bounds(i)
            r = requests.get(URL, headers={"accept": "application/vnd.github.raw",
                                           "range": f"bytes={s}-{e}"},
                             timeout=(30, 90), verify=False)
            if r.status_code != 206 or len(r.content) != want:
                time.sleep(0.4); continue
            with open(path, "wb") as f:
                f.write(r.content)
            return True
        except Exception:
            time.sleep(0.6)
    return False

t0 = time.time()
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    results = list(ex.map(fetch, range(nsh)))

bad = [i for i, ok in enumerate(results) if not ok]
done = nsh - len(bad)
print(f"done {done}/{nsh} failed {len(bad)} elapsed {time.time()-t0:.0f}s", flush=True)
if bad:
    print("failed shards:", bad[:20], flush=True)
    sys.exit(1)

print("merging...", flush=True)
with open(DST, "wb") as out:
    for i in range(nsh):
        with open(parth + f".{i:04d}", "rb") as f:
            while True:
                b = f.read(1 << 20)
                if not b:
                    break
                out.write(b)
        os.remove(parth + f".{i:04d}")
print(f"merged size {os.path.getsize(DST)} expect {TOTAL}", flush=True)
print("ALL DONE", flush=True)