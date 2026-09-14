import os, requests, urllib3
urllib3.disable_warnings()

URL = "https://raw.githubusercontent.com/jain-jatin/LearnerSystem/main/Data/Problems.csv"
DST = "c:/Users/SolimPurmiss/Desktop/know-map-ai-learning-alpha-v0.3.5/know-map-ai-learning-alpha-v0.3.5/training/data/assist_fa_Problems.csv"

r = requests.get(URL, timeout=(30, 120), verify=False)
print(f"status {r.status_code} len {len(r.content)}")
with open(DST, "wb") as f:
    f.write(r.content)
print(f"saved to {DST}")
