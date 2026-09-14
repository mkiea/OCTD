import base64, requests, urllib3, time
urllib3.disable_warnings()
API = "https://api.github.com/repos/jain-jatin/LearnerSystem/contents/Data/Problems.csv"
DST = "c:/Users/SolimPurmiss/Desktop/know-map-ai-learning-alpha-v0.3.5/know-map-ai-learning-alpha-v0.3.5/training/data/assist_fa_Problems.csv"
for attempt in range(5):
    try:
        r = requests.get(API, headers={"User-Agent": "t", "Accept": "application/vnd.github.raw"},
                         timeout=(30, 120), verify=False)
        print("status", r.status_code, "len", len(r.content))
        if r.status_code == 200 and len(r.content) > 100000:
            with open(DST, "wb") as f:
                f.write(r.content)
            print("saved", DST)
            break
    except Exception as e:
        print("err", e)
    time.sleep(2)