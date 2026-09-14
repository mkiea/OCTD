import requests, urllib3, re, sys
urllib3.disable_warnings()
u = sys.argv[1]
r = requests.get(u, timeout=(15, 40))
print('status', r.status_code, 'len', len(r.content))
t = r.text
# try common patterns
for pat in [r'href=[\'\"]([^\'\"]+)[\'\"]', r'[A-Za-z0-9_./\-]+\.(?:csv|txt|zip|json|gz)']:
    m = re.findall(pat, t)
    print('pat', pat, '->', m[:60])