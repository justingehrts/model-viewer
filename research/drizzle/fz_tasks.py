import sys; sys.path.insert(0, "."); import requests, io, pandas as pd, json
from fetch_history import FAM, get
import os
os.makedirs("small", exist_ok=True)
S = {"CMH": (39.997, -82.891), "DSM": (41.534, -93.663), "ORD": (41.978, -87.905), "DTW": (42.212, -83.349), "DAY": (39.906, -84.219), "IND": (39.717, -86.295)}
wins = []
for a, b in (("2022-10-01", "2023-04-30"), ("2023-10-01", "2024-04-30"), ("2024-10-01", "2025-04-30"), ("2025-10-01", "2026-04-30")):
    d = pd.Timestamp(a)
    while d <= pd.Timestamp(b):
        e = min(d + pd.Timedelta(days=29), pd.Timestamp(b)); wins.append((d, e)); d = e + pd.Timedelta(days=1)
tasks = []
for s, (lat, lon) in S.items():
    r = requests.get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py", params={"station": s, "data": ["tmpf", "vsby", "wxcodes"], "year1": 2022, "month1": 10, "day1": 1, "year2": 2026, "month2": 11, "day2": 28,
        "tz": "Etc/UTC", "format": "onlycomma", "latlon": "no", "missing": "M", "trace": "T", "direct": "no", "report_type": [3, 4]}, timeout=180)
    open(f"iem_{s}.csv", "w").write(r.text)
    d = pd.read_csv(io.StringIO(r.text), na_values=["M"], keep_default_na=False, dtype=str); d["wx"] = d["wxcodes"].fillna("")
    d["hour"] = (pd.to_datetime(d["valid"]) + pd.Timedelta(minutes=9)).dt.floor("h"); fz = sorted(set(d[d.wx.str.contains("FZDZ")]["hour"]))
    for a, b in wins:
        n = sum(a <= h < b + pd.Timedelta(days=1) for h in fz)
        if n >= 2: tasks.append((s, lat, lon, a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d"), n))
print(len(tasks), "windows:", [(t[0], t[3], t[5]) for t in tasks])
json.dump(tasks, open("fz_tasks.json", "w"))
