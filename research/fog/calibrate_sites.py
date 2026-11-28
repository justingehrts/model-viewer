#!/usr/bin/env python3
"""Per-site fog calibration for the Fog Domain tab.

Downloads hourly surface reports for the Fog Domain sites from the Iowa
Environmental Mesonet ASOS archive (keyless), scores a simple ingredient ladder
(dew-point spread and wind -- fields every model in the extract has) and prints
`FOG_SITE_PROB` and `FOG_SITE_CLIMO` for app.py:

    python calibrate_sites.py            # about 3 minutes, ~1.5 million reports

Fog here is visibility <= 1 mile or a reported FG. For each site the script
counts how often fog was observed in hours at each tier, shrinks those
frequencies toward the all-site frequency (SHRINK pseudo-hours, so a site with few
hours at a tier is not over-read) and also checks that site differences repeat
between independent sets of years. Keep FOG_SITE_TIERS in sync with app.py.

Checked on KCMH (Nov 2026): GFS-based tiers gave fog frequencies of
0.4 / 1.7 / 4.2 / 24% and observed-based tiers 0.4 / 2.1 / 3.5 / 13%, so the
observation tables transfer to model ingredients (slightly conservatively at
the top tier).
"""
import io
import numpy as np
import pandas as pd
import requests

STATIONS = ["CMH", "OSU", "LCK", "TZR", "LHQ", "VTA", "DLZ", "MRT", "4I3", "MNN", "I40", "ZZV", "CDI", "UNI", "JRO", "RZT", "I23", "ILN", "GDK", "SGH"]
# (max dew-point spread F, max wind mph) per tier, tested from the top down. Keep in sync with app.py.
FOG_SITE_TIERS = {3: (1.0, 6.0), 2: (2.0, 9.0), 1: (3.5, 12.0)}
SHRINK = 200
START, END = (2022, 10, 1), (2026, 11, 28)


def download():
    out = []
    for i in range(0, len(STATIONS), 5):
        r = requests.get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py", params={
            "station": STATIONS[i:i + 5], "data": ["tmpf", "dwpf", "sknt", "vsby", "wxcodes"],
            "year1": START[0], "month1": START[1], "day1": START[2], "year2": END[0], "month2": END[1], "day2": END[2],
            "tz": "Etc/UTC", "format": "onlycomma", "latlon": "no", "missing": "M", "trace": "T", "direct": "no",
            "report_type": [3, 4]}, timeout=300)
        r.raise_for_status()
        out.append(pd.read_csv(io.StringIO(r.text), na_values=["M"], keep_default_na=False, dtype=str))
    d = pd.concat(out)
    for c in ("tmpf", "dwpf", "sknt", "vsby"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["valid"] = pd.to_datetime(d["valid"])
    d["hour"] = (d["valid"] + pd.Timedelta(minutes=9)).dt.floor("h")   # a :51 routine report belongs to the next hour
    d["fgwx"] = d["wxcodes"].fillna("").str.contains("FG")
    return d.sort_values("valid")


def hourly(d):
    g = d.groupby(["station", "hour"]).agg(vis=("vsby", "min"), fgwx=("fgwx", "max"), tmpf=("tmpf", "last"),
                                           dwpf=("dwpf", "last"), sknt=("sknt", "last")).reset_index()
    g["fog"] = (g.vis <= 1.0) | g.fgwx
    g["spread"] = g.tmpf - g.dwpf
    g["wind"] = g.sknt * 1.15078
    g = g[g.spread.notna() & g.wind.notna() & g.vis.notna() & (g.spread >= 0)].copy()
    tier = np.zeros(len(g), dtype=int)
    for t in (1, 2, 3):                       # later (higher) tiers overwrite lower ones
        sp, wd = FOG_SITE_TIERS[t]
        tier = np.where((g.spread <= sp) & (g.wind <= wd), t, tier)
    g["tier"] = tier
    return g


def main():
    g = hourly(download())
    pooled = g.groupby("tier").fog.mean()
    print(f"# {len(g)} station-hours; fog rate {100 * g.fog.mean():.2f}%; pooled by tier: " + ", ".join(f"{100 * pooled[t]:.1f}%" for t in range(4)))
    prob, climo = {}, {}
    for st in STATIONS:
        s = g[g.station == st]
        row = []
        for t in range(4):
            x = s[s.tier == t].fog
            row.append(round(float((x.sum() + SHRINK * pooled[t]) / (len(x) + SHRINK)), 4))
        prob[st] = tuple(row)
        climo[st] = (round(float(100 * s.fog.mean()), 2), round(float(100 * s[s.tier == 3].fog.mean()), 1))
    # do the site differences repeat in independent years?  (fog years run Oct-Sep)
    g["fy"] = np.where(g.hour.dt.month >= 10, g.hour.dt.year, g.hour.dt.year - 1)
    hi = g[g.tier >= 2]
    a = hi[hi.fy.isin([2022, 2024])].groupby("station").fog.mean()
    b = hi[hi.fy.isin([2023, 2025])].groupby("station").fog.mean()
    print(f"# site ranking, fog years 2022+2024 vs 2023+2025: Spearman {a.corr(b, method='spearman'):.2f}")
    print("FOG_SITE_PROB = {")
    for st in STATIONS:
        print(f'    "{st}": {prob[st]},')
    print("}")
    print("FOG_SITE_CLIMO = {  # (% of all hours with fog, % of tier-3 hours with fog)")
    for st in STATIONS:
        print(f'    "{st}": {climo[st]},')
    print("}")


if __name__ == "__main__":
    main()
