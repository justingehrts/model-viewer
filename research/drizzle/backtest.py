import ast, re, glob, sys, numpy as np, pandas as pd
src = open("../../app.py").read(); tree = ast.parse(src)
want = {"_hourly_on", "_first_valid", "drizzle_tiers"}; consts = {"DRIZZLE_TIERS"}
ns = {"np": np, "pd": pd}
for n in tree.body:
    if isinstance(n, ast.FunctionDef) and n.name in want: exec(ast.get_source_segment(src, n), ns)
    elif isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) in consts: exec(ast.get_source_segment(src, n), ns)
LV = [1000, 975, 950, 925, 900, 850, 800, 700]
def station(stn):
    wins = sorted({re.search(rf"{stn}_(\d{{4}}-\d\d-\d\d)_", f).group(1) for f in glob.glob(f"small/{stn}_*_rh.csv")}); parts = []
    for w in wins:
        try: dfs = [pd.read_csv(f"small/{stn}_{w}_{fam}.csv") for fam in ("rh", "t", "sfc")]
        except FileNotFoundError: continue
        m = dfs[0]
        for d in dfs[1:]: m = m.merge(d, on="time")
        parts.append(m)
    g = pd.concat(parts).drop_duplicates("time"); g["time"] = pd.to_datetime(g["time"]); g = g.sort_values("time").reset_index(drop=True)
    mem = {"t2m": g.temperature_2m.to_numpy(), "td2m": g.dew_point_2m.to_numpy(), "precip": g.precipitation.to_numpy(), "sfc_p": g.surface_pressure.to_numpy()}
    for p in LV: mem[f"rh{p}"] = g[f"relative_humidity_{p}hPa"].to_numpy(); mem[f"t{p}"] = g[f"temperature_{p}hPa"].to_numpy()
    window = pd.DatetimeIndex(g["time"]); out = ns["drizzle_tiers"](mem, window, window)
    d = pd.read_csv(f"iem_{stn}.csv", na_values=["M"], keep_default_na=False, dtype=str); d["wx"] = d["wxcodes"].fillna("")
    d["hour"] = (pd.to_datetime(d["valid"]) + pd.Timedelta(minutes=9)).dt.floor("h")
    o = d.groupby("hour").agg(wx=("wx", lambda s: " ".join(s)), n=("wx", "size"))
    f = pd.DataFrame({"dz_tier": out["drizzle"], "fz_tier": out["fzdz"], "t2m": out["t2m"], "top_c": out["top_c"]}, index=window)
    f["dz"] = o["wx"].reindex(window).fillna("").str.contains("DZ").to_numpy(); f["fzdz"] = o["wx"].reindex(window).fillna("").str.contains("FZDZ").to_numpy(); f["has_obs"] = o["n"].reindex(window).notna().to_numpy(); f["stn"] = stn
    return f[f.has_obs & f.index.month.isin([10, 11, 12, 1, 2, 3, 4]) & f.dz_tier.notna()]
F = {s: station(s) for s in ("CMH", "DSM", "ORD", "DTW", "DAY", "IND")}
# Drizzle: real app function over 4 winters of KCMH
c = F["CMH"]; t = c.groupby("dz_tier").agg(hours=("dz", "size"), dz=("dz", "sum")); t["P(DZ)%"] = (100 * t.dz / t.hours).round(1); t["share of DZ%"] = (100 * t.dz / t.dz.sum()).round(0)
print("KCMH, 4 winters: hours", len(c), "DZ", int(c.dz.sum()), "base %.2f%%" % (100 * c.dz.mean())); print(t.to_string())
print("\nDrizzle tier frequency (%) by station, hours with t2m>33F (tiers 0/1/2/3):")
rows = []
for s, f in F.items():
    w = f[f.t2m > 33]; g = w.groupby("dz_tier").agg(h=("dz", "size"), dz=("dz", "sum")); rows.append((s, *[round(100 * g.dz.get(k, 0) / max(g.h.get(k, 0), 1), 1) for k in (0, 1, 2, 3)], *[int(g.h.get(k, 0)) for k in (2, 3)]))
print(pd.DataFrame(rows, columns=["stn", "t0", "t1", "t2", "t3", "n_t2", "n_t3"]).to_string(index=False))
# FZDZ: pooled stations, sub-freezing hours
Z = pd.concat(F.values()); Z = Z[Z.t2m <= 33]
t = Z.groupby("fz_tier").agg(hours=("fzdz", "size"), fzdz=("fzdz", "sum")); t["pct%"] = (100 * t.fzdz / t.hours).round(1); t["share of FZDZ%"] = (100 * t.fzdz / t.fzdz.sum()).round(0); t["lift"] = (t["pct%"] / (100 * Z.fzdz.mean())).round(1)
print("\nFZDZ tiers, sub-freezing hours, 6 stations pooled (FZDZ-rich months => absolute % inflated):"); print(t.to_string())
print("\nFZDZ tiers per station (tier3 / tier2 / tier1 precision %, POD of tier>=1 %):")
for s, f in F.items():
    d = f[f.t2m <= 33]
    if d.fzdz.sum() < 3: print(s, "too few FZDZ hours", int(d.fzdz.sum())); continue
    g = d.groupby("fz_tier").agg(h=("fzdz", "size"), z=("fzdz", "sum")); print(s, [round(100 * g.z.get(k, 0) / max(g.h.get(k, 0), 1), 1) for k in (3, 2, 1)], "POD>=1: %.0f%%" % (100 * d[d.fz_tier >= 1].fzdz.sum() / d.fzdz.sum()), "POD>=2: %.0f%%" % (100 * d[d.fz_tier >= 2].fzdz.sum() / d.fzdz.sum()))
