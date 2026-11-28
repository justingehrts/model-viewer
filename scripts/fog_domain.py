#!/usr/bin/env python3
"""Fog-domain extract: crops fog-relevant fields for central Ohio out of NOAA's
GRIB2 model output and writes small files the dashboard can read.

Run by .github/workflows/fog-domain.yml (triggered by cron-job.org, with a
GitHub schedule as backup) against a checkout of the orphan `fog-domain` branch:

    python scripts/fog_domain.py --out out [--models gfs,nbm] [--max-hours 6]

For each model it finds the newest complete cycle, downloads only the needed
GRIB2 messages (HTTP byte ranges from the .idx index files), decodes them with
eccodes, regrids to one regular lat/lon grid (DOMAIN / STEP) and saves
`<model>.npz`. `latest.json` lists what is available. A model whose newest
cycle is already published is skipped, so frequent triggers are cheap.

Adding a model = adding an entry to MODELS (where its files live, which hours,
and which message is which canonical field). Canonical fields:
    vis_mi      visibility (miles, capped at 10)
    p_vis_1mi   chance (%) visibility is under 1 mile   (probabilistic models)
    p_vis_3mi   chance (%) visibility is under 3 miles
    t2m_f       2 m temperature (F)
    spread_f    2 m temperature minus dew point (F)
    wind_mph    10 m wind speed (mph)
    cloud_pct   low cloud cover (%) when the model has it, else total cloud cover
    ceil_ft     cloud ceiling (ft above ground, capped at 12000)

This script must not import app.py (importing it runs the Streamlit UI).
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Domain: central Ohio and the surrounding river valleys, on a ~2.7 x 2.1 km grid.
# ---------------------------------------------------------------------------
DOMAIN = {"lat_min": 38.0, "lat_max": 41.6, "lon_min": -85.3, "lon_max": -80.3}
STEP = 0.025

M_TO_MI = 1 / 1609.344
M_TO_FT = 3.28084
MS_TO_MPH = 2.236936


def k_to_f(k):
    return (k - 273.15) * 9 / 5 + 32


# A message matcher gets (var, level, fcst, extra) from the .idx line and the
# forecast hour. `extra` is the part after the forecast-time field (empty for a
# plain deterministic field; "prob <1609.34:..." for NBM probabilities).
def plain(var, level):
    """Instantaneous deterministic field, e.g. TMP at 2 m above ground."""
    def match(v, lev, fc, extra, fh):
        return v == var and lev == level and extra == "" and fc == f"{fh} hour fcst"
    return match


def prob(var, level, threshold_prefix):
    """NBM/REFS probability message, matched on the start of its threshold text."""
    def match(v, lev, fc, extra, fh):
        return v == var and lev == level and extra.startswith(threshold_prefix) and fc == f"{fh} hour fcst"
    return match


# raw name -> (matcher, conversion to the unit named in the comment)
GFS_FIELDS = {
    "vis": (plain("VIS", "surface"), lambda a: a * M_TO_MI),                     # miles
    "t2m": (plain("TMP", "2 m above ground"), k_to_f),                           # F
    "td2m": (plain("DPT", "2 m above ground"), k_to_f),                          # F
    "u10": (plain("UGRD", "10 m above ground"), lambda a: a * MS_TO_MPH),        # mph
    "v10": (plain("VGRD", "10 m above ground"), lambda a: a * MS_TO_MPH),        # mph
    "lcdc": (plain("LCDC", "low cloud layer"), lambda a: a),                     # %
}
NBM_FIELDS = {
    "vis": (plain("VIS", "surface"), lambda a: a * M_TO_MI),
    "p1": (prob("VIS", "surface", "prob <1609"), lambda a: a),                   # % (already percent)
    "p3": (prob("VIS", "surface", "prob <4828"), lambda a: a),
    "t2m": (plain("TMP", "2 m above ground"), k_to_f),
    "td2m": (plain("DPT", "2 m above ground"), k_to_f),
    "wind": (plain("WIND", "10 m above ground"), lambda a: a * MS_TO_MPH),
    "ceil": (plain("CEIL", "cloud ceiling"), lambda a: a * M_TO_FT),             # ft
    "tcdc": (plain("TCDC", "surface"), lambda a: a),
}

MODELS = {
    "gfs": {
        "label": "GFS",
        "regrid": "idw",           # 0.25 deg grid is coarser than the target: smooth between points
        "cycle_step_h": 6,
        "latency_h": 4,            # newest cycle is normally complete ~4 h after its nominal time
        # hourly for the first day (where fog timing matters most), then 3-hourly
        "hours": list(range(1, 25)) + list(range(27, 73, 3)),
        "url": lambda cyc, fh: (f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{cyc:%Y%m%d}/{cyc:%H}/atmos/"
                                f"gfs.t{cyc:%H}z.pgrb2.0p25.f{fh:03d}"),
        "fields": GFS_FIELDS,
    },
    "nbm": {
        "label": "NBM",
        "regrid": "nn",            # 2.5 km, same scale as the target grid
        "cycle_step_h": 3,         # NBM runs hourly; every 3rd cycle is plenty for a 3-hourly job
        "latency_h": 1,
        "hours": list(range(1, 37)) + list(range(39, 73, 3)),
        "url": lambda cyc, fh: (f"https://noaa-nbm-grib2-pds.s3.amazonaws.com/blend.{cyc:%Y%m%d}/{cyc:%H}/core/"
                                f"blend.t{cyc:%H}z.core.f{fh:03d}.co.grib2"),
        "fields": NBM_FIELDS,
    },
}

HEADERS = {"User-Agent": "model-viewer-fog-domain"}


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------
def http_get(url, headers=None, retries=4, timeout=60):
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers={**HEADERS, **(headers or {})}, timeout=timeout)
            if r.status_code in (200, 206):
                return r
            if r.status_code == 404:
                return r
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = type(e).__name__
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{url}: {last}")


def read_index(url):
    """Parse a GRIB2 .idx file into [{off, var, level, fcst, extra}, ...] or None if missing."""
    r = http_get(url + ".idx")
    if r.status_code == 404:
        return None
    rows = []
    for line in r.text.strip().splitlines():
        p = line.split(":")
        rows.append({"off": int(p[1]), "var": p[3], "level": p[4], "fcst": p[5], "extra": ":".join(p[6:]).strip(":")})
    return rows


def fetch_message(url, rows, i):
    """Download message i of a GRIB2 file with an HTTP byte range."""
    start = rows[i]["off"]
    end = rows[i + 1]["off"] - 1 if i + 1 < len(rows) else ""
    r = http_get(url, headers={"Range": f"bytes={start}-{end}"})
    if r.status_code not in (200, 206):
        raise RuntimeError(f"{url}: HTTP {r.status_code} for byte range")
    return r.content


def pick_messages(rows, fields, fh):
    """{raw name: index of the first matching idx row}."""
    picks = {}
    for name, (match, _conv) in fields.items():
        for i, r in enumerate(rows):
            if match(r["var"], r["level"], r["fcst"], r["extra"], fh):
                picks[name] = i
                break
    return picks


# ---------------------------------------------------------------------------
# Decoding and regridding
# ---------------------------------------------------------------------------
def decode(msg):
    """(values, lat, lon) of one GRIB2 message; missing values become NaN."""
    import eccodes
    gid = eccodes.codes_new_from_message(msg)
    try:
        eccodes.codes_set(gid, "missingValue", 1e20)
        vals = eccodes.codes_get_values(gid).astype("float64")
        vals[vals > 1e19] = np.nan
        lat = eccodes.codes_get_array(gid, "latitudes")
        lon = eccodes.codes_get_array(gid, "longitudes")
    finally:
        eccodes.codes_release(gid)
    lon = ((lon + 180.0) % 360.0) - 180.0
    return vals, lat, lon


def values_only(msg):
    import eccodes
    gid = eccodes.codes_new_from_message(msg)
    try:
        eccodes.codes_set(gid, "missingValue", 1e20)
        vals = eccodes.codes_get_values(gid).astype("float64")
    finally:
        eccodes.codes_release(gid)
    vals[vals > 1e19] = np.nan
    return vals


def target_grid():
    lats = np.round(np.arange(DOMAIN["lat_min"], DOMAIN["lat_max"] + 1e-9, STEP), 4)
    lons = np.round(np.arange(DOMAIN["lon_min"], DOMAIN["lon_max"] + 1e-9, STEP), 4)
    return lats, lons


class Regridder:
    """Maps a model's full grid onto the target grid. Built once per model run
    (from the first message's lat/lon) and reused for every field and hour."""

    def __init__(self, lat, lon, method):
        from scipy.spatial import cKDTree
        lats, lons = target_grid()
        margin = 0.15 if method == "nn" else 0.6
        sel = ((lat >= DOMAIN["lat_min"] - margin) & (lat <= DOMAIN["lat_max"] + margin) &
               (lon >= DOMAIN["lon_min"] - margin) & (lon <= DOMAIN["lon_max"] + margin))
        self.idx = np.flatnonzero(sel)
        if len(self.idx) == 0:
            raise RuntimeError("domain is outside the model grid")
        c = np.cos(np.radians((DOMAIN["lat_min"] + DOMAIN["lat_max"]) / 2))
        tree = cKDTree(np.c_[lon[self.idx] * c, lat[self.idx]])
        glon, glat = np.meshgrid(lons, lats)
        k = 1 if method == "nn" else 4
        dist, ind = tree.query(np.c_[glon.ravel() * c, glat.ravel()], k=k)
        if k == 1:
            self.ind, self.w = ind[:, None], np.ones((len(ind), 1))
        else:
            w = 1.0 / (dist + 1e-6) ** 2
            self.ind, self.w = ind, w / w.sum(axis=1, keepdims=True)
        self.shape = (len(lats), len(lons))

    def apply(self, vals):
        v = vals[self.idx][self.ind]               # (npts, k)
        out = (v * self.w).sum(axis=1)             # NaN anywhere -> NaN (rare: masked cells)
        return out.reshape(self.shape)


# ---------------------------------------------------------------------------
# One model run
# ---------------------------------------------------------------------------
def newest_cycle(cfg, now, max_hours):
    """Latest cycle whose final needed forecast file is already published."""
    hours = cfg["hours"][:max_hours] if max_hours else cfg["hours"]
    step = cfg["cycle_step_h"]
    t = (now - timedelta(hours=cfg["latency_h"])).replace(minute=0, second=0, microsecond=0)
    t = t.replace(hour=(t.hour // step) * step)
    for _ in range(int(48 / step) + 2):
        if read_index(cfg["url"](t, hours[-1])) is not None:
            return t
        t -= timedelta(hours=step)
    raise RuntimeError("no complete cycle found in the last 48 h")


def process_hour(cfg, cyc, fh, regridder_holder):
    url = cfg["url"](cyc, fh)
    rows = read_index(url)
    if rows is None:
        raise RuntimeError(f"missing {url}")
    picks = pick_messages(rows, cfg["fields"], fh)
    raw = {}
    for name, i in picks.items():
        msg = fetch_message(url, rows, i)
        if regridder_holder.get("r") is None:   # first message ever: build the regridder from its lat/lon
            vals, lat, lon = decode(msg)
            regridder_holder["r"] = Regridder(lat, lon, cfg["regrid"])
        else:
            vals = values_only(msg)
        raw[name] = cfg["fields"][name][1](regridder_holder["r"].apply(vals))
    return raw


def canonical(cfg_name, raw):
    """Raw per-hour arrays -> the canonical fields (see the module docstring)."""
    out = {}
    if "vis" in raw:
        out["vis_mi"] = np.clip(raw["vis"], 0, 10)
    for k_raw, k_out in (("p1", "p_vis_1mi"), ("p3", "p_vis_3mi")):
        if k_raw in raw:
            out[k_out] = np.clip(raw[k_raw], 0, 100)
    if "t2m" in raw:
        out["t2m_f"] = raw["t2m"]
        if "td2m" in raw:
            out["spread_f"] = np.clip(raw["t2m"] - raw["td2m"], 0, 40)
    if "wind" in raw:
        out["wind_mph"] = raw["wind"]
    elif "u10" in raw and "v10" in raw:
        out["wind_mph"] = np.hypot(raw["u10"], raw["v10"])
    cloud = raw.get("lcdc", raw.get("tcdc"))
    if cloud is not None:
        out["cloud_pct"] = np.clip(cloud, 0, 100)
    if "ceil" in raw:
        out["ceil_ft"] = np.clip(np.nan_to_num(raw["ceil"], nan=12000), 0, 12000)
    return out


def run_model(name, cfg, out_dir, manifest, now, max_hours, force):
    hours = cfg["hours"][:max_hours] if max_hours else cfg["hours"]
    cyc = newest_cycle(cfg, now, max_hours)
    prev = manifest.get("models", {}).get(name, {})
    if not force and prev.get("cycle_utc") == cyc.strftime("%Y-%m-%dT%H:%MZ") and prev.get("n_times") == len(hours):
        print(f"{name}: cycle {cyc:%Y-%m-%d %HZ} already published; skipping")
        return
    print(f"{name}: processing cycle {cyc:%Y-%m-%d %HZ}, {len(hours)} forecast hours")
    holder = {}
    # Hour 1 builds the regridder; the rest run in parallel threads.
    first = process_hour(cfg, cyc, hours[0], holder)
    results = {hours[0]: first}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {fh: pool.submit(process_hour, cfg, cyc, fh, holder) for fh in hours[1:]}
        for fh, f in futs.items():
            try:
                results[fh] = f.result()
            except Exception as e:  # keep going: a missing hour becomes NaN
                print(f"  f{fh:03d}: {e}", file=sys.stderr)
    done = sorted(results)
    if len(done) < max(3, len(hours) // 2):
        raise RuntimeError(f"only {len(done)} of {len(hours)} hours downloaded")
    per_hour = [canonical(name, results[fh]) for fh in done]
    fields = sorted(set().union(*[set(p) for p in per_hour]))
    lats, lons = target_grid()
    arrays = {}
    for f in fields:
        stack = np.stack([p.get(f, np.full((len(lats), len(lons)), np.nan)) for p in per_hour]).astype("float32")
        if f.startswith("p_vis"):
            arrays[f] = np.where(np.isnan(stack), 255, np.round(stack)).astype("uint8")
        elif f == "cloud_pct":
            arrays[f] = np.where(np.isnan(stack), 255, np.round(stack)).astype("uint8")
        else:
            arrays[f] = stack.astype("float16")
    valid_min = np.array([int((cyc + timedelta(hours=fh)).timestamp() // 60) for fh in done], dtype="int64")
    path = out_dir / f"{name}.npz"
    np.savez_compressed(path, lat=lats.astype("float32"), lon=lons.astype("float32"), valid_min=valid_min,
                        cycle_min=np.int64(cyc.timestamp() // 60), **arrays)
    manifest.setdefault("models", {})[name] = {
        "label": cfg["label"], "cycle_utc": cyc.strftime("%Y-%m-%dT%H:%MZ"), "file": path.name,
        "n_times": len(hours), "hours": done, "fields": fields, "bytes": path.stat().st_size,
    }
    print(f"{name}: wrote {path.name} ({path.stat().st_size / 1e6:.1f} MB, {len(done)} hours, fields {fields})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out", help="output directory (a checkout of the fog-domain branch)")
    ap.add_argument("--models", default=",".join(MODELS), help="comma-separated subset of: " + ",".join(MODELS))
    ap.add_argument("--max-hours", type=int, default=0, help="only the first N forecast hours (for quick tests)")
    ap.add_argument("--force", action="store_true", help="rebuild even if the newest cycle is already published")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mpath = out_dir / "latest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    manifest["domain"] = {**DOMAIN, "step": STEP}
    now = datetime.now(timezone.utc)
    failures = 0
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        if name not in MODELS:
            print(f"unknown model {name}", file=sys.stderr)
            failures += 1
            continue
        try:
            run_model(name, MODELS[name], out_dir, manifest, now, args.max_hours, args.force)
        except Exception as e:
            failures += 1
            print(f"{name}: FAILED: {e}", file=sys.stderr)
            manifest.setdefault("errors", {})[name] = f"{now:%Y-%m-%dT%H:%MZ} {e}"[:300]
        else:
            manifest.get("errors", {}).pop(name, None)
    manifest["generated_utc"] = now.strftime("%Y-%m-%dT%H:%MZ")
    mpath.write_text(json.dumps(manifest, indent=1))
    # Non-zero only if nothing worked at all, so one model's outage doesn't hide the others.
    if failures and failures >= len([m for m in args.models.split(",") if m.strip()]):
        sys.exit(1)


if __name__ == "__main__":
    main()
