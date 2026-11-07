#!/usr/bin/env python3
"""Daily KCMH temperature forecast archive for the future skill-weighted
Grand Ensemble (see CLAUDE.md, "Forecast archive").

Run once a day by .github/workflows/archive-forecasts.yml against a checkout
of the orphan `forecast-archive` branch:

    python scripts/archive_forecasts.py --archive-dir archive

Each run:
1. Saves every model's temperature forecast (all ensemble members plus the
   deterministic runs) at 00/06/12/18 UTC valid times out to 14 days, to
   forecasts/KCMH/YYYY-MM-DD.csv.gz.
2. Upserts the last few days of KCMH ASOS observations from the Iowa
   Environmental Mesonet into observations/KCMH.csv.
3. Saves each ensemble member's station-local daily high and low for every
   complete local day in the run, to daily/KCMH/YYYY-MM-DD.csv.gz (the app's
   Trends tab reads these for the Grand Ensemble trend).
4. Recomputes scores/KCMH_temperature.csv: CRPS per run, model and lead
   bucket for every archived forecast that now has an observation
   (deterministic runs score |forecast - obs|, the CRPS of a single value).

Raw Open-Meteo slugs are stored rather than display names, so app.py's
ENS_NAME_MAP / DET_MODEL_SLUGS remain the only translation. This script must
not import app.py (importing it runs the Streamlit UI).
"""
import argparse
import io
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

STATION = "KCMH"
IEM_STATION = "CMH"
STATION_TZ = "America/New_York"
LAT, LON = 39.998, -82.892

ENSEMBLE_SLUGS = ["ecmwf_ifs025", "ecmwf_aifs025", "gfs_seamless", "google_weathernext2_ensemble"]
DETERMINISTIC_SLUGS = ["ecmwf_ifs025", "gfs_seamless", "ncep_nbm_conus"]

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

FORECAST_DAYS = 14
VERIFY_HOURS_UTC = (0, 6, 12, 18)
OBS_LOOKBACK_DAYS = 5
OBS_MAX_OFFSET = pd.Timedelta(minutes=30)
LEAD_BUCKETS = [(0, 48, "day1-2"), (48, 120, "day3-5"), (120, np.inf, "day6+")]
RETRY_WAITS_S = (10, 30, 90)

FORECAST_COLUMNS = ["issued_utc", "kind", "model_slug", "member", "valid_utc", "lead_hours", "temp_f"]
DAILY_COLUMNS = ["issued_utc", "kind", "model_slug", "member", "local_date", "tmax", "tmin"]
MIN_HOURS_PER_DAY = 23  # a complete local day (23 on the spring DST change)


def log(msg):
    print(msg, flush=True)


def get_with_retries(url, params, timeout=60):
    """GET with backoff on throttling (429), server errors and connection
    failures. Raises after the last retry."""
    for attempt, wait in enumerate((0,) + RETRY_WAITS_S):
        if wait:
            log(f"  retrying in {wait}s")
            time.sleep(wait)
        try:
            res = requests.get(url, params=params, timeout=timeout)
            if res.status_code == 429 or res.status_code >= 500:
                log(f"  HTTP {res.status_code}: {res.text[:200]}")
                continue
            res.raise_for_status()
            return res
        except requests.RequestException as e:
            log(f"  request failed: {e}")
    raise RuntimeError(f"giving up on {url}")


def to_rows(times, series_by_member, kind, slug, issued):
    """Long-format rows at the verification hours only."""
    valid = pd.to_datetime(times, utc=True)
    keep = valid.hour.isin(VERIFY_HOURS_UTC)
    frames = []
    for member, values in series_by_member.items():
        temps = pd.to_numeric(pd.Series(values), errors="coerce")
        frames.append(pd.DataFrame({
            "kind": kind,
            "model_slug": slug,
            "member": member,
            "valid_utc": valid[keep],
            "temp_f": temps[keep].to_numpy(),
        }))
    df = pd.concat(frames, ignore_index=True).dropna(subset=["temp_f"])
    df.insert(0, "issued_utc", issued)
    df["lead_hours"] = ((df["valid_utc"] - issued) / pd.Timedelta(hours=1)).round().astype(int)
    return df[df["lead_hours"] >= 0][FORECAST_COLUMNS]


def daily_extremes(times, series_by_member, kind, slug, issued):
    """Each member's station-local daily high/low, complete local days only."""
    local_date = pd.to_datetime(times, utc=True).tz_convert(STATION_TZ).date
    frames = []
    for member, values in series_by_member.items():
        df = pd.DataFrame({"local_date": local_date, "temp_f": pd.to_numeric(pd.Series(values), errors="coerce")})
        g = df.dropna().groupby("local_date")["temp_f"].agg(["max", "min", "count"])
        g = g[g["count"] >= MIN_HOURS_PER_DAY]
        frames.append(pd.DataFrame({"kind": kind, "model_slug": slug, "member": member,
                                    "local_date": g.index, "tmax": g["max"].to_numpy(), "tmin": g["min"].to_numpy()}))
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "issued_utc", issued)
    return df[DAILY_COLUMNS]


def fetch_ensemble(slug, issued):
    params = {
        "latitude": LAT, "longitude": LON, "models": slug, "hourly": "temperature_2m",
        "temperature_unit": "fahrenheit", "timezone": "GMT", "forecast_days": FORECAST_DAYS,
    }
    hourly = get_with_retries(ENSEMBLE_URL, params).json()["hourly"]
    # The control is "temperature_2m"; perturbed members are "_memberNN".
    members = {
        (k[len("temperature_2m_"):] if k != "temperature_2m" else "control"): v
        for k, v in hourly.items() if k == "temperature_2m" or k.startswith("temperature_2m_member")
    }
    return (to_rows(hourly["time"], members, "ensemble", slug, issued),
            daily_extremes(hourly["time"], members, "ensemble", slug, issued))


def fetch_deterministic(issued):
    params = {
        "latitude": LAT, "longitude": LON, "models": ",".join(DETERMINISTIC_SLUGS),
        "hourly": "temperature_2m", "temperature_unit": "fahrenheit", "timezone": "GMT",
        "forecast_days": FORECAST_DAYS,
    }
    hourly = get_with_retries(FORECAST_URL, params).json()["hourly"]
    frames = []
    for slug in DETERMINISTIC_SLUGS:
        key = f"temperature_2m_{slug}"
        if key in hourly:
            frames.append(to_rows(hourly["time"], {"det": hourly[key]}, "deterministic", slug, issued))
        else:
            log(f"  {slug}: no temperature_2m in response")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FORECAST_COLUMNS)


def fetch_observations(now):
    """Routine KCMH METARs from IEM, matched to the nearest verification hour
    within OBS_MAX_OFFSET (an :51 ob counts for the next hour)."""
    start = (now - timedelta(days=OBS_LOOKBACK_DAYS)).date()
    end = (now + timedelta(days=1)).date()
    params = {
        "station": IEM_STATION, "data": "tmpf", "tz": "Etc/UTC", "format": "onlycomma",
        "latlon": "no", "missing": "M", "report_type": "3",
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": end.year, "month2": end.month, "day2": end.day,
    }
    text = get_with_retries(IEM_URL, params).text
    raw = pd.read_csv(io.StringIO(text), na_values="M")
    raw["valid"] = pd.to_datetime(raw["valid"], utc=True)
    raw = raw.dropna(subset=["tmpf"])
    raw["valid_utc"] = raw["valid"].dt.round("h")
    raw["offset"] = (raw["valid"] - raw["valid_utc"]).abs()
    raw = raw[raw["valid_utc"].dt.hour.isin(VERIFY_HOURS_UTC) & (raw["offset"] <= OBS_MAX_OFFSET)]
    raw = raw.sort_values("offset").drop_duplicates("valid_utc")
    return raw.rename(columns={"tmpf": "temp_f"})[["valid_utc", "temp_f"]].sort_values("valid_utc")


def upsert_observations(path, new):
    if path.exists():
        old = pd.read_csv(path, parse_dates=["valid_utc"])
        old["valid_utc"] = pd.to_datetime(old["valid_utc"], utc=True)
        new = pd.concat([old, new]).drop_duplicates("valid_utc", keep="last")
    new = new.sort_values("valid_utc")
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False, date_format="%Y-%m-%dT%H:%MZ")
    return new


def crps(values, obs):
    """CRPS of an ensemble (or a single value) against one observation:
    mean|x_i - y| - (1 / 2n^2) * sum_ij |x_i - x_j|, using the sorted-order
    identity sum_ij |x_i - x_j| = 2 * sum_i (2i - n - 1) x_(i)."""
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    spread = 2 * np.sum((2 * np.arange(1, n + 1) - n - 1) * x)
    return np.mean(np.abs(x - obs)) - spread / (2 * n * n)


def compute_scores(forecast_dir, obs):
    files = sorted(forecast_dir.glob("*.csv.gz"))
    if not files or obs.empty:
        return pd.DataFrame(columns=["issued_utc", "kind", "model_slug", "lead_bucket", "n", "crps"])
    fc = pd.concat((pd.read_csv(f, parse_dates=["issued_utc", "valid_utc"]) for f in files), ignore_index=True)
    for col in ("issued_utc", "valid_utc"):
        fc[col] = pd.to_datetime(fc[col], utc=True)
    fc = fc.merge(obs.rename(columns={"temp_f": "obs_f"}), on="valid_utc", how="inner")
    if fc.empty:
        return pd.DataFrame(columns=["issued_utc", "kind", "model_slug", "lead_bucket", "n", "crps"])

    keys = ["issued_utc", "kind", "model_slug", "valid_utc", "lead_hours"]
    per_time = (fc.groupby(keys)
                  .apply(lambda g: crps(g["temp_f"].to_numpy(), g["obs_f"].iloc[0]), include_groups=False)
                  .rename("crps").reset_index())
    edges = [lo for lo, _, _ in LEAD_BUCKETS] + [np.inf]
    per_time["lead_bucket"] = pd.cut(per_time["lead_hours"], bins=edges, right=False,
                                     labels=[name for _, _, name in LEAD_BUCKETS])
    scores = (per_time.groupby(["issued_utc", "kind", "model_slug", "lead_bucket"], observed=True)["crps"]
                      .agg(n="size", crps="mean").reset_index())
    scores["crps"] = scores["crps"].round(3)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive-dir", required=True, type=Path,
                        help="checkout of the forecast-archive branch")
    args = parser.parse_args()
    root = args.archive_dir

    now = datetime.now(timezone.utc)
    issued = pd.Timestamp(now).floor("h")
    log(f"Archiving {STATION} temperature forecasts issued {issued:%Y-%m-%dT%H:%MZ}")

    frames, daily_frames = [], []
    for slug in ENSEMBLE_SLUGS:
        log(f"ensemble {slug}")
        try:
            df, daily = fetch_ensemble(slug, issued)
            log(f"  {df['member'].nunique()} members, {len(df)} rows, {daily['local_date'].nunique()} full local days")
            frames.append(df)
            daily_frames.append(daily)
        except Exception as e:
            log(f"  skipped: {e}")
        time.sleep(1)
    log("deterministic " + ",".join(DETERMINISTIC_SLUGS))
    try:
        df = fetch_deterministic(issued)
        log(f"  {df['model_slug'].nunique()} models, {len(df)} rows")
        frames.append(df)
    except Exception as e:
        log(f"  skipped: {e}")

    forecast_dir = root / "forecasts" / STATION
    if frames:
        forecasts = pd.concat(frames, ignore_index=True)
        forecast_dir.mkdir(parents=True, exist_ok=True)
        out = forecast_dir / f"{issued:%Y-%m-%d}.csv.gz"
        forecasts.to_csv(out, index=False, date_format="%Y-%m-%dT%H:%MZ", compression="gzip")
        log(f"wrote {out} ({len(forecasts)} rows)")
    if daily_frames:
        daily_dir = root / "daily" / STATION
        daily_dir.mkdir(parents=True, exist_ok=True)
        out = daily_dir / f"{issued:%Y-%m-%d}.csv.gz"
        daily_all = pd.concat(daily_frames, ignore_index=True)
        daily_all.to_csv(out, index=False, date_format="%Y-%m-%dT%H:%MZ", compression="gzip")
        log(f"wrote {out} ({len(daily_all)} rows)")

    obs_path = root / "observations" / f"{STATION}.csv"
    obs = pd.DataFrame(columns=["valid_utc", "temp_f"])
    try:
        obs = upsert_observations(obs_path, fetch_observations(now))
        log(f"observations: {len(obs)} rows through {obs['valid_utc'].max():%Y-%m-%dT%H:%MZ}")
    except Exception as e:
        log(f"observations skipped: {e}")
        if obs_path.exists():
            obs = pd.read_csv(obs_path)
            obs["valid_utc"] = pd.to_datetime(obs["valid_utc"], utc=True)

    scores = compute_scores(forecast_dir, obs)
    scores_path = root / "scores" / f"{STATION}_temperature.csv"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(scores_path, index=False, date_format="%Y-%m-%dT%H:%MZ")
    log(f"scores: {len(scores)} rows")

    if not frames:
        log("ERROR: no forecasts archived")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
