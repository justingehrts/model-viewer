import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone
import time
import re
import io
import math
import os
import subprocess

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Multi-Model Ensemble Weather Dashboard",
    page_icon="🌤️",
    layout="wide"
)

# ==============================================================================
# CONFIGURATION & METADATA CONFIG
# ==============================================================================
WEATHER_VARS = {
    "temperature_2m": {
        "label": "Air Temperature",
        "unit": "°F",
        "hourly_param": "temperature_2m",
        "daily_agg": "max",          # 'max' for daily highs
        "chart_title": "Daily High Temperature Spread",
        "grid": {"colorscale": "RdYlBu_r", "spread_floor": 8}
    },
    "apparent_temperature": {
        "label": "Feels Like",
        "unit": "°F",
        "hourly_param": "apparent_temperature",
        "daily_agg": "max",          # 'max' for daily highs, same as temperature_2m
        "chart_title": "Daily High Feels Like Spread",
        "grid": {"colorscale": "RdYlBu_r", "spread_floor": 8}
    },
    "dew_point_2m": {
        "label": "Dew Point",
        "unit": "°F",
        "hourly_param": "dew_point_2m",
        # No daily_agg/chart_title: hourly line chart only, no daily
        # box-and-whisker or high/low tracking for this variable.
        "grid": {"colorscale": "YlGnBu", "spread_floor": 8}
    },
    "precipitation": {
        "label": "Total Precipitation",
        "unit": "in",
        "hourly_param": "precipitation",
        "daily_agg": "sum",          # 'sum' for total daily rainfall
        "chart_title": "Daily Total Precipitation Spread",
        "verify": False,
        "grid": {"colorscale": "Greens", "zmin": 0, "zmax_floor": 0.5, "spread_floor": 0.25,
                 "cumulative": True, "decimals": 2}
    },
    # Snowfall is derived rather than requested directly ("derived": True):
    # Open-Meteo's own `snowfall` assumes a fixed 7:1 ratio, so we back out the
    # model's snow liquid equivalent and apply our own ratios. See
    # compute_snowfall().
    "snowfall_10to1": {
        "label": "Snowfall (10:1)",
        "unit": "in",
        "derived": True,
        "verify": False,
        "daily_agg": "sum",
        "chart_title": "Daily Snowfall Spread (10:1)",
        "grid": {"colorscale": "Blues", "zmin": 0, "zmax_floor": 1, "spread_floor": 1,
                 "cumulative": True, "decimals": 1}
    },
    "snowfall_kuchera": {
        "label": "Snowfall (Kuchera)",
        "unit": "in",
        "derived": True,
        "verify": False,
        "daily_agg": "sum",
        "chart_title": "Daily Snowfall Spread (Kuchera)",
        "grid": {"colorscale": "Blues", "zmin": 0, "zmax_floor": 1, "spread_floor": 1,
                 "cumulative": True, "decimals": 1}
    },
    "snowfall_cobb": {
        "label": "Snowfall (Cobb, approx.)",
        "unit": "in",
        "derived": True,
        "verify": False,
        "daily_agg": "sum",
        "chart_title": "Daily Snowfall Spread (Cobb, approx.)",
        "grid": {"colorscale": "Blues", "zmin": 0, "zmax_floor": 1, "spread_floor": 1,
                 "cumulative": True, "decimals": 1}
    },
    "wind_speed_10m": {
        "label": "Wind Speed",
        "unit": "mph",
        "hourly_param": "wind_speed_10m",
        "daily_agg": "max",          # 'max' for daily peak wind
        "chart_title": "Daily High Wind Speed Spread",
        "grid": {"colorscale": "YlOrRd", "zmin": 0, "spread_floor": 5}
    },
    "wind_gusts_10m": {
        "label": "Wind Gust",
        "unit": "mph",
        "hourly_param": "wind_gusts_10m",
        "daily_agg": "max",          # 'max' for daily peak gust
        "chart_title": "Daily High Wind Gust Spread",
        "grid": {"colorscale": "YlOrRd", "zmin": 0, "spread_floor": 5}
    }
}

# Color Vision Deficiency (CVD) Safe Palette (Okabe-Ito), grouped by modeling
# center: ECMWF models are cool (blues, bluish green), NOAA models warm
# (vermilion, orange), Google's WeatherNext reddish purple.
MODEL_CONFIG = {
    # Deterministic Operational Runs
    "ECMWF Operational": {"color": "#0072B2"},  # Blue
    "GFS Operational":   {"color": "#D55E00"},  # Vermilion
    "NBM Operational":   {"color": "#888888"},  # Mid-Gray
    "RRFS Operational":  {"color": "#882255"},  # Wine (NOAA, Paul Tol)
    "Deterministic":     {"color": "#0072B2"},

    # Ensemble Model Families
    "EPS":            {"color": "#56B4E9"},  # Sky Blue (ECMWF)
    "AIFS":           {"color": "#009E73"},  # Bluish Green (ECMWF)
    "GEFS":           {"color": "#E69F00"},  # Orange (NOAA)
    "WeatherNext":    {"color": "#CC79A7"},  # Reddish Purple (Google)
    "Grand Ensemble": {"color": "#000000"}   # Black (white in dark mode, see model_color)
}


def is_dark_theme():
    """Whether the viewer's Streamlit theme is dark (st.context.theme, newer
    Streamlit; older versions report light). Read when the script runs, so a
    mid-session theme switch applies on the next rerun."""
    theme = getattr(getattr(st, "context", None), "theme", None)
    return getattr(theme, "type", None) == "dark"


def model_color(name):
    """A model's chart color from MODEL_CONFIG. Black becomes white in dark
    mode, where a black trace would vanish against the background."""
    color = MODEL_CONFIG.get(name, MODEL_CONFIG["ECMWF Operational"])["color"]
    return "#FFFFFF" if color == "#000000" and is_dark_theme() else color

# Shared axis styling: explicit, semi-transparent colors so gridlines/ticks
# stay visible under both light and dark browser/OS themes, instead of
# falling back to Streamlit's auto-theming (which can render them at
# low-to-no contrast, e.g. near-invisible in dark mode).
AXIS_STYLE = dict(
    showgrid=True,
    gridcolor="rgba(128, 128, 128, 0.15)",
    gridwidth=1,
    showline=True,
    linecolor="rgba(128, 128, 128, 0.4)",
    linewidth=1.5,
    tickfont=dict(size=12, family="sans-serif"),
    title_font=dict(size=13, family="sans-serif", color="rgba(128, 128, 128, 0.9)")
)

ENS_ORDER = ["EPS", "AIFS", "GEFS", "WeatherNext", "Grand Ensemble"]

ENS_NAME_MAP = {
    "ecmwf_ifs025": "EPS",
    "ecmwf_aifs025": "AIFS",
    "gfs_seamless": "GEFS",
    "google_weathernext2_ensemble": "WeatherNext"
}

DET_MODEL_SLUGS = {
    "ECMWF Operational": "ecmwf_ifs025",
    "GFS Operational": "gfs_seamless",
    "NBM Operational": "ncep_nbm_conus",
    # NOAA RRFS 3 km CONUS (HRRR/NAM successor), 84 h, so it covers only
    # the first ~3.5 days; later hours are simply empty.
    "RRFS Operational": "ncep_rrfs_conus"
}

# Trends tab: Open-Meteo's Previous Runs API returns each deterministic
# model's forecast made 1..7 days before every valid hour.
PREVIOUS_RUN_DAYS = 7
# The daily forecast archive (scripts/archive_forecasts.py) lives on the
# public forecast-archive branch; ensemble trends are available only for
# stations it covers.
ARCHIVE_RAW_URL = "https://raw.githubusercontent.com/justingehrts/model-viewer/forecast-archive"
ARCHIVE_STATIONS = {"KCMH"}

# NCEI 1991-2020 daily normals (high/low normal and standard deviation) for the
# temperature underlay. Normals belong to a weather station, so only airports
# listed here (airport code -> GHCN-Daily station ID) get the underlay.
NORMALS_STATIONS = {"KCMH": "USW00014821"}

# Hazards tab (fog potential). RRFS is NOAA's 3 km CONUS model, HRRR's
# successor on Open-Meteo (HRRR is discontinued there end of Nov 2026); it's
# optional, so the tab still works if it's unavailable.
RRFS_SLUG = DET_MODEL_SLUGS["RRFS Operational"]
HAZARD_HOURS = 72
FOG_TIERS = {
    "mist_mi": 3.0,             # visibility < 3 mi: mist (low)
    "fog_mi": 1.0,              # < 1 mi: fog (high)
    "dense_mi": 0.25,           # < 1/4 mi: dense fog (NWS Dense Fog Advisory)
    "pct_bands": (10, 30, 60),  # % of members/models -> low / moderate / high
    "dense_member_pct": 30,     # GEFS: % of members < dense_mi -> dense likely
    "spread_f": 2.0,            # Signals: dew-point spread <= 2 F ...
    "wind_mph": 10.0,           # ... and wind <= 10 mph
    "radiation_wind_mph": 5.0,  # type: night + wind <= 5 mph -> radiation
    "advection_wind_mph": 15.0, # type: wind 5-15 mph -> advection
}
FOG_LEVELS = ["None", "Low", "Moderate", "High", "Dense likely"]

# Hazards view: drizzle and freezing drizzle. Drizzle falls from a shallow,
# saturated layer with little or no deep moisture above it (no ice crystals
# to grow into snow/rain), and only a little model QPF. Freezing drizzle also
# needs a subfreezing surface and a cloud top warmer than about -10 C, where
# ice nuclei are scarce and the cloud stays liquid. Each model/member is a
# yes/no per hour; ensembles show the share of members that say yes.
DRIZZLE_TIERS = {
    "sat_rh": 90.0,             # saturated: RH >= 90% on a level between the surface and low_top_hpa
    "low_top_hpa": 850,         # ... the saturated layer must be below this level
    "aloft_rh": 85.0,           # no deep moisture: RH < 85% at 800 and 700 mb (where the model has them)
    "max_precip_in": 0.04,      # light QPF only (in/hr); heavier is rain/snow, not drizzle
    "fz_sfc_f": 32.0,           # freezing drizzle: 2 m temperature <= 32 F ...
    "fz_top_c": -10.0,          # ... and saturated-layer top warmer than -10 C
    "pct_bands": (10, 30, 60),  # % of members -> low / moderate / high
}
DRIZZLE_LEVELS = ["None", "Low", "Moderate", "High"]
# Pressure levels each source returns on Open-Meteo (checked live at KCMH).
# RRFS and GFS have the finest low-level spacing; ensembles only 925/850/700.
DRIZZLE_DET_LEVELS = [1000, 975, 950, 925, 900, 850, 800, 700]
DRIZZLE_ENS_T_LEVELS = [1000, 925, 850, 700]
DRIZZLE_ENS_RH_LEVELS = [925, 850, 700]
DRIZZLE_SOURCES = [  # (row label, Open-Meteo slug, is ensemble)
    ("GFS", "gfs_seamless", False), ("ECMWF", "ecmwf_ifs025", False), ("RRFS", RRFS_SLUG, False),
    ("EPS", "ecmwf_ifs025", True), ("AIFS", "ecmwf_aifs025", True), ("GEFS", "gfs_seamless", True),
]

# Inputs for the derived snowfall variables, fetched in a separate request so a
# model that rejects them can't break the main fetch. These are the pressure
# levels the ensemble API actually returns (checked live: nothing in between
# has data). Kuchera uses the levels up to 500 mb; Cobb uses all of them.
# Levels a model doesn't return are simply skipped.
SNOW_DERIVED_VARS = ("snowfall_10to1", "snowfall_kuchera", "snowfall_cobb")
SNOW_PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300]
KUCHERA_TOP_HPA = 500
SNOW_INPUT_PARAMS = ["snowfall", "temperature_2m", "surface_pressure"] + [
    f"{var}_{p}hPa"
    for var in ("temperature", "relative_humidity", "vertical_velocity", "geopotential_height")
    for p in SNOW_PRESSURE_LEVELS
]

# Cobb & Waldstreicher (2005) snow ratio vs. layer temperature (°C), read off
# their Fig. 5 (the paper gives only the plotted curve), anchored to their
# worked example: -25 °C -> 10, -15 °C -> 45, -8 °C -> 6. Approximate.
COBB_SR_TEMPS_C = [-30, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -6, -4, -2, 0]
COBB_SR_VALUES = [10, 10, 11, 14, 20, 29, 40, 50, 45, 36, 25, 15, 9, 7, 6, 6, 6.5, 6.5, 5, 3]
# The column is resampled to 25 mb slices (as NBM's Cobb does) so the
# -12 to -18 °C peak isn't averaged away across 100 mb gaps (700-500 mb).
COBB_SLICE_HPA = 25

# Daily aggregates need at least this many hourly values in the day.
MIN_HOURS_FOR_DAILY = 20

# Verification: if the nearest available forecast hour is farther than this
# from the actual observation time, treat it as "not found" rather than
# silently comparing mismatched hours.
VERIFICATION_MAX_GAP = pd.Timedelta(hours=2)

# ==============================================================================
# HELPER 1: LIVE MODEL RUN CYCLE CALCULATOR (STRICTLY UTC)
# ==============================================================================

def get_actual_run_cycles():
    """Calculates active UTC model run cycles based on UTC time and server ingestion latency."""
    now_utc = datetime.now(timezone.utc)
    
    # GFS / GEFS: 6-hour cycles (00Z, 06Z, 12Z, 18Z) with ~3.5 hour ingest lag
    gfs_cutoff = now_utc - pd.Timedelta(hours=3, minutes=30)
    gfs_hour = (gfs_cutoff.hour // 6) * 6
    gfs_str = gfs_cutoff.replace(hour=gfs_hour, minute=0, second=0, microsecond=0).strftime("%m/%d %HZ")
    
    # ECMWF / EPS / AIFS: 12-hour main cycles (00Z/12Z) with ~7 hour ingest lag
    ecmwf_cutoff = now_utc - pd.Timedelta(hours=7)
    ecmwf_hour = (ecmwf_cutoff.hour // 12) * 12
    ecmwf_str = ecmwf_cutoff.replace(hour=ecmwf_hour, minute=0, second=0, microsecond=0).strftime("%m/%d %HZ")

    # NBM: hourly cycles (updated every UTC hour) with ~1.5 hour ingest lag
    nbm_cutoff = now_utc - pd.Timedelta(hours=1, minutes=30)
    nbm_str = nbm_cutoff.replace(minute=0, second=0, microsecond=0).strftime("%m/%d %HZ")

    # RRFS: 6-hour cycles (00Z, 06Z, 12Z, 18Z) with ~2.5 hour ingest lag (estimate)
    rrfs_cutoff = now_utc - pd.Timedelta(hours=2, minutes=30)
    rrfs_str = rrfs_cutoff.replace(hour=(rrfs_cutoff.hour // 6) * 6, minute=0, second=0, microsecond=0).strftime("%m/%d %HZ")

    return {
        "ECMWF Operational": ecmwf_str,
        "GFS Operational": gfs_str,
        "NBM Operational": nbm_str,
        "RRFS Operational": rrfs_str,
        "EPS": ecmwf_str,
        "AIFS": ecmwf_str,
        "GEFS": gfs_str,
        "WeatherNext": ecmwf_str
    }

# ==============================================================================
# HELPER 2: DEV MOCK DATA GENERATOR (OFFLINE / 0 API CALLS)
# ==============================================================================

def generate_mock_data(days=7):
    """Generates synthetic hourly weather data so you can test layout/charts offline."""
    now = datetime.now()
    dates = pd.date_range(start=now, periods=days * 24, freq='h')
    
    # Generate realistic diurnal temperature curve (60°F to 80°F)
    base_temp = 70 + 10 * np.sin(np.linspace(0, days * 2 * np.pi, len(dates)))
    # Generate a mild diurnal wind speed curve (5 to 15 mph)
    base_wind = 10 + 5 * np.sin(np.linspace(0, days * 2 * np.pi, len(dates)))
    # Gusts run stronger than sustained wind speed
    base_gust = base_wind * 1.4
    # Dew point trails a few degrees below air temperature
    base_dew_point = base_temp - 8
    # Feels-like temperature runs a few degrees warmer than actual air
    # temperature (humidity effect)
    base_apparent_temp = base_temp + 4

    dict_det = {
        "temperature_2m": pd.DataFrame({
            'time': dates,
            'ECMWF Operational': base_temp + 1.0,
            'GFS Operational': base_temp - 1.0
        }),
        "apparent_temperature": pd.DataFrame({
            'time': dates,
            'ECMWF Operational': base_apparent_temp + 1.0,
            'GFS Operational': base_apparent_temp - 1.0
        }),
        "precipitation": pd.DataFrame({
            'time': dates,
            'ECMWF Operational': np.zeros(len(dates)),
            'GFS Operational': np.zeros(len(dates))
        }),
        "wind_speed_10m": pd.DataFrame({
            'time': dates,
            'ECMWF Operational': base_wind + 1.0,
            'GFS Operational': base_wind - 1.0
        }),
        "wind_gusts_10m": pd.DataFrame({
            'time': dates,
            'ECMWF Operational': base_gust + 1.0,
            'GFS Operational': base_gust - 1.0
        }),
        "dew_point_2m": pd.DataFrame({
            'time': dates,
            'ECMWF Operational': base_dew_point + 1.0,
            'GFS Operational': base_dew_point - 1.0
        })
    }

    # Synthetic snow event (liquid equivalent, in/hr) about 30-50 hours out;
    # Kuchera uses a fixed 14:1 here, standing in for a cold column.
    snow_swe = np.zeros(len(dates))
    snow_swe[30:50] = 0.03 * np.sin(np.linspace(0, np.pi, 20))
    dict_det["snowfall_10to1"] = pd.DataFrame({
        'time': dates,
        'ECMWF Operational': snow_swe * 10 * 1.1,
        'GFS Operational': snow_swe * 10 * 0.9
    })
    dict_det["snowfall_kuchera"] = pd.DataFrame({
        'time': dates,
        'ECMWF Operational': snow_swe * 14 * 1.1,
        'GFS Operational': snow_swe * 14 * 0.9
    })
    dict_det["snowfall_cobb"] = pd.DataFrame({
        'time': dates,
        'ECMWF Operational': snow_swe * 17 * 1.1,
        'GFS Operational': snow_swe * 17 * 0.9
    })

    # RRFS only runs 84 h; mirror that so Dev Mode shows its lines stopping.
    rrfs_reach = np.arange(len(dates)) < 84
    for var_key, df in dict_det.items():
        if var_key != "precipitation" and "ECMWF Operational" in df.columns:
            df["RRFS Operational"] = np.where(rrfs_reach, df["ECMWF Operational"] - 0.5, np.nan)

    dict_ens = {var_key: {} for var_key in WEATHER_VARS}
    run_cycles = {}

    # AIFS and WeatherNext are AI-based models without gust parameterization,
    # so they're left out of wind_gusts_10m here to mirror the live API's
    # exclusion behavior for Dev Mode testing.
    gust_capable_nicknames = {"EPS", "GEFS"}

    for nickname in ["EPS", "AIFS", "GEFS", "WeatherNext"]:
        df_t = pd.DataFrame({'time': dates})
        df_a = pd.DataFrame({'time': dates})
        df_p = pd.DataFrame({'time': dates})
        df_w = pd.DataFrame({'time': dates})
        df_g = pd.DataFrame({'time': dates})
        df_d = pd.DataFrame({'time': dates})

        # Add synthetic ensemble member variation
        for m in range(1, 31):
            df_t[f"member_{m}"] = base_temp + np.random.normal(0, 2.5, len(dates))
            df_a[f"member_{m}"] = base_apparent_temp + np.random.normal(0, 2.5, len(dates))
            df_p[f"member_{m}"] = np.maximum(0, np.random.normal(0, 0.05, len(dates)))
            df_w[f"member_{m}"] = np.maximum(0, base_wind + np.random.normal(0, 2.0, len(dates)))
            df_g[f"member_{m}"] = np.maximum(0, base_gust + np.random.normal(0, 3.0, len(dates)))
            df_d[f"member_{m}"] = base_dew_point + np.random.normal(0, 2.0, len(dates))

        dict_ens["temperature_2m"][nickname] = df_t
        dict_ens["apparent_temperature"][nickname] = df_a
        dict_ens["precipitation"][nickname] = df_p
        dict_ens["wind_speed_10m"][nickname] = df_w
        if nickname in gust_capable_nicknames:
            dict_ens["wind_gusts_10m"][nickname] = df_g
        dict_ens["dew_point_2m"][nickname] = df_d

        member_swe = {f"member_{m}": snow_swe * np.random.uniform(0.5, 1.5) for m in range(1, 31)}
        dict_ens["snowfall_10to1"][nickname] = pd.DataFrame(
            {'time': dates, **{k: v * 10 for k, v in member_swe.items()}})
        dict_ens["snowfall_kuchera"][nickname] = pd.DataFrame(
            {'time': dates, **{k: v * np.random.uniform(11, 17) for k, v in member_swe.items()}})
        dict_ens["snowfall_cobb"][nickname] = pd.DataFrame(
            {'time': dates, **{k: v * np.random.uniform(10, 25) for k, v in member_swe.items()}})
        run_cycles[nickname] = "DEV-MOCK 00Z"

    det_run_cycles = {
        "ECMWF Operational": "DEV-MOCK 00Z",
        "GFS Operational": "DEV-MOCK 00Z",
        "RRFS Operational": "DEV-MOCK 00Z"
    }

    return dict_det, dict_ens, det_run_cycles, run_cycles


def generate_mock_fog_inputs():
    """Synthetic fog inputs for Dev Mode, shaped like fetch_fog_inputs(): an
    overnight fog event roughly 30-40 hours out (dense in the middle)."""
    rng = np.random.RandomState(11)
    times = pd.date_range(pd.Timestamp.now().floor('h'), periods=HAZARD_HOURS + 24, freq='h')
    hours = np.arange(len(times))
    dip = np.clip(1 - np.abs(hours - 35) / 6, 0, 1)  # 0..1 peak at hour 35
    base_vis = 10 - 9.9 * dip

    def vis_frame(cols, noise):
        df = pd.DataFrame({'time': times})
        for c in cols:
            df[c] = np.clip(base_vis + rng.normal(0, noise, len(times)) * (1 + 3 * dip), 0.05, 10)
        return df

    is_day = pd.DataFrame({'time': times, 'is_day': ((times.hour >= 7) & (times.hour < 18)).astype(int)})
    low_cloud = pd.DataFrame({'time': times, 'GFS Operational': 100 * dip, 'NBM Operational': 100 * dip})
    return {
        "deterministic": {"vis": vis_frame(["GFS Operational", "NBM Operational"], 0.1),
                          "low_cloud": low_cloud, "is_day": is_day},
        "RRFS": {"vis": vis_frame(["RRFS"], 0.1)},
        "GEFS": {"vis": vis_frame([f"member{m:02d}" for m in range(1, 31)], 0.6)},
    }, []


def generate_mock_previous_runs():
    """Synthetic Previous Runs data for Dev Mode, shaped like
    fetch_previous_runs(): columns n0 (latest) .. n7, where nN is the forecast
    made N days before each hour; the error grows with N."""
    rng = np.random.RandomState(7)
    start = pd.Timestamp.now().normalize() - pd.Timedelta(days=PREVIOUS_RUN_DAYS)
    times = pd.date_range(start, periods=(PREVIOUS_RUN_DAYS + 8) * 24, freq='h')
    base = 55 + 10 * np.sin((times.hour.to_numpy() - 9) / 24 * 2 * np.pi)
    runs = {}
    for i, model_name in enumerate(["ECMWF Operational", "GFS Operational"]):
        df = pd.DataFrame({'time': times})
        for n in range(PREVIOUS_RUN_DAYS + 1):
            drift = rng.normal(0, 0.8 * n + 0.3, size=len(times) // 24 + 1).repeat(24)[:len(times)]
            df[f"n{n}"] = base + (i - 0.5) * 2 + drift
        runs[model_name] = df
    return runs

# ==============================================================================
# HELPER 3: AIRPORT GEOCODING LOOKUP
# ==============================================================================

@st.cache_data(ttl=86400)
def get_coordinates_from_airport(airport_code):
    """Looks up lat/lon for ICAO/IATA airport codes (e.g., KCMH, CMH) via NOAA + Open-Meteo."""
    code = airport_code.strip().upper()
    if not code:
        return 39.99, -82.89, "Port Columbus Intl (KCMH)"
        
    url = f"https://aviationweather.gov/api/data/stationinfo?ids={code}&format=json"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) > 0:
                station = data[0]
                lat = station.get("lat")
                lon = station.get("lon")
                name = station.get("site", code)
                if lat is not None and lon is not None:
                    return lat, lon, name
    except Exception:
        pass
        
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={code}&count=1"
    try:
        res = requests.get(geo_url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if "results" in data and len(data["results"]) > 0:
                item = data["results"][0]
                return item["latitude"], item["longitude"], item["name"]
    except Exception:
        pass

    return 39.99, -82.89, "Default Location (KCMH)"

@st.cache_data
def get_deployed_commit():
    """Short git commit SHA of the running code, so what's actually
    deployed (e.g. on Streamlit Cloud) can be confirmed at a glance
    instead of inferred from traceback line numbers."""
    try:
        repo_dir = os.path.dirname(os.path.abspath(__file__))
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir, stderr=subprocess.DEVNULL, timeout=2
        ).decode().strip()
        return sha
    except Exception:
        return "unknown"

# ==============================================================================
# HELPER 4: MODEL VERIFICATION (LIVE METAR vs. FORECAST)
# ==============================================================================

def calculate_apparent_temperature(temp_c, dewpoint_c, wind_speed_ms):
    """Australian Bureau of Meteorology 'Apparent Temperature' formula --
    the same one Open-Meteo uses for its apparent_temperature variable --
    so a METAR-derived feels-like is comparable to the forecast models'
    value. e (vapor pressure, hPa) comes from dewpoint via Magnus-Tetens;
    wind_speed_ms must be m/s. Returns degrees C.
    """
    e = 6.1094 * math.exp(17.625 * dewpoint_c / (243.04 + dewpoint_c))
    return temp_c + 0.33 * e - 0.70 * wind_speed_ms - 4.00

@st.cache_data(ttl=600)
def get_current_conditions(airport_code):
    """Fetches the latest METAR observation for a station from
    aviationweather.gov (same host/pattern as get_coordinates_from_airport).
    600s TTL: obs post roughly hourly (sometimes half-hourly), so the 900s
    forecast TTL would serve a stale "actual" for too long, while a much
    shorter TTL would just add API calls with nothing new landing.
    """
    code = airport_code.strip().upper()
    url = f"https://aviationweather.gov/api/data/metar?ids={code}&format=json"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) > 0:
                ob = data[0]
                if ob.get("obsTime") is not None and ob.get("temp") is not None:
                    return {
                        "temp_c": ob.get("temp"),
                        "dewp_c": ob.get("dewp"),
                        "wspd_kt": ob.get("wspd"),
                        "wgst_kt": ob.get("wgst"),   # often absent -- no gust
                        "obs_time": ob.get("obsTime"),
                    }, None
        return None, "No recent observation available"
    except Exception as e:
        return None, str(e)

# ==============================================================================
# LAYER 1: LIVE DATA INGESTION (CACHED FOR 1 HOUR)
# ==============================================================================

def fetch_deterministic_data(lat, lon, days=7):
    """Fetches explicit operational deterministic runs for ECMWF IFS, GFS, and
    NBM. Returns (dict_det, fetch_time, det_run_cycles, error, utc_offset_seconds).
    Failures are returned here but never cached (see _fetch_deterministic_cached),
    so the next rerun tries again instead of showing the error for an hour."""
    try:
        return _fetch_deterministic_cached(lat, lon, days)
    except Exception as e:
        return {var_key: pd.DataFrame() for var_key in WEATHER_VARS}, "", {}, str(e), 0


@st.cache_data(ttl=3600)
def _fetch_deterministic_cached(lat, lon, days):
    """Cached body of fetch_deterministic_data. Raises on failure:
    st.cache_data doesn't cache exceptions, only successful results."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(cfg["hourly_param"] for cfg in WEATHER_VARS.values() if not cfg.get("derived")),
        "models": list(DET_MODEL_SLUGS.values()),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "auto",  # Set to auto to ensure timestamps match local station time
        "forecast_days": days
    }

    all_cycles = get_actual_run_cycles()
    det_run_cycles = {
        "ECMWF Operational": all_cycles["ECMWF Operational"],
        "GFS Operational": all_cycles["GFS Operational"],
        "NBM Operational": all_cycles["NBM Operational"],
        "RRFS Operational": all_cycles["RRFS Operational"]
    }

    res = requests.get(url, params=params, timeout=10)
    res.raise_for_status()
    data = res.json()

    hourly = data["hourly"]
    utc_offset_seconds = data.get("utc_offset_seconds", 0)
    dict_det = {}
    for var_key, cfg in WEATHER_VARS.items():
        if cfg.get("derived"):
            continue
        hourly_param = cfg["hourly_param"]
        df_var = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
        for model_name, slug in DET_MODEL_SLUGS.items():
            key = f"{hourly_param}_{slug}"
            if key in hourly:
                df_var[model_name] = pd.to_numeric(hourly[key], errors="coerce")
            # else: this model doesn't return this variable -- leave it
            # out entirely so it's excluded from charts/legends instead
            # of showing an empty/all-NaN series.
        dict_det[var_key] = df_var

    # Snowfall inputs: separate request so a failure only costs snowfall.
    snow_hourly = None
    try:
        res_snow = requests.get(url, params={**params, "hourly": ",".join(SNOW_INPUT_PARAMS)}, timeout=10)
        res_snow.raise_for_status()
        snow_hourly = res_snow.json()["hourly"]
    except Exception:
        pass
    snow_times = pd.to_datetime((snow_hourly or hourly)["time"])
    snow_frames = {k: pd.DataFrame({"time": snow_times}) for k in SNOW_DERIVED_VARS}
    if snow_hourly:
        for model_name, slug in DET_MODEL_SLUGS.items():
            derived = compute_snowfall(lambda param: snow_hourly.get(f"{param}_{slug}"))
            for var_key, values in derived.items():
                snow_frames[var_key][model_name] = values
    dict_det.update(snow_frames)

    fetch_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return dict_det, fetch_time, det_run_cycles, None, utc_offset_seconds

@st.cache_data(ttl=3600)
def fetch_ensemble_data(lat, lon, days=7):
    """Fetches probabilistic ensemble members across EPS, AIFS, GEFS, WeatherNext."""
    url = "https://ensemble-api.open-meteo.com/v1/ensemble"
    models = ["ecmwf_ifs025", "ecmwf_aifs025", "gfs_seamless", "google_weathernext2_ensemble"]

    dict_ens = {var_key: {} for var_key in WEATHER_VARS}
    errors = []

    all_cycles = get_actual_run_cycles()
    run_cycles = {
        "EPS": all_cycles["EPS"],
        "AIFS": all_cycles["AIFS"],
        "GEFS": all_cycles["GEFS"],
        "WeatherNext": all_cycles["WeatherNext"]
    }

    for m in models:
        nickname = ENS_NAME_MAP.get(m, m)
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(cfg["hourly_param"] for cfg in WEATHER_VARS.values() if not cfg.get("derived")),
            "models": m,
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "timezone": "auto",  # Set to auto to ensure timestamps match local station time
            "forecast_days": days
        }

        try:
            res = requests.get(url, params=params, timeout=15)
            if res.status_code != 200:
                errors.append(f"{nickname}: HTTP {res.status_code}")
                continue

            data = res.json()
            if "hourly" not in data or "time" not in data["hourly"]:
                continue

            hourly = data["hourly"]

            for var_key, cfg in WEATHER_VARS.items():
                if cfg.get("derived"):
                    continue
                hourly_param = cfg["hourly_param"]
                var_keys = [k for k in hourly.keys() if k.startswith(hourly_param)]
                if not var_keys:
                    # This model's response has no columns for this variable
                    # (e.g. AI-based models often lack diagnostic fields like
                    # wind gusts) -- skip it here rather than adding an
                    # empty/all-NaN series, so it's cleanly excluded from
                    # this variable's charts instead of erroring downstream.
                    continue

                df_m_var = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
                for k in var_keys:
                    col = k.replace(f"{hourly_param}_", "")
                    df_m_var[col] = pd.to_numeric(hourly[k], errors="coerce")

                dict_ens[var_key][nickname] = df_m_var

            # Snowfall inputs: separate request so a model that rejects the
            # pressure-level fields still keeps its other variables.
            time.sleep(0.15)
            try:
                res_snow = requests.get(url, params={**params, "hourly": ",".join(SNOW_INPUT_PARAMS)}, timeout=15)
                res_snow.raise_for_status()
                snow_hourly = res_snow.json()["hourly"]
                # Members are suffixed "_memberNN"; the control has no suffix.
                suffixes = [k[len("snowfall"):] for k in snow_hourly if k == "snowfall" or k.startswith("snowfall_member")]
                snow_times = pd.to_datetime(snow_hourly["time"])
                snow_frames = {k: pd.DataFrame({"time": snow_times}) for k in SNOW_DERIVED_VARS}
                for sfx in suffixes:
                    col = sfx.lstrip("_") or "control"
                    derived = compute_snowfall(lambda param: snow_hourly.get(f"{param}{sfx}"))
                    for var_key, values in derived.items():
                        snow_frames[var_key][col] = values
                for var_key, df_snow in snow_frames.items():
                    if len(df_snow.columns) > 1:
                        dict_ens[var_key][nickname] = df_snow
            except Exception as e:
                errors.append(f"{nickname} snowfall: {str(e)}")

            time.sleep(0.15)

        except Exception as e:
            errors.append(f"{nickname}: {str(e)}")
            continue

    return dict_ens, run_cycles, errors

def fetch_previous_runs(lat, lon):
    """Deterministic temperature forecasts from earlier runs via Open-Meteo's
    Previous Runs API. Returns ({model_name: DataFrame(time, n0..n7)}, error),
    where column nN is the forecast made N days before each (station-local)
    hour and n0 is the latest run. Models or leads missing from the response
    are left out. Failures are returned but never cached, so the next rerun
    tries again."""
    try:
        return _fetch_previous_runs_cached(lat, lon), None
    except Exception as e:
        return {}, str(e)


@st.cache_data(ttl=3600)
def _fetch_previous_runs_cached(lat, lon):
    """Cached body of fetch_previous_runs. Retries once on a timeout or
    connection error, then raises (st.cache_data doesn't cache exceptions)."""
    url = "https://previous-runs-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(["temperature_2m"] + [f"temperature_2m_previous_day{n}" for n in range(1, PREVIOUS_RUN_DAYS + 1)]),
        "models": ",".join(DET_MODEL_SLUGS.values()),
        "temperature_unit": "fahrenheit",
        "timezone": "auto",
        "past_days": PREVIOUS_RUN_DAYS,
        "forecast_days": 8
    }
    for attempt in range(2):
        try:
            res = requests.get(url, params=params, timeout=30)
            break
        except (requests.Timeout, requests.ConnectionError):
            if attempt == 1:
                raise
            time.sleep(2)
    res.raise_for_status()
    hourly = res.json()["hourly"]
    runs = {}
    for model_name, slug in DET_MODEL_SLUGS.items():
        df = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
        for n in range(PREVIOUS_RUN_DAYS + 1):
            key = f"temperature_2m_{slug}" if n == 0 else f"temperature_2m_previous_day{n}_{slug}"
            if key in hourly:
                df[f"n{n}"] = pd.to_numeric(hourly[key], errors="coerce")
        if "n0" in df.columns:
            runs[model_name] = df
    if not runs:
        raise ValueError("no deterministic models in the Previous Runs response")
    return runs


@st.cache_data(ttl=3600)
def fetch_archive_daily(station, utc_offset_seconds):
    """Per-member ensemble daily highs/lows from the last PREVIOUS_RUN_DAYS + 1
    archive runs (forecast-archive branch, daily/<station>/<date>.csv.gz).
    Adds run_day (the run's station-local date) and family (display name via
    ENS_NAME_MAP). Returns an empty DataFrame if nothing is archived yet."""
    today_utc = datetime.now(timezone.utc).date()
    frames = []
    for i in range(PREVIOUS_RUN_DAYS + 1):
        day = today_utc - pd.Timedelta(days=i)
        try:
            res = requests.get(f"{ARCHIVE_RAW_URL}/daily/{station}/{day:%Y-%m-%d}.csv.gz", timeout=10)
            if res.status_code != 200:
                continue
            frames.append(pd.read_csv(io.BytesIO(res.content), compression="gzip"))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[df["kind"] == "ensemble"].copy()
    issued_local = pd.to_datetime(df["issued_utc"], utc=True).dt.tz_localize(None) + pd.Timedelta(seconds=utc_offset_seconds)
    df["run_day"] = issued_local.dt.normalize()
    df["target_day"] = pd.to_datetime(df["local_date"])
    df["family"] = df["model_slug"].map(ENS_NAME_MAP)
    return df.dropna(subset=["family"])

def generate_mock_normals():
    """Synthetic daily normals for Dev Mode (same shape as fetch_daily_normals)."""
    keys = pd.date_range("2020-01-01", "2020-12-31").strftime("%m-%d")
    return pd.DataFrame({"tmax": 72.0, "tmin": 58.0, "sd_max": 9.0, "sd_min": 8.0}, index=keys)


def generate_mock_hourly_normals():
    """Synthetic hourly normals for Dev Mode (same shape as fetch_hourly_normals)."""
    idx = pd.date_range("2021-01-01", "2021-12-31 23:00", freq="h")
    normal = 65 + 8 * np.sin((idx.hour.to_numpy() - 9) / 24 * 2 * np.pi)
    return pd.DataFrame({"normal": normal, "p10": normal - 9, "p90": normal + 9}, index=idx.strftime("%m-%dT%H"))


def generate_mock_drizzle_inputs():
    """Synthetic drizzle inputs for Dev Mode, shaped like fetch_drizzle_inputs():
    a shallow saturated layer with a subfreezing surface about 20-40 hours out."""
    rng = np.random.RandomState(5)
    times = pd.date_range(pd.Timestamp.now().floor('h'), periods=HAZARD_HOURS + 24, freq='h')
    hours = np.arange(len(times))
    event = ((hours >= 20) & (hours <= 40)).astype(float)

    def member(skill):
        flip = rng.rand(len(times)) < skill  # chance this member agrees with the "truth" this hour
        on = np.where(flip, event, 1 - event) if skill < 1 else event
        m = {"t2m": np.where(on > 0, 29.0, 38.0) + rng.normal(0, 1, len(times)),
             "precip": np.where(on > 0, 0.01, 0.0), "sfc_p": np.full(len(times), 985.0)}
        for p in DRIZZLE_DET_LEVELS:
            m[f"rh{p}"] = np.where(on > 0, 96.0, 60.0) if p >= 850 else np.full(len(times), 55.0)
            m[f"t{p}"] = 29.0 - (1000 - p) * 0.08 + np.where(on > 0, 0, 3) + 0 * hours
        return m

    out = {}
    for label, _slug, is_ens in DRIZZLE_SOURCES:
        n = 25 if is_ens else 1
        skill = 0.85 if is_ens else 1.0
        out[label] = {"time": times, "members": {f"m{i:02d}": member(skill) for i in range(n)}}
    return out, []


@st.cache_data(ttl=7 * 24 * 3600)
def fetch_daily_normals(ghcn_id):
    """NCEI 1991-2020 daily normals for one station: DataFrame indexed by
    "MM-DD" with tmax, tmin (normal high/low, F) and sd_max, sd_min (standard
    deviation of the daily high/low, F). Raises on failure (never cached)."""
    res = requests.get("https://www.ncei.noaa.gov/access/services/data/v1", params={
        "dataset": "normals-daily-1991-2020", "stations": ghcn_id,
        "startDate": "2020-01-01", "endDate": "2020-12-31",  # a leap year, so 02-29 is included
        "dataTypes": "DLY-TMAX-NORMAL,DLY-TMIN-NORMAL,DLY-TMAX-STDDEV,DLY-TMIN-STDDEV",
        "format": "json", "units": "standard"}, timeout=30)
    res.raise_for_status()
    rows = res.json()
    if not rows:
        raise ValueError(f"no normals returned for {ghcn_id}")
    df = pd.DataFrame(rows).set_index("DATE")
    out = pd.DataFrame({
        "tmax": pd.to_numeric(df["DLY-TMAX-NORMAL"], errors="coerce"), "tmin": pd.to_numeric(df["DLY-TMIN-NORMAL"], errors="coerce"),
        "sd_max": pd.to_numeric(df["DLY-TMAX-STDDEV"], errors="coerce"), "sd_min": pd.to_numeric(df["DLY-TMIN-STDDEV"], errors="coerce"),
    })
    if out.isna().all().any():
        raise ValueError("normals are missing a field")
    return out.sort_index().ffill().bfill()


def normal_daily(normals, dates):
    """Normal high/low and their standard deviations for ISO date strings:
    (tmax, tmin, sd_max, sd_min) arrays."""
    rows = normals.reindex([pd.to_datetime(d).strftime("%m-%d") for d in dates])
    return rows["tmax"].to_numpy(), rows["tmin"].to_numpy(), rows["sd_max"].to_numpy(), rows["sd_min"].to_numpy()


@st.cache_data(ttl=7 * 24 * 3600)
def fetch_hourly_normals(ghcn_id):
    """NCEI 1991-2020 hourly temperature normals for one station: DataFrame
    indexed by "MM-DDTHH" with normal, p10 and p90 (the 10th and 90th
    percentile of the hour's temperature, F). NCEI publishes no standard
    deviation by hour. 365 days (no Feb 29). Raises on failure."""
    res = requests.get("https://www.ncei.noaa.gov/access/services/data/v1", params={
        "dataset": "normals-hourly-1991-2020", "stations": ghcn_id,
        "startDate": "2020-01-01T00:00:00", "endDate": "2020-12-31T23:00:00",
        "dataTypes": "HLY-TEMP-NORMAL,HLY-TEMP-10PCTL,HLY-TEMP-90PCTL",
        "format": "json", "units": "standard"}, timeout=60)
    res.raise_for_status()
    rows = res.json()
    if not rows:
        raise ValueError(f"no hourly normals returned for {ghcn_id}")
    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "normal": pd.to_numeric(df["HLY-TEMP-NORMAL"], errors="coerce").to_numpy(),
        "p10": pd.to_numeric(df["HLY-TEMP-10PCTL"], errors="coerce").to_numpy(),
        "p90": pd.to_numeric(df["HLY-TEMP-90PCTL"], errors="coerce").to_numpy(),
    }, index=df["DATE"].str[:-6].to_numpy())  # "11-18T15:00:00" -> "11-18T15"
    if out["normal"].isna().all():
        raise ValueError("hourly normals are missing temperatures")
    return out.sort_index()


def normal_hourly(normals_hourly, times):
    """Hourly normal and 10th/90th percentile at station-local `times`:
    (normal, p10, p90) arrays. Feb 29 uses Feb 28's values."""
    keys = [t.replace(day=28).strftime("%m-%dT%H") if (t.month == 2 and t.day == 29) else t.strftime("%m-%dT%H")
            for t in pd.DatetimeIndex(times)]
    rows = normals_hourly.reindex(keys)
    return rows["normal"].to_numpy(), rows["p10"].to_numpy(), rows["p90"].to_numpy()


def departure_chips_html(labels, values, normal, sigma):
    """Row of faint chips: temperature, departure in degrees and in standard
    deviations (orange warmer, blue colder, deeper = bigger departure)."""
    cells = []
    for lab, v, n, sd in zip(labels, values, normal, sigma):
        if np.isnan(v) or np.isnan(n) or not sd:
            cells.append(f"<div style='flex:1;text-align:center;padding:6px 2px;font-size:13px;opacity:.5'><b>{lab}</b><br>—</div>")
            continue
        z = (v - n) / sd
        alpha = min(abs(z) / 2, 1) * 0.5 + 0.08
        rgb = "213,94,0" if z > 0 else "0,114,178"  # Okabe-Ito vermilion / blue
        cells.append(
            f"<div style='flex:1;text-align:center;border-radius:6px;padding:6px 2px;font-size:13px;line-height:1.35;"
            f"background:rgba({rgb},{alpha:.2f})'><b>{lab}</b><br>{v:.0f}°F<br><span style='font-size:12px;opacity:.85'>{v - n:+.0f}° · {z:+.1f}σ</span></div>")
    return "<div style='display:flex;gap:6px;margin:2px 0 6px'>" + "".join(cells) + "</div>"


def visibility_to_miles(values, unit):
    """Open-Meteo visibility (meters by default, feet with imperial units) in miles."""
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr / 5280.0 if unit in ("ft", "feet") else arr / 1609.344


def _fog_request(url, params):
    res = requests.get(url, params=params, timeout=20)
    res.raise_for_status()
    data = res.json()
    return data["hourly"], data.get("hourly_units", {})


FOG_DET_MODELS = {"GFS Operational": "gfs_seamless", "NBM Operational": "ncep_nbm_conus", "ECMWF Operational": "ecmwf_ifs025"}


@st.cache_data(ttl=3600)
def _fetch_fog_det_cached(lat, lon):
    hourly, units = _fog_request("https://api.open-meteo.com/v1/forecast", {
        "latitude": lat, "longitude": lon, "hourly": "visibility,cloud_cover_low,is_day",
        "models": ",".join(FOG_DET_MODELS.values()), "timezone": "auto", "forecast_days": 4
    })
    times = pd.to_datetime(hourly["time"])
    vis, low_cloud = pd.DataFrame({"time": times}), pd.DataFrame({"time": times})
    is_day = None
    for name, slug in FOG_DET_MODELS.items():
        key = f"visibility_{slug}"
        if key in hourly and any(v is not None for v in hourly[key]):
            vis[name] = visibility_to_miles(hourly[key], units.get(key))
        key = f"cloud_cover_low_{slug}"
        if key in hourly and any(v is not None for v in hourly[key]):
            low_cloud[name] = pd.to_numeric(hourly[key], errors="coerce")
        key = f"is_day_{slug}"
        if is_day is None and key in hourly and any(v is not None for v in hourly[key]):
            is_day = pd.DataFrame({"time": times, "is_day": pd.to_numeric(hourly[key], errors="coerce")})
    return {"vis": vis, "low_cloud": low_cloud, "is_day": is_day}


@st.cache_data(ttl=3600)
def _fetch_fog_rrfs_cached(lat, lon):
    hourly, units = _fog_request("https://api.open-meteo.com/v1/forecast", {
        "latitude": lat, "longitude": lon, "hourly": "visibility", "models": RRFS_SLUG,
        "timezone": "auto", "forecast_days": 4
    })
    if not any(v is not None for v in hourly.get("visibility", [])):
        raise ValueError("no RRFS visibility for this location")
    return {"vis": pd.DataFrame({"time": pd.to_datetime(hourly["time"]),
                                 "RRFS": visibility_to_miles(hourly["visibility"], units.get("visibility"))})}


@st.cache_data(ttl=3600)
def _fetch_fog_gefs_cached(lat, lon):
    hourly, units = _fog_request("https://ensemble-api.open-meteo.com/v1/ensemble", {
        "latitude": lat, "longitude": lon, "hourly": "visibility", "models": "gfs_seamless",
        "timezone": "auto", "forecast_days": 4
    })
    vis = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
    for k, v in hourly.items():
        if k == "visibility" or k.startswith("visibility_member"):
            if any(x is not None for x in v):
                vis[k[len("visibility_"):] if k != "visibility" else "control"] = visibility_to_miles(v, units.get(k))
    if len(vis.columns) == 1:
        raise ValueError("no GEFS member visibility for this location")
    return {"vis": vis}


def fetch_fog_inputs(lat, lon):
    """Visibility (miles), low cloud and day/night for the Hazards tab:
    {"deterministic": {...}, "RRFS": {...}, "GEFS": {...}}, plus a list of
    errors for sources that failed. Each source is its own request (RRFS can
    fail without affecting the others) and failures are never cached."""
    data, errors = {}, []
    for name, fetch in (("deterministic", _fetch_fog_det_cached), ("RRFS", _fetch_fog_rrfs_cached),
                        ("GEFS", _fetch_fog_gefs_cached)):
        try:
            data[name] = fetch(lat, lon)
        except Exception as e:
            errors.append(f"{name}: {e}")
    return data, errors

@st.cache_data(ttl=3600)
def _fetch_drizzle_source_cached(lat, lon, slug, ensemble):
    """One model's low-level profile for the drizzle/freezing-drizzle tabs:
    {"time": DatetimeIndex, "members": {id: {"t2m", "precip", "sfc_p", "t<p>", "rh<p>": array}}}.
    Temperatures in F, precipitation in inches/hr. One request per source so
    one failure doesn't take the others down; failures are never cached."""
    t_levels = DRIZZLE_ENS_T_LEVELS if ensemble else DRIZZLE_DET_LEVELS
    rh_levels = DRIZZLE_ENS_RH_LEVELS if ensemble else DRIZZLE_DET_LEVELS
    names = ["temperature_2m", "precipitation", "surface_pressure"] \
        + [f"temperature_{p}hPa" for p in t_levels] + [f"relative_humidity_{p}hPa" for p in rh_levels]
    url = "https://ensemble-api.open-meteo.com/v1/ensemble" if ensemble else "https://api.open-meteo.com/v1/forecast"
    hourly, _units = _fog_request(url, {
        "latitude": lat, "longitude": lon, "hourly": ",".join(names), "models": slug,
        "temperature_unit": "fahrenheit", "precipitation_unit": "inch", "timezone": "auto", "forecast_days": 4
    })
    # Single-model responses use bare names; ensemble members add "_memberNN".
    suffixes = [k[len("temperature_2m"):] for k in hourly if k == "temperature_2m" or k.startswith("temperature_2m_member")]
    members = {}
    for sfx in suffixes:
        def arr(name):
            vals = hourly.get(f"{name}{sfx}")
            return None if vals is None else pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy(dtype=float)
        m = {"t2m": arr("temperature_2m"), "precip": arr("precipitation"), "sfc_p": arr("surface_pressure")}
        for p in t_levels:
            m[f"t{p}"] = arr(f"temperature_{p}hPa")
        for p in rh_levels:
            m[f"rh{p}"] = arr(f"relative_humidity_{p}hPa")
        m = {k: v for k, v in m.items() if v is not None and not np.isnan(v).all()}
        if "t2m" in m and any(k.startswith("rh") for k in m):
            members[sfx.lstrip("_") or "control"] = m
    if not members:
        raise ValueError("no low-level humidity data for this location")
    return {"time": pd.to_datetime(hourly["time"]), "members": members}


def fetch_drizzle_inputs(lat, lon):
    """Low-level profiles for every source in DRIZZLE_SOURCES: ({label: source}, errors)."""
    data, errors = {}, []
    for label, slug, ensemble in DRIZZLE_SOURCES:
        try:
            data[label] = _fetch_drizzle_source_cached(lat, lon, slug, ensemble)
        except Exception as e:
            errors.append(f"{label}: {e}")
    return data, errors


# ==============================================================================
# SNOWFALL: 10:1, KUCHERA AND COBB RATIOS
# ==============================================================================

def kuchera_slr(tmax_k):
    """Kuchera snow-to-liquid ratio from the warmest temperature (K) in the
    surface-to-500 mb column. Floored at 0 for very warm columns."""
    slr = np.where(tmax_k > 271.16, 12 + 2 * (271.16 - tmax_k), 12 + (271.16 - tmax_k))
    return np.clip(slr, 0, None)


def rh_wrt_ice(rh_water, t_c):
    """Converts relative humidity over water (%) to over ice below 0 °C, using
    Magnus-type saturation vapor pressures."""
    es_water = 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))
    es_ice = 6.112 * np.exp(22.46 * t_c / (t_c + 272.62))
    return np.where(t_c < 0, rh_water * es_water / es_ice, rh_water)


def cobb_slr(profiles, sfc_p):
    """Cobb & Waldstreicher (2005) snow ratio per hour.

    profiles maps pressure (hPa) -> (temp °C, RH %, vertical velocity, height m)
    arrays for the levels a model returned. The column is resampled to
    COBB_SLICE_HPA slices (linear in log-pressure); slices below ground are
    dropped. Cloudy slices (RH w.r.t. ice >= 90%) with rising air get weight
    w * (w / w_max)^2 * thickness, and the ratio is the weighted average of
    each slice's ratio from the temperature curve. The weights are normalized,
    so vertical velocity units cancel; only its sign (positive up) matters.

    Returns an array of ratios, NaN where no slice is cloudy and rising."""
    known = sorted(profiles, reverse=True)  # surface-most level first
    x_known = -np.log(np.array(known, dtype=float))
    grid_p = np.arange(known[0], known[-1] - 1, -COBB_SLICE_HPA, dtype=float)
    x_grid = -np.log(grid_p)
    idx = np.clip(np.searchsorted(x_known, x_grid, side="right") - 1, 0, len(known) - 2)
    frac = ((x_grid - x_known[idx]) / (x_known[idx + 1] - x_known[idx]))[:, None]

    def to_grid(field):
        stack = np.vstack([profiles[p][field] for p in known])
        return stack[idx] * (1 - frac) + stack[idx + 1] * frac

    t, rh, w, z = (to_grid(i) for i in range(4))
    t_mid, rh_mid, w_mid = ((a[:-1] + a[1:]) / 2 for a in (t, rh, w))
    thickness = z[1:] - z[:-1]
    above_ground = ~(grid_p[:-1, None] > sfc_p[None, :]) if sfc_p is not None else True
    with np.errstate(invalid="ignore"):
        cloudy = above_ground & (rh_wrt_ice(rh_mid, t_mid) >= 90) & (w_mid > 0) & (thickness > 0)
    w_cloudy = np.where(cloudy, w_mid, 0.0)
    w_max = w_cloudy.max(axis=0)
    weight = np.where(cloudy, w_cloudy * (w_cloudy / np.where(w_max > 0, w_max, 1.0)) ** 2 * thickness, 0.0)
    slice_sr = np.where(cloudy, np.interp(np.nan_to_num(t_mid), COBB_SR_TEMPS_C, COBB_SR_VALUES), 0.0)
    total = weight.sum(axis=0)
    return np.where(total > 0, (slice_sr * weight).sum(axis=0) / np.where(total > 0, total, 1.0), np.nan)


def compute_snowfall(get_param):
    """Derives hourly snowfall (in) for one model run or ensemble member.

    get_param(name) returns that Open-Meteo hourly series (a list, °F for
    temperatures) or None. Open-Meteo's `snowfall` is the model's frozen
    liquid equivalent x 7 (checked live: 0.7 cm per mm in all-snow hours), so
    dividing by 7 recovers the liquid equivalent; the model still decides rain
    vs. snow, we only replace the ratio.

    Returns {var_key: array} for whichever of SNOW_DERIVED_VARS this model can
    support. Kuchera needs at least one pressure-level temperature; Cobb needs
    at least two levels with temperature, RH, vertical velocity and height.
    Models without those are left out rather than approximated. Hours with
    snow but no cloudy, rising layer have no Cobb ratio, so those hours fall
    back to Kuchera (or 10:1 without upper-air temperatures)."""
    def series(name):
        values = get_param(name)
        if values is None:
            return None
        arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
        return None if np.isnan(arr).all() else arr

    snow_api = series("snowfall")
    if snow_api is None:
        return {}
    swe = snow_api / 7.0
    derived = {"snowfall_10to1": swe * 10}

    sfc_p = series("surface_pressure")
    f_to_c = lambda f: (f - 32) * 5 / 9
    temps_c = {p: f_to_c(t) for p in SNOW_PRESSURE_LEVELS if (t := series(f"temperature_{p}hPa")) is not None}

    # Kuchera: warmest temperature from the surface to 500 mb, ignoring levels
    # that are below ground at this station.
    kuchera = None
    column = [np.where(sfc_p < p, np.nan, t) if sfc_p is not None else t
              for p, t in temps_c.items() if p >= KUCHERA_TOP_HPA]
    if any(not np.isnan(t).all() for t in column):
        t2m = series("temperature_2m")
        if t2m is not None:
            column.append(f_to_c(t2m))
        tmax_c = column[0]
        for t in column[1:]:
            tmax_c = np.fmax(tmax_c, t)  # NaN-skipping max across the column
        kuchera = kuchera_slr(tmax_c + 273.15)
        derived["snowfall_kuchera"] = swe * kuchera

    # Cobb: full profiles of temperature, RH, vertical velocity and height.
    profiles = {}
    for p, t in temps_c.items():
        rh, w, z = (series(f"{var}_{p}hPa") for var in ("relative_humidity", "vertical_velocity", "geopotential_height"))
        if rh is not None and w is not None and z is not None:
            profiles[p] = (t, rh, w, z)
    if len(profiles) >= 2:
        cobb = cobb_slr(profiles, sfc_p)
        fallback = kuchera if kuchera is not None else np.full_like(swe, 10.0)
        derived["snowfall_cobb"] = swe * np.where(np.isnan(cobb), fallback, cobb)

    # No snow is no snow, whatever the temperatures say.
    return {k: np.where(swe == 0, 0.0, v) for k, v in derived.items()}

# ==============================================================================
# LAYER 2 & 3: PROCESSING & GRAND ENSEMBLE BUILDER
# ==============================================================================

ENS_FAMILIES = [n for n in ENS_ORDER if n != "Grand Ensemble"]


def grand_family_of(col):
    """Model family of a Grand Ensemble member column ("GEFS_member07" -> "GEFS")."""
    return next((n for n in ENS_FAMILIES if col.startswith(f"{n}_")), col)


def equal_share_weights(values, families):
    """Member weights for the Grand Ensemble: every model family present in a
    row gets an equal share, split evenly across its non-missing members, so a
    64-member family doesn't outvote a 31-member one. values is a (rows,
    members) array; families gives each column's family. Missing values get
    weight 0, and a family with no data in a row drops out of that row."""
    values = np.asarray(values, dtype=float)
    valid = ~np.isnan(values)
    fams = list(dict.fromkeys(families))
    fam_idx = np.array([fams.index(f) for f in families])
    counts = np.stack([valid[:, fam_idx == i].sum(axis=1) for i in range(len(fams))], axis=1)
    per_member = np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0)
    n_present = np.maximum((counts > 0).sum(axis=1), 1)
    return valid * per_member[:, fam_idx] / n_present[:, None]


def weighted_quantile_rows(values, weights, q):
    """Row-wise weighted quantile (midpoint-CDF interpolation; with equal
    weights the median matches the ordinary median). NaN where a row has no
    weight."""
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape[0], np.nan)
    for i in range(values.shape[0]):
        keep = weights[i] > 0
        if not keep.any():
            continue
        order = np.argsort(values[i, keep])
        v, w = values[i, keep][order], weights[i, keep][order]
        cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
        out[i] = np.interp(q, cdf, v)
    return out


def grand_stats(df_members, quantiles=(0.25, 0.5, 0.75)):
    """Equal-share weighted quantiles and mean of Grand Ensemble member
    columns (rows = hours or days). Returns {q: array, "mean": array}."""
    cols = [c for c in df_members.columns if c not in ('time', 'date')]
    values = df_members[cols].to_numpy(dtype=float)
    weights = equal_share_weights(values, [grand_family_of(c) for c in cols])
    stats = {q: weighted_quantile_rows(values, weights, q) for q in quantiles}
    total = weights.sum(axis=1)
    stats["mean"] = np.where(total > 0, np.nansum(values * weights, axis=1) / np.where(total > 0, total, 1), np.nan)
    return stats


def grand_box_stats(df_daily):
    """Precomputed Plotly box stats for the Grand Ensemble (weighted
    quartiles, whiskers at the most extreme members within 1.5 IQR)."""
    stats = grand_stats(df_daily)
    q1, med, q3 = stats[0.25], stats[0.5], stats[0.75]
    iqr = q3 - q1
    values = df_daily.to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        lo = np.where(values >= (q1 - 1.5 * iqr)[:, None], values, np.inf).min(axis=1)
        hi = np.where(values <= (q3 + 1.5 * iqr)[:, None], values, -np.inf).max(axis=1)
    return dict(q1=q1, median=med, q3=q3,
                lowerfence=np.where(np.isfinite(lo), lo, np.nan), upperfence=np.where(np.isfinite(hi), hi, np.nan))


def process_ensemble_data(dict_ens, df_det, selected_var_key="temperature_2m"):
    all_member_dfs = []
    for model_name, df_m in dict_ens.items():
        cols_to_rename = {c: f"{model_name}_{c}" for c in df_m.columns if c != 'time'}
        df_renamed = df_m.rename(columns=cols_to_rename).set_index('time')
        all_member_dfs.append(df_renamed)
        
    if all_member_dfs:
        df_grand = pd.concat(all_member_dfs, axis=1).reset_index()
        dict_ens["Grand Ensemble"] = df_grand

    hourly_summaries = {}
    for name in ENS_ORDER:
        if name in dict_ens:
            df = dict_ens[name]
            member_cols = [c for c in df.columns if c != 'time']
            df_summary = pd.DataFrame({'time': df['time']})
            if name == "Grand Ensemble":
                stats = grand_stats(df)
                df_summary['median'], df_summary['mean'] = stats[0.5], stats["mean"]
                df_summary['q25'], df_summary['q75'] = stats[0.25], stats[0.75]
            else:
                df_summary['median'] = df[member_cols].median(axis=1)
                df_summary['mean'] = df[member_cols].mean(axis=1)
                df_summary['q25'] = df[member_cols].quantile(0.25, axis=1)
                df_summary['q75'] = df[member_cols].quantile(0.75, axis=1)
            hourly_summaries[name] = df_summary

    daily_ens_highs = {}
    daily_ens_lows = {}

    daily_det_highs = pd.DataFrame()
    daily_det_lows = pd.DataFrame()

    daily_agg = WEATHER_VARS[selected_var_key].get("daily_agg")
    if daily_agg is not None:
        # A "max" aggregation implies a two-sided daily range (e.g. daily
        # high/low temperature), so also compute the complementary "min"
        # for the lows.
        compute_lows = daily_agg == "max"

        df_det_daily = df_det.copy()
        if not df_det_daily.empty and 'time' in df_det_daily.columns:
            df_det_daily['date'] = df_det_daily['time'].dt.strftime('%Y-%m-%d')
            det_cols = [c for c in df_det.columns if c != 'time']

            for name in ENS_ORDER:
                if name in dict_ens:
                    df = dict_ens[name]
                    member_cols = [c for c in df.columns if c != 'time']
                    df_daily = df.copy()
                    df_daily['date'] = df_daily['time'].dt.strftime('%Y-%m-%d')
                    grouped = df_daily.groupby('date')[member_cols]
                    daily_ens_highs[name] = getattr(grouped, daily_agg)()
                    if compute_lows:
                        daily_ens_lows[name] = grouped.min()

            det_grouped = df_det_daily.groupby('date')[det_cols]
            # A model that stops mid-day (RRFS at 84 h, or anyone's last
            # partial day) would give a bogus daily high/low/total, so
            # require most of the day's hours.
            enough_hours = det_grouped.count() >= MIN_HOURS_FOR_DAILY
            daily_det_highs = getattr(det_grouped, daily_agg)().where(enough_hours)
            if compute_lows:
                daily_det_lows = det_grouped.min().where(enough_hours)
    # else: this variable has no daily_agg configured (e.g. dew point) --
    # it only gets the hourly line chart, so daily aggregates stay empty.

    return hourly_summaries, daily_ens_highs, daily_ens_lows, daily_det_highs, daily_det_lows


def get_value_at_time(df, column, target_time):
    """Nearest-hour lookup for a single deterministic model column. Returns
    (value, matched_time) or (None, None) if missing or farther than
    VERIFICATION_MAX_GAP from target_time."""
    if df is None or df.empty or column not in df.columns or 'time' not in df.columns:
        return None, None
    idx = (df['time'] - target_time).abs().idxmin()
    matched_time = df.loc[idx, 'time']
    if abs(matched_time - target_time) > VERIFICATION_MAX_GAP:
        return None, None
    value = df.loc[idx, column]
    if pd.isna(value):
        return None, None
    return value, matched_time


def get_grand_ensemble_median_at_time(dict_ens_var, target_time):
    """Pools every ensemble member column across all model families for one
    variable's dict_ens[var_key] dict and returns the Grand Ensemble median
    (equal share per family, see equal_share_weights) at the nearest hour to
    target_time. Each model family finds its own nearest
    row independently (simpler than process_ensemble_data's shared-index
    concat approach above, since we only need one pooled number here, not
    a column-addressable DataFrame). Returns (median_value, matched_time)
    or (None, None).
    """
    pooled_values, pooled_families = [], []
    matched_time = None
    for family, df_m in dict_ens_var.items():
        if df_m is None or df_m.empty or 'time' not in df_m.columns:
            continue
        member_cols = [c for c in df_m.columns if c != 'time']
        if not member_cols:
            continue
        idx = (df_m['time'] - target_time).abs().idxmin()
        row_time = df_m.loc[idx, 'time']
        if abs(row_time - target_time) > VERIFICATION_MAX_GAP:
            continue
        if matched_time is None:
            matched_time = row_time
        row_vals = [v for v in df_m.loc[idx, member_cols].tolist() if pd.notna(v)]
        if row_vals:
            pooled_values.extend(row_vals)
            pooled_families.extend([family] * len(row_vals))

    if not pooled_values:
        return None, None
    values = np.array([pooled_values], dtype=float)
    weights = equal_share_weights(values, pooled_families)
    return float(weighted_quantile_rows(values, weights, 0.5)[0]), matched_time


def build_hourly_grid(df_det, hourly_summaries, start_time, hours=48):
    """Rows = models (deterministic first, then ensemble medians in ENS_ORDER),
    columns = `hours` consecutive hours starting at start_time (naive
    station-local, floored to the hour). Models with no data in the window are
    skipped. Returns a DataFrame indexed by model name with a time-labelled
    column per hour, or an empty DataFrame if nothing falls in the window."""
    series = {}
    for det_col in [c for c in df_det.columns if c != 'time']:
        series[det_col] = pd.Series(
            df_det[det_col].values, index=df_det['time'].dt.floor('h')
        )
    for ens_name in ENS_ORDER:
        if ens_name in hourly_summaries:
            df_sum = hourly_summaries[ens_name]
            series[ens_name] = pd.Series(
                df_sum['median'].values, index=df_sum['time'].dt.floor('h')
            )

    window = pd.date_range(start=start_time, periods=hours, freq='h')
    rows = {}
    for name, s in series.items():
        s = s[~s.index.duplicated(keep='first')].reindex(window)
        if s.notna().any():
            rows[name] = s
    return pd.DataFrame(rows).T


def station_now_local(dev_mode, utc_offset_seconds):
    """"Now" in the station's local time (API timestamps are naive local;
    Dev Mode's mock data uses this machine's clock)."""
    if dev_mode:
        return pd.Timestamp(datetime.now())
    return pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)) + pd.Timedelta(seconds=utc_offset_seconds)


def run_forecast_value(df_runs, target_day, run_day, today, how):
    """One run's temperature forecast for a target day, from Previous Runs
    data (see fetch_previous_runs). The run made on run_day is read from
    column n{days between run and target}. how is "max" (daily high), "min"
    (daily low) or an int local hour. NaN outside 0..PREVIOUS_RUN_DAYS days,
    for future runs, or where data is missing."""
    n = (target_day - run_day).days
    col = f"n{n}"
    if n < 0 or n > PREVIOUS_RUN_DAYS or run_day > today or col not in df_runs.columns:
        return np.nan
    in_day = (df_runs['time'] >= target_day) & (df_runs['time'] < target_day + pd.Timedelta(days=1))
    values = df_runs.loc[in_day, ['time', col]].dropna()
    if values.empty:
        return np.nan
    if how == "max":
        return values[col].max()
    if how == "min":
        return values[col].min()
    at_hour = values.loc[values['time'].dt.hour == how, col]
    return at_hour.mean() if not at_hour.empty else np.nan


def vis_level(vis_mi):
    """Fog level per hour from a visibility series (miles): >= 3 mi none,
    mist low, fog high, dense fog dense likely. NaN stays NaN."""
    v = np.asarray(vis_mi, dtype=float)
    lvl = np.where(v < FOG_TIERS["dense_mi"], 4, np.where(v < FOG_TIERS["fog_mi"], 3,
                   np.where(v < FOG_TIERS["mist_mi"], 1, 0))).astype(float)
    return np.where(np.isnan(v), np.nan, lvl)


def pct_level(pct):
    """Fog level from a % of members/models: below the first band none, then
    low / moderate / high."""
    p = np.asarray(pct, dtype=float)
    lo, mid, hi = FOG_TIERS["pct_bands"]
    lvl = np.where(p >= hi, 3, np.where(p >= mid, 2, np.where(p >= lo, 1, 0))).astype(float)
    return np.where(np.isnan(p), np.nan, lvl)


def combine_fog_levels(rows):
    """Overall level per hour: rounded mean of the available rows, capped at
    high unless at least two rows say dense."""
    stack = np.vstack(rows)
    valid = ~np.isnan(stack)
    count = valid.sum(axis=0)
    mean = np.where(count > 0, np.nansum(np.where(valid, stack, 0), axis=0) / np.maximum(count, 1), np.nan)
    overall = np.floor(mean + 0.5)
    dense_votes = (stack == 4).sum(axis=0)
    overall = np.where(dense_votes >= 2, 4, np.minimum(overall, 3))
    return np.where(count > 0, overall, np.nan)


def fog_type(is_night, wind_mph):
    """Likely fog type for one hour: radiation (night, light wind), advection
    (5-15 mph), else unclear."""
    if np.isnan(wind_mph):
        return "Unclear"
    if is_night and wind_mph <= FOG_TIERS["radiation_wind_mph"]:
        return "Radiation"
    if FOG_TIERS["radiation_wind_mph"] < wind_mph <= FOG_TIERS["advection_wind_mph"]:
        return "Advection"
    return "Unclear"


def _on_window(df, col, window):
    """One column of a (time, ...) frame on an hourly window (NaN where missing)."""
    if df is None or col not in df.columns:
        return np.full(len(window), np.nan)
    s = pd.Series(df[col].to_numpy(dtype=float), index=pd.to_datetime(df['time']).dt.floor('h'))
    return s[~s.index.duplicated()].reindex(window).to_numpy()


def _members(df_var, param):
    """Ensemble member columns, with the control column (named after the
    variable) renamed "control" so members line up across variables."""
    return df_var.rename(columns={param: "control"}).set_index('time')


def build_fog_rows(window, fog, dict_det, dict_ens):
    """Hazards tab rows for an hourly window. Returns (rows, evidence):
    rows is an ordered {label: level array}; evidence is a dict of per-hour
    arrays/lists used for hover text and episode labels."""
    rows, ev = {}, {}
    det = fog.get("deterministic", {})
    if det.get("vis") is not None and "NBM Operational" in det["vis"].columns:
        ev["nbm_vis"] = _on_window(det["vis"], "NBM Operational", window)
        if not np.isnan(ev["nbm_vis"]).all():
            rows["NBM vis"] = vis_level(ev["nbm_vis"])
    if "GEFS" in fog:
        gv = fog["GEFS"]["vis"]
        cols = [c for c in gv.columns if c != 'time']
        mat = np.vstack([_on_window(gv, c, window) for c in cols])
        n_valid = (~np.isnan(mat)).sum(axis=0)
        ev["gefs_fog"] = (mat < FOG_TIERS["fog_mi"]).sum(axis=0)
        ev["gefs_n"] = n_valid
        with np.errstate(invalid="ignore", divide="ignore"):
            pct_fog = np.where(n_valid > 0, 100 * ev["gefs_fog"] / n_valid, np.nan)
            pct_dense = np.where(n_valid > 0, 100 * (mat < FOG_TIERS["dense_mi"]).sum(axis=0) / n_valid, np.nan)
        lvl = pct_level(pct_fog)
        rows["GEFS %"] = np.where(pct_dense >= FOG_TIERS["dense_member_pct"], 4, lvl)
    if "RRFS" in fog:
        ev["rrfs_vis"] = _on_window(fog["RRFS"]["vis"], "RRFS", window)
        if not np.isnan(ev["rrfs_vis"]).all():
            rows["RRFS vis"] = vis_level(ev["rrfs_vis"])

    # Signals: dew-point spread and wind from every model/member that has
    # both (WeatherNext has no dew point), each source an equal share.
    t_all, td_all, w_all = dict_det.get("temperature_2m"), dict_det.get("dew_point_2m"), dict_det.get("wind_speed_10m")
    spreads, winds, fams = [], [], []
    if t_all is not None and td_all is not None and w_all is not None:
        for model in [c for c in t_all.columns if c != 'time']:
            if model in td_all.columns and model in w_all.columns:
                spreads.append(_on_window(t_all, model, window) - _on_window(td_all, model, window))
                winds.append(_on_window(w_all, model, window))
                fams.append(model)
    for fam in [n for n in ENS_ORDER if n != "Grand Ensemble"]:
        frames = [dict_ens.get(v, {}).get(fam) for v in ("temperature_2m", "dew_point_2m", "wind_speed_10m")]
        if any(f is None for f in frames):
            continue
        t, td, w = (_members(f, p) for f, p in zip(frames, ("temperature_2m", "dew_point_2m", "wind_speed_10m")))
        for m in sorted(set(t.columns) & set(td.columns) & set(w.columns)):
            to_w = lambda df: _on_window(df.reset_index(), m, window)
            spreads.append(to_w(t) - to_w(td))
            winds.append(to_w(w))
            fams.append(fam)
    if spreads:
        spread_mat, wind_mat = np.vstack(spreads).T, np.vstack(winds).T  # (hours, sources)
        with np.errstate(invalid="ignore"):
            ok = (spread_mat <= FOG_TIERS["spread_f"]) & (wind_mat <= FOG_TIERS["wind_mph"])
        ok = np.where(np.isnan(spread_mat) | np.isnan(wind_mat), np.nan, ok.astype(float))
        weights = equal_share_weights(ok, fams)
        total = weights.sum(axis=1)
        pct = np.where(total > 0, 100 * np.nansum(np.nan_to_num(ok) * weights, axis=1) / np.where(total > 0, total, 1), np.nan)
        rows["Signals"] = pct_level(pct)
        ev["signals_pct"] = pct
        with warnings_ignored():
            ev["spread"] = np.nanmedian(spread_mat, axis=1)
            ev["wind"] = np.nanmedian(wind_mat, axis=1)
    if det.get("low_cloud") is not None:
        lc_cols = [c for c in det["low_cloud"].columns if c != 'time']
        if lc_cols:
            lc = np.vstack([_on_window(det["low_cloud"], c, window) for c in lc_cols])
            with warnings_ignored():
                ev["low_cloud"] = np.nanmedian(lc, axis=0)
    night = _on_window(det.get("is_day"), "is_day", window) == 0 if det.get("is_day") is not None \
        else np.array([(t.hour >= 19 or t.hour < 7) for t in window])
    ev["night"] = night

    if rows:
        rows["Overall"] = combine_fog_levels(list(rows.values()))
        wind = ev.get("wind", np.full(len(window), np.nan))
        ev["type"] = [fog_type(bool(n), w) if lvl >= 1 else "" for n, w, lvl in zip(night, wind, rows["Overall"])]
    return rows, ev


class warnings_ignored:
    """Context manager: silence RuntimeWarnings from all-NaN reductions."""
    def __enter__(self):
        import warnings
        self._cm = warnings.catch_warnings()
        self._cm.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)
    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)


def fog_episodes(window, overall, types):
    """Contiguous hours at Low or above: [(start, end, peak level, dominant type)]."""
    episodes, start = [], None
    for i, lvl in enumerate(list(overall) + [np.nan]):
        if not np.isnan(lvl) and lvl >= 1:
            start = i if start is None else start
        elif start is not None:
            seg = range(start, i)
            peak = int(np.nanmax([overall[j] for j in seg]))
            # The episode's type is the one covering most of its hours;
            # otherwise "Unclear" (e.g. mostly daytime with a short night tail).
            kinds = [types[j] or "Unclear" for j in seg]
            kind = max(set(kinds), key=kinds.count)
            episodes.append((window[start], window[i - 1], peak, kind))
            start = None
    return episodes


def _hourly_on(times, values, window):
    """A series on an hourly window (NaN where missing); None -> all NaN."""
    if values is None:
        return np.full(len(window), np.nan)
    s = pd.Series(np.asarray(values, dtype=float), index=pd.to_datetime(times).floor('h'))
    return s[~s.index.duplicated()].reindex(window).to_numpy()


def drizzle_flags(member, times, window):
    """Drizzle and freezing-drizzle yes/no per hour for one model or member.
    Returns dict of arrays: drizzle, fzdz (1.0 / 0.0 / NaN where the profile
    is missing), plus evidence: sat, deep, top_c (top of the saturated low
    layer, C), t2m, precip. Levels below ground (pressure above surface
    pressure) are ignored."""
    T = DRIZZLE_TIERS
    g = lambda key: _hourly_on(times, member.get(key), window)
    n = len(window)
    sfc_p, t2m, precip = g("sfc_p"), g("t2m"), g("precip")
    sat, have_low = np.zeros(n, bool), np.zeros(n, bool)
    top_c = np.full(n, np.nan)
    for p in sorted((int(k[2:]) for k in member if k.startswith("rh")), reverse=True):
        if p < T["low_top_hpa"]:
            continue
        rh, t = g(f"rh{p}"), g(f"t{p}")
        with np.errstate(invalid="ignore"):
            valid = ~np.isnan(rh) & ~np.isnan(t) & ~(sfc_p < p)
            hit = valid & (rh >= T["sat_rh"])
        have_low |= valid
        sat |= hit
        top_c = np.where(hit, (t - 32) * 5 / 9, top_c)  # higher levels overwrite lower ones
    deep, have_aloft = np.zeros(n, bool), np.zeros(n, bool)
    for p in (800, 700):
        if f"rh{p}" not in member:
            continue
        rh = g(f"rh{p}")
        with np.errstate(invalid="ignore"):
            valid = ~np.isnan(rh) & ~(sfc_p < p)
            deep |= valid & (rh >= T["aloft_rh"])
        have_aloft |= valid
    known = have_low & have_aloft & ~np.isnan(precip)
    with np.errstate(invalid="ignore"):
        dz = sat & ~deep & (precip <= T["max_precip_in"])
        fz = dz & (t2m <= T["fz_sfc_f"]) & (top_c > T["fz_top_c"])
    known_fz = known & ~np.isnan(t2m)
    return {"drizzle": np.where(known, dz.astype(float), np.nan),
            "fzdz": np.where(known_fz, fz.astype(float), np.nan),
            "sat": sat, "deep": deep, "top_c": top_c, "t2m": t2m, "precip": precip}


def build_drizzle_rows(window, inputs, kind):
    """Hazards rows for drizzle (kind "drizzle") or freezing drizzle ("fzdz").
    Deterministic rows are yes (Moderate) / no; ensemble rows are the share of
    members saying yes, banded like the fog GEFS row. Returns (rows, hover)
    with hover[label] a list of per-hour text."""
    rows, hover = {}, {}
    fmt = lambda v, spec, suffix="": "—" if v is None or np.isnan(v) else f"{v:{spec}}{suffix}"
    for label, _slug, is_ens in DRIZZLE_SOURCES:
        src = inputs.get(label)
        if not src:
            continue
        flags = [drizzle_flags(m, src["time"], window) for m in src["members"].values()]
        mat = np.vstack([f[kind] for f in flags])  # (members, hours)
        n_valid = (~np.isnan(mat)).sum(axis=0)
        if not n_valid.any():
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            yes = np.nansum(mat, axis=0)
            pct = np.where(n_valid > 0, 100 * yes / np.maximum(n_valid, 1), np.nan)
        with warnings_ignored():
            med = lambda key: np.nanmedian(np.vstack([f[key] for f in flags]), axis=0)
            t2m_med, top_med, precip_med = med("t2m"), med("top_c"), med("precip")
        if is_ens:
            rows[f"{label} %"] = pct_level(pct)
        else:
            rows[label] = np.where(np.isnan(pct), np.nan, np.where(pct >= 50, 2.0, 0.0))
        key = f"{label} %" if is_ens else label
        text = []
        for i, t in enumerate(window):
            lvl = rows[key][i]
            parts = [f"<b>{key}</b> · {t:%a %b %d %I %p}", f"Level: {'—' if np.isnan(lvl) else DRIZZLE_LEVELS[int(lvl)]}"]
            if is_ens and n_valid[i]:
                parts.append(f"{int(yes[i])} of {int(n_valid[i])} members")
            else:
                f0 = flags[0]
                parts.append(f"Saturated low layer: {'yes' if f0['sat'][i] else 'no'}; deep moisture aloft: {'yes' if f0['deep'][i] else 'no'}")
            parts.append(f"2 m temp: {fmt(t2m_med[i], '.0f', '°F')}")
            parts.append(f"Saturated-layer top: {fmt(top_med[i] * 9 / 5 + 32 if not np.isnan(top_med[i]) else np.nan, '.0f', '°F')}")
            parts.append(f"Precip: {fmt(precip_med[i], '.3f', ' in/h')}")
            text.append("<br>".join(parts))
        hover[key] = text
    if rows:
        rows["Overall"] = combine_fog_levels(list(rows.values()))
        text = []
        for i, t in enumerate(window):
            lvl = rows["Overall"][i]
            text.append(f"<b>Overall</b> · {t:%a %b %d %I %p}<br>Level: {'—' if np.isnan(lvl) else DRIZZLE_LEVELS[int(lvl)]}"
                        f"<br>Average of the rows above")
        hover["Overall"] = text
    return rows, hover


def grand_trend_value(df_daily, target_day, run_day, how):
    """Grand Ensemble daily high ("max") or low ("min") forecast for a target
    day from one archived run: {0.25, 0.5, 0.75} quantiles with equal share
    per model family (grand_stats). None if the run isn't archived."""
    rows = df_daily[(df_daily["run_day"] == run_day) & (df_daily["target_day"] == target_day)]
    if rows.empty:
        return None
    field = "tmax" if how == "max" else "tmin"
    members = pd.DataFrame([rows[field].to_numpy()], columns=(rows["family"] + "_" + rows["member"].astype(str)).tolist())
    stats = grand_stats(members)
    return {q: float(stats[q][0]) for q in (0.25, 0.5, 0.75)}


def format_value_with_unit(val, unit, signed=False):
    """°F gets no space (matches Tab 3's existing '74.0°F' convention);
    mph gets a space, since there's no existing precedent for it appearing
    inline with a value."""
    sep = "" if unit.startswith("°") else " "
    fmt = "{:+.1f}" if signed else "{:.1f}"
    return f"{fmt.format(val)}{sep}{unit}"

# ==============================================================================
# STREAMLIT UI & SIDEBAR
# ==============================================================================

st.title("🌤️ Multi-Model Ensemble Weather Consensus Dashboard")
st.markdown("Comparing deterministic operational runs against **197 probabilistic ensemble members** across European (ECMWF/EPS/AIFS), American (GFS/GEFS/NBM/RRFS), and AI (Google WeatherNext 2) forecasting systems.")

with st.sidebar:
    # Outside the form so switching views is instant.
    view = st.radio("View", ["Forecast", "Hazards"], horizontal=True, key="view")
    st.header("⚙️ Location & Forecast Controls")
    
    with st.form("forecast_controls_form"):
        dev_mode = st.toggle("🛠️ Dev Mode (Use Offline Mock Data)", value=False)
        loc_mode = st.radio("Location Mode", ["Airport Code", "Manual Lat/Lon"], horizontal=True)
        
        if loc_mode == "Airport Code":
            airport_input = st.text_input("Airport Code (ICAO / IATA)", value="KCMH").strip().upper()
            
            # ---> AIRPORT CODE STRIPPER GOES HERE <---
            # Strips the 'K' if the user entered 4 letters (e.g. KCMH -> CMH)
            station_id = airport_input[1:] if len(airport_input) == 4 and airport_input.startswith("K") else airport_input
            
            auto_lat, auto_lon, station_name = get_coordinates_from_airport(airport_input)
            lat, lon = auto_lat, auto_lon
            st.caption(f"📍 **{station_name}** ({lat:.2f}°, {lon:.2f}°)")
        else:
            lat = st.number_input("Latitude", value=39.97, step=0.01, format="%.2f")
            lon = st.number_input("Longitude", value=-83.00, step=0.01, format="%.2f")
            
            # Default to CMH if they are entering coordinates manually
            station_id = "CMH" 
            
        forecast_days = st.slider("Forecast Horizon (Days)", min_value=3, max_value=14, value=7)
        
        if view == "Forecast":
            selected_var_key = st.selectbox(
                "Forecast Parameter",
                options=list(WEATHER_VARS.keys()),
                format_func=lambda x: WEATHER_VARS[x]["label"],
                key="selected_var_key"
            )
        else:
            # Hazards view: no parameter choice; temperature keeps the shared
            # fetch/processing path happy (fog uses temp, dew point and wind).
            selected_var_key = "temperature_2m"
        
        submitted = st.form_submit_button("🚀 Load / Update Forecast", use_container_width=True)

    var_cfg = WEATHER_VARS[selected_var_key]

    # NCEI normals underlay: air temperature only, and only for stations that
    # have normals. Outside the form so it toggles instantly.
    normals_df, normals_hourly, show_normals = None, None, False
    if view == "Forecast" and selected_var_key == "temperature_2m":
        if dev_mode:
            normals_df, normals_hourly = generate_mock_normals(), generate_mock_hourly_normals()
        elif loc_mode == "Airport Code" and airport_input in NORMALS_STATIONS:
            normals_df = True  # placeholder until the toggle is on (fetched below)
        if normals_df is not None:
            show_normals = st.toggle("Show NCEI normals (1991–2020)", value=False, key="show_normals",
                                     help="Faint normal temperature with ±1σ / ±2σ bands behind the charts.")
            if show_normals and not dev_mode:
                try:
                    normals_df = fetch_daily_normals(NORMALS_STATIONS[airport_input])
                except Exception as e:
                    st.caption(f"NCEI normals unavailable: {e}")
                    show_normals = False
                else:
                    try:
                        normals_hourly = fetch_hourly_normals(NORMALS_STATIONS[airport_input])
                    except Exception as e:
                        st.caption(f"NCEI hourly normals unavailable (daily charts still use normals): {e}")
    if not show_normals:
        normals_df, normals_hourly = None, None
    
    st.divider()
    if st.button("🔄 Force Clear Cache & Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    tick_interval_hours = 12
    if view == "Forecast":
        st.divider()
        tick_interval_label = st.radio(
            "Hourly Chart Tick Interval",
            ["6 Hours", "12 Hours"],
            index=1,
            horizontal=True
        )
        tick_interval_hours = 6 if tick_interval_label == "6 Hours" else 12

# ==============================================================================
# ROUTING: DEV MODE VS LIVE FETCH
# ==============================================================================

if dev_mode:
    dict_det, dict_ens, det_run_cycles, run_cycles = generate_mock_data(days=forecast_days)
    fetch_time = "OFFLINE DEV MODE"
    det_err = None
    utc_offset_seconds = 0  # Dev Mode's mock times use this machine's clock
    prev_runs, prev_runs_err = generate_mock_previous_runs(), None
    archive_daily = pd.DataFrame()
    fog_inputs, fog_errors = generate_mock_fog_inputs()
    drizzle_inputs, drizzle_errors = generate_mock_drizzle_inputs()
else:
    with st.spinner("Fetching multi-model ensemble payloads..."):
        dict_det, fetch_time, det_run_cycles, det_err, utc_offset_seconds = fetch_deterministic_data(lat, lon, days=forecast_days)
        dict_ens, run_cycles, ens_errs = fetch_ensemble_data(lat, lon, days=forecast_days)
        prev_runs, prev_runs_err = fetch_previous_runs(lat, lon)
        archive_station = airport_input if loc_mode == "Airport Code" else None
        archive_daily = (fetch_archive_daily(archive_station, utc_offset_seconds)
                         if archive_station in ARCHIVE_STATIONS else pd.DataFrame())
        fog_inputs, fog_errors = fetch_fog_inputs(lat, lon)
        # Only the Hazards view uses these (6 small requests, cached for an hour).
        drizzle_inputs, drizzle_errors = fetch_drizzle_inputs(lat, lon) if view == "Hazards" else ({}, [])

if not dev_mode and (det_err or dict_det[selected_var_key].empty):
    st.error(f"⚠️ Unable to fetch weather data from Open-Meteo. Details: `{det_err}`")
    st.info("💡 **Tip:** Switch on '🛠️ Dev Mode' in the sidebar to test layout & charts offline without hitting API rate limits.")
    st.stop()

# Display Model Run Cycles in Sidebar
with st.sidebar:
    st.caption(f"🕒 **Last API Fetch:** {fetch_time}")
    st.markdown("**Model Run Cycles Loaded:**")
    
    # Deterministic Operational Run Cycles
    for model_name, cycle_str in det_run_cycles.items():
        st.text(f"• {model_name:<18}: {cycle_str}")
        
    # Ensemble Run Cycles
    for model_name, cycle_str in run_cycles.items():
        st.text(f"• {model_name:<18}: {cycle_str}")

    st.caption(f"Build: {get_deployed_commit()}")

# Select Payload based on Dropdown
df_det_active = dict_det[selected_var_key].copy()
dict_ens_active = dict_ens[selected_var_key].copy()

# Process Data dynamically
hourly_summaries, daily_ens_highs, daily_ens_lows, daily_det_highs, daily_det_lows = process_ensemble_data(
    dict_ens_active,
    df_det_active,
    selected_var_key=selected_var_key
)

# ==============================================================================
# FORECAST VIEW: VERIFICATION PANEL + FORECAST TABS
# ==============================================================================

if view == "Forecast":
    # ==============================================================================
    # MODEL VERIFICATION: RIGHT NOW VS. LATEST METAR
    # ==============================================================================

    if dev_mode:
        with st.expander("✅ Model Verification vs. METAR", expanded=False):
            st.info("Model verification is unavailable in Dev Mode, which makes zero live API calls.")
    elif loc_mode != "Airport Code":
        with st.expander("✅ Model Verification vs. METAR", expanded=False):
            st.info("Model verification requires Airport Code location mode (a station is needed for a live observation).")
    elif var_cfg.get("verify", True) is False:
        with st.expander(f"✅ Model Verification vs. METAR — {station_name}", expanded=False):
            st.info(f"Verification isn't available for {var_cfg['label']}.")
    else:
        with st.expander(f"✅ Model Verification vs. METAR — {var_cfg['label']} at {station_name}", expanded=False):
            obs, obs_err = get_current_conditions(airport_input)
            if obs is None:
                st.info(f"No current METAR available for {airport_input} right now. ({obs_err})")
            else:
                obs_dt_utc = datetime.fromtimestamp(obs["obs_time"], tz=timezone.utc)
                obs_dt_local = (obs_dt_utc + pd.Timedelta(seconds=utc_offset_seconds)).replace(tzinfo=None)
                temp_c, dewp_c, wspd_kt, wgst_kt = obs["temp_c"], obs["dewp_c"], obs["wspd_kt"], obs["wgst_kt"]

                actual_val = None
                if selected_var_key == "temperature_2m" and temp_c is not None:
                    actual_val = temp_c * 9 / 5 + 32
                elif selected_var_key == "apparent_temperature" and None not in (temp_c, dewp_c, wspd_kt):
                    actual_val = calculate_apparent_temperature(temp_c, dewp_c, wspd_kt * 0.514444) * 9 / 5 + 32
                elif selected_var_key == "dew_point_2m" and dewp_c is not None:
                    actual_val = dewp_c * 9 / 5 + 32
                elif selected_var_key == "wind_speed_10m" and wspd_kt is not None:
                    actual_val = wspd_kt * 1.15078
                elif selected_var_key == "wind_gusts_10m" and wgst_kt is not None:
                    actual_val = wgst_kt * 1.15078

                if actual_val is None:
                    st.info(f"Current METAR for {airport_input} doesn't report {var_cfg['label']} right now.")
                else:
                    cols = st.columns(len(DET_MODEL_SLUGS) + 2)
                    cols[0].metric("METAR Actual", format_value_with_unit(actual_val, var_cfg["unit"]))

                    for i, model_name in enumerate(DET_MODEL_SLUGS, start=1):
                        val, _ = get_value_at_time(dict_det.get(selected_var_key, pd.DataFrame()), model_name, obs_dt_local)
                        if val is None:
                            cols[i].metric(model_name, "N/A")
                        else:
                            cols[i].metric(
                                model_name,
                                format_value_with_unit(val, var_cfg["unit"]),
                                format_value_with_unit(val - actual_val, var_cfg["unit"], signed=True),
                                delta_color="off"
                            )

                    ens_median, _ = get_grand_ensemble_median_at_time(dict_ens.get(selected_var_key, {}), obs_dt_local)
                    if ens_median is None:
                        cols[-1].metric("Grand Ensemble", "N/A")
                    else:
                        cols[-1].metric(
                            "Grand Ensemble",
                            format_value_with_unit(ens_median, var_cfg["unit"]),
                            format_value_with_unit(ens_median - actual_val, var_cfg["unit"], signed=True),
                            delta_color="off"
                        )

                    st.caption(f"METAR observed at {obs_dt_local.strftime('%a %I:%M %p')} local")

    # ==============================================================================
    # LAYER 4: PLOTLY VISUALIZATIONS
    # ==============================================================================

    st.caption(
        "Grand Ensemble gives each ensemble family (EPS, AIFS, GEFS, WeatherNext) an equal share, "
        "regardless of how many members it has."
    )
    if selected_var_key in SNOW_DERIVED_VARS:
        st.caption(
            "Snowfall uses each model's own snow liquid equivalent (Open-Meteo's 7:1 snowfall ÷ 7). "
            "10:1 multiplies it by 10; Kuchera uses a ratio from the warmest temperature between the surface and 500 mb; "
            "Cobb weights a temperature-based ratio by upward motion in saturated layers, on 25 mb slices interpolated from "
            "Open-Meteo's pressure levels, falling back to Kuchera in hours with no saturated rising layer."
        )
        if selected_var_key != "snowfall_10to1":
            needs = "upper-air temperatures" if selected_var_key == "snowfall_kuchera" else "full upper-air profiles"
            missing = [c for c in dict_det["snowfall_10to1"].columns if c != 'time' and c not in df_det_active.columns]
            missing += [n for n in dict_ens["snowfall_10to1"] if n not in dict_ens[selected_var_key]]
            if missing:
                st.caption(f"Not in this view (no {needs}): {', '.join(missing)}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Hourly Time-Series", "📊 Daily Distribution Spread", "📋 Consensus Summary Table", "🔲 Hourly Grid", "📉 Trends"])

    # --- TAB 1: HOURLY TIME-SERIES ---
    with tab1:
        fig_hourly = go.Figure()

        if normals_hourly is not None:
            # NCEI hourly normal underlay (measured 1991-2020 hourly normals, not
            # interpolated): a faint 10th-90th percentile band first so every
            # line draws over it.
            n_times = pd.DatetimeIndex(sorted(df_det_active['time'].dt.floor('h').unique()))
            n_norm, n_p10, n_p90 = normal_hourly(normals_hourly, n_times)
            ens_med = None
            if "Grand Ensemble" in hourly_summaries:
                gs = hourly_summaries["Grand Ensemble"]
                ens_med = pd.Series(gs['median'].values, index=gs['time'].dt.floor('h')).pipe(
                    lambda x: x[~x.index.duplicated()]).reindex(n_times).to_numpy()
            fig_hourly.add_trace(go.Scatter(x=n_times, y=n_p90, mode='lines', line=dict(width=0),
                                            hoverinfo='skip', showlegend=False, legendgroup="normals"))
            fig_hourly.add_trace(go.Scatter(x=n_times, y=n_p10, mode='lines', line=dict(width=0), fill='tonexty',
                                            fillcolor="rgba(128,128,128,0.14)", name="Normal 10th–90th percentile",
                                            hoverinfo='skip', legendgroup="normals"))
            dep = (ens_med - n_norm) if ens_med is not None else np.full(len(n_times), np.nan)
            fig_hourly.add_trace(go.Scatter(
                x=n_times, y=n_norm, mode='lines', name="NCEI normal", legendgroup="normals",
                line=dict(color="rgba(128,128,128,0.85)", width=1.5, dash='dot'),
                customdata=np.c_[dep, n_p10, n_p90],
                hovertemplate="<b>NCEI normal</b>: %{y:.0f} °F (10th–90th: %{customdata[1]:.0f}–%{customdata[2]:.0f})"
                              "<br>Grand Ens. median vs. normal: %{customdata[0]:+.0f}°<extra></extra>"
            ))
    
        # Render all available operational deterministic models
        for det_col in [c for c in df_det_active.columns if c != 'time']:
            color = model_color(det_col)
            fig_hourly.add_trace(go.Scatter(
                x=df_det_active['time'],
                y=df_det_active[det_col],
                mode='lines',
                name=det_col,
                line=dict(color=color, width=3),
                hovertemplate=f"%{{x|%a %b %d, %I:%M %p}}<br><b>{det_col}</b>: %{{y:.2f}} {var_cfg['unit']}<extra></extra>"
            ))
            
        for ens_name in ENS_ORDER:
            if ens_name in hourly_summaries:
                df_sum = hourly_summaries[ens_name]
                color = model_color(ens_name)
                width = 3.5 if ens_name == "Grand Ensemble" else 2
                dash = 'solid' if ens_name == "Grand Ensemble" else 'dash'
            
                fig_hourly.add_trace(go.Scatter(
                    x=df_sum['time'],
                    y=df_sum['median'],
                    mode='lines',
                    name=f"{ens_name} (Median)",
                    line=dict(color=color, width=width, dash=dash),
                    hovertemplate=f"%{{x|%a %b %d, %I:%M %p}}<br><b>{ens_name} (Median)</b>: %{{y:.2f}} {var_cfg['unit']}<extra></extra>"
                ))

        fig_hourly.update_layout(
            title=dict(text=f"Hourly {var_cfg['label']} Trajectory ({var_cfg['unit']})", font=dict(size=18)),
            xaxis=dict(
                title="Date / Time (Local)",
                tickformat="%a %m/%d %I%p",
                tickangle=-45,
                automargin=True,
                dtick=tick_interval_hours * 60 * 60 * 1000,
                **AXIS_STYLE
            ),
            yaxis=dict(title=f"{var_cfg['label']} ({var_cfg['unit']})", **AXIS_STYLE),
            hovermode="x unified",
            height=650,
            margin=dict(b=160),
            legend=dict(orientation="h", yanchor="bottom", y=-0.55, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_hourly, use_container_width=True)


    # --- TAB 2: DAILY DISTRIBUTION SPREAD ---
    def grand_box_trace(df_daily, dates, name, color, showlegend):
        """Grand Ensemble box from equal-share weighted stats (Plotly can't weight
        raw points, so the quartiles/whiskers are precomputed; no outlier dots)."""
        df_days = df_daily.loc[[d for d in dates if d in df_daily.index]]
        stats = grand_box_stats(df_days)
        return go.Box(
            x=list(df_days.index), name=name, marker_color=color, line=dict(width=2),
            whiskerwidth=0.8, legendgroup=name, offsetgroup=name, showlegend=showlegend, hoverinfo="y+name",
            **stats
        )


    with tab2:
        if var_cfg.get("daily_agg") is None:
            st.info(f"Daily distribution isn't tracked for {var_cfg['label']}.")
        else:
            dates = list(daily_det_highs.index)
            # Display labels add the day of week (e.g. "Tue 08/25"); the underlying
            # trace x-values stay as ISO date strings for correct grouping/sorting.
            date_labels = {d: pd.to_datetime(d).strftime('%a %m/%d') for d in dates}

            def add_daily_traces(fig, daily_ens, daily_det, row, showlegend):
                """One row of ensemble boxes + deterministic diamonds. Every trace
                is in its model's legendgroup, so a legend click or double-click
                hides/isolates that model in every row."""
                for ens_name in ENS_ORDER:
                    if ens_name not in daily_ens:
                        continue
                    df_m = daily_ens[ens_name]
                    color = model_color(ens_name)
                    if ens_name == "Grand Ensemble":
                        fig.add_trace(grand_box_trace(df_m, dates, ens_name, color, showlegend=showlegend), row=row, col=1)
                        continue
                    x_vals, y_vals = [], []
                    for date_str in dates:
                        if date_str in df_m.index:
                            vals = df_m.loc[date_str].values
                            x_vals.extend([date_str] * len(vals))
                            y_vals.extend(vals)
                    fig.add_trace(go.Box(
                        x=x_vals, y=y_vals, name=ens_name, marker_color=color, line=dict(width=2),
                        whiskerwidth=0.8, boxpoints='outliers', legendgroup=ens_name, offsetgroup=ens_name,
                        showlegend=showlegend, hoverinfo="y+name"
                    ), row=row, col=1)
                for det_col in daily_det.columns:
                    fig.add_trace(go.Scatter(
                        x=daily_det.index, y=daily_det[det_col], mode='markers', name=det_col,
                        legendgroup=det_col, showlegend=showlegend,
                        marker=dict(color=model_color(det_col), size=11, symbol='diamond', line=dict(width=1.5, color='black')),
                        hovertemplate=f"<b>{det_col}</b><br>%{{y:.1f}} " + var_cfg['unit'] + "<extra></extra>"
                    ), row=row, col=1)

            # Highs and (for two-sided variables like temperature) lows share one
            # figure, so the legend controls both charts.
            chart_a_title = var_cfg["chart_title"]
            has_lows = not daily_det_lows.empty
            titles = [f"{chart_a_title} ({var_cfg['unit']})"]
            y_titles = [f"{var_cfg['label']} ({var_cfg['unit']})"]
            if has_lows:
                titles.append(f"{chart_a_title.replace('High', 'Low')} ({var_cfg['unit']})")
                y_titles.append(f"Low {var_cfg['label']} ({var_cfg['unit']})")
            fig_daily = make_subplots(rows=len(titles), cols=1, subplot_titles=titles, vertical_spacing=0.12)
            add_daily_traces(fig_daily, daily_ens_highs, daily_det_highs, row=1, showlegend=True)
            if has_lows:
                add_daily_traces(fig_daily, daily_ens_lows, daily_det_lows, row=2, showlegend=False)

            for row, y_title in enumerate(y_titles, start=1):
                fig_daily.update_xaxes(
                    title_text="Calendar Day", type='category', categoryorder='array', categoryarray=dates,
                    tickvals=dates, ticktext=[date_labels[d] for d in dates], row=row, col=1, **AXIS_STYLE
                )
                fig_daily.update_yaxes(title_text=y_title, zeroline=False, row=row, col=1, **AXIS_STYLE)
            fig_daily.update_annotations(font=dict(size=18))  # subplot titles
            fig_daily.update_layout(
                boxmode='group',
                boxgap=0.3,
                boxgroupgap=0.08,
                height=1000 if has_lows else 520,
                hovermode="closest",
                margin=dict(t=110),
                legend=dict(orientation="h", yanchor="bottom", y=1.04 if has_lows else 1.08, xanchor="center", x=0.5)
            )
            if normals_df is not None:
                n_hi, n_lo, sd_hi, sd_lo = normal_daily(normals_df, dates)
                for row, (mid, sd, word) in enumerate([(n_hi, sd_hi, "high")] + ([(n_lo, sd_lo, "low")] if has_lows else []), start=1):
                    for k, alpha in ((2, 0.07), (1, 0.12)):
                        for i in range(len(dates)):
                            fig_daily.add_shape(type="rect", x0=i - 0.5, x1=i + 0.5, y0=mid[i] - k * sd[i], y1=mid[i] + k * sd[i],
                                                fillcolor=f"rgba(128,128,128,{alpha})", line=dict(width=0), layer="below", row=row, col=1)
                    for i in range(len(dates)):
                        fig_daily.add_shape(type="line", x0=i - 0.5, x1=i + 0.5, y0=mid[i], y1=mid[i], layer="below",
                                            line=dict(color="rgba(128,128,128,0.85)", width=1.5, dash="dot"), row=row, col=1)
                # Legend entries for the bands (shapes don't get any).
                for k, alpha in ((1, 0.12), (2, 0.07)):
                    fig_daily.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"Normal ±{k}σ",
                                                   marker=dict(size=12, symbol="square", color=f"rgba(128,128,128,{alpha + 0.15})")), row=1, col=1)
                fig_daily.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="NCEI normal",
                                               line=dict(color="rgba(128,128,128,0.85)", width=1.5, dash="dot")), row=1, col=1)
            st.plotly_chart(fig_daily, use_container_width=True)
            if normals_df is not None and "Grand Ensemble" in daily_ens_highs:
                def ge_median(daily_ens):
                    df_g = daily_ens["Grand Ensemble"]
                    return np.array([float(grand_stats(df_g.loc[[d]])[0.5][0]) if d in df_g.index else np.nan for d in dates])
                st.markdown("**Departure from normal — Grand Ensemble median** (orange warmer, blue colder; σ = NCEI standard deviation for that date)")
                st.caption("Highs")
                st.markdown(departure_chips_html([date_labels[d] for d in dates], ge_median(daily_ens_highs), n_hi, sd_hi), unsafe_allow_html=True)
                if has_lows and "Grand Ensemble" in daily_ens_lows:
                    st.caption("Lows")
                    st.markdown(departure_chips_html([date_labels[d] for d in dates], ge_median(daily_ens_lows), n_lo, sd_lo), unsafe_allow_html=True)

    # --- TAB 3: SUMMARY DATA TABLE & CSV DOWNLOAD ---
    with tab3:
        if var_cfg.get("daily_agg") is None:
            st.info(f"A daily summary table isn't available for {var_cfg['label']}.")
        else:
            summary_rows = []
            dates = list(daily_det_highs.index)
    
            for d in dates:
                date_obj = pd.to_datetime(d)
                row = {"Date": date_obj.strftime("%a %b %d, %Y")}
        
                # Populate deterministic run data dynamically
                for det_col in daily_det_highs.columns:
                    if selected_var_key in ("temperature_2m", "apparent_temperature") and det_col in daily_det_lows.columns and d in daily_det_lows.index:
                        low_val = daily_det_lows.loc[d, det_col]
                        high_val = daily_det_highs.loc[d, det_col]
                        row[f"{det_col} (L/H)"] = ("—" if pd.isna(low_val) or pd.isna(high_val)
                                                   else f"{low_val:.1f}° / {high_val:.1f}°F")
                    else:
                        v = daily_det_highs.loc[d, det_col]
                        row[det_col] = "—" if pd.isna(v) else round(v, 2)
                
                for ens_name in ["EPS", "AIFS", "GEFS", "WeatherNext"]:
                    if ens_name in daily_ens_highs and d in daily_ens_highs[ens_name].index:
                        high_vals = daily_ens_highs[ens_name].loc[d].values
                        if selected_var_key in ("temperature_2m", "apparent_temperature") and ens_name in daily_ens_lows and d in daily_ens_lows[ens_name].index:
                            low_vals = daily_ens_lows[ens_name].loc[d].values
                            row[f"{ens_name} Med (L/H)"] = f"{np.median(low_vals):.1f}° / {np.median(high_vals):.1f}°F"
                        else:
                            row[f"{ens_name} Median"] = round(float(np.median(high_vals)), 2)
                
                if "Grand Ensemble" in daily_ens_highs and d in daily_ens_highs["Grand Ensemble"].index:
                    # Equal share per model family (see equal_share_weights).
                    g_hi = {k: float(v[0]) for k, v in grand_stats(daily_ens_highs["Grand Ensemble"].loc[[d]]).items()}
                    if selected_var_key in ("temperature_2m", "apparent_temperature") and "Grand Ensemble" in daily_ens_lows and d in daily_ens_lows["Grand Ensemble"].index:
                        g_lo_med = float(grand_stats(daily_ens_lows["Grand Ensemble"].loc[[d]])[0.5][0])
                        row["Grand Ens Med (L/H)"] = f"{g_lo_med:.1f}° / {g_hi[0.5]:.1f}°F"
                        row["High IQR Spread"] = f"{g_hi[0.25]:.1f}° to {g_hi[0.75]:.1f}°F"
                    else:
                        row["Grand Ens Median"] = round(g_hi[0.5], 2)
                        row["Consensus IQR"] = f"{g_hi[0.25]:.2f} to {g_hi[0.75]:.2f} {var_cfg['unit']}"
            
                summary_rows.append(row)
        
            df_summary_table = pd.DataFrame(summary_rows)
    
            st.subheader(f"Daily Consensus Summary Table ({var_cfg['unit']})")
            st.dataframe(df_summary_table, use_container_width=True, hide_index=True)
    
            csv = df_summary_table.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Daily Summary CSV",
                data=csv,
                file_name=f"weather_consensus_{selected_var_key}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )


    # --- TAB 4: HOURLY GRID ---
    with tab4:
        GRID_HOURS = 48
        grid_cfg = var_cfg.get("grid")
        unit = var_cfg["unit"]

        if grid_cfg is None:
            st.info(f"An hourly grid isn't available for {var_cfg['label']}.")
        else:
            st.subheader(f"Hourly {var_cfg['label']} Grid ({unit})")
            st.caption(
                f"Next {GRID_HOURS} hours, station-local time. Ensemble families show the median. "
                "One color scale is shared across all models."
                + (" Values are a running total from the current hour." if grid_cfg.get("cumulative") else "")
            )

            color_by_departure = False
            if normals_hourly is not None:
                color_by_departure = st.radio(
                    "Color cells by", ["Temperature", "Departure from normal (°F)"], horizontal=True, key="grid_color_mode"
                ) != "Temperature"
            grid_start = station_now_local(dev_mode, utc_offset_seconds).floor('h')

            df_grid = build_hourly_grid(df_det_active, hourly_summaries, grid_start, hours=GRID_HOURS)
            if grid_cfg.get("cumulative") and not df_grid.empty:
                df_grid = df_grid.cumsum(axis=1)
            decimals = grid_cfg.get("decimals", 0)
            cell_font = dict(size=10 if decimals else 11)

            def format_cell(v):
                """Short labels that fit the narrow hourly cells: whole numbers
                from 10 up, "0" for zero, and no leading zero below 1 (".25")."""
                if np.isnan(v):
                    return ""
                if decimals == 0 or abs(v) >= 10:
                    return f"{v:.0f}"
                text = f"{v:.{decimals}f}"
                if float(text) == 0:
                    return "0"
                return text.replace("0.", ".", 1) if abs(v) < 1 else text

            def format_cells(values):
                return [[format_cell(v) for v in row] for row in np.asarray(values, dtype=float)]

            if df_grid.empty:
                st.info(f"No {var_cfg['label'].lower()} data is available for the next {GRID_HOURS} hours.")
            else:
                times = list(df_grid.columns)
                models = list(df_grid.index)
                z = df_grid.values.astype(float)

                # Models that returned nothing for this variable (e.g. AIFS and
                # WeatherNext for gusts): say so rather than dropping them silently.
                missing = [n for n in list(DET_MODEL_SLUGS) + ENS_ORDER if n not in models]
                if missing and not dev_mode:
                    st.caption(f"No data for this variable from: {', '.join(missing)}")

                # make_subplots lets the spread row use its own (neutral) color
                # scale without sharing the variable's one.
                show_spread = len(models) > 1
                fig_grid = make_subplots(
                    rows=2 if show_spread else 1, cols=1, shared_xaxes=True,
                    row_heights=[len(models) + (1 if normals_hourly is not None else 0), 1.2] if show_spread else [1],
                    vertical_spacing=0.04
                )
                # Optional NCEI hourly-normal row on top (same temperature scale), and an
                # optional departure-from-normal (°F) coloring.
                z_show, y_show, hover_dep = z, models, None
                z_color, scale, zmin, zmax, cbar_title = z, grid_cfg["colorscale"], None, None, unit
                if normals_hourly is not None:
                    n_row, n_p10, n_p90 = normal_hourly(normals_hourly, pd.DatetimeIndex(times))
                    z_show = np.vstack([n_row, z])
                    y_show = ["NCEI normal"] + models
                    z_color = z_show
                    dev = z_show - n_row

                    def dep_note(i, j):
                        if i == 0:
                            return f"<br>10th–90th percentile: {n_p10[j]:.0f}–{n_p90[j]:.0f} {unit}"
                        outside = " (outside the normal 10th–90th range)" if (z_show[i, j] < n_p10[j] or z_show[i, j] > n_p90[j]) else ""
                        return f"<br>Normal {n_row[j]:.0f} · {dev[i, j]:+.0f}°{outside}"
                    hover_dep = [[f"{y_show[i]}<br>{times[j]:%a %b %d, %I:%M %p}<br><b>{z_show[i, j]:.0f} {unit}</b>" + dep_note(i, j)
                                  for j in range(len(times))] for i in range(len(y_show))]
                    if color_by_departure:
                        z_color = dev
                        scale = [[0, "#2166AC"], [0.25, "#92C5DE"], [0.5, "#F7F7F7"], [0.75, "#F4A582"], [1, "#B2182B"]]
                        lim = max(10.0, float(np.ceil(np.nanmax(np.abs(dev)) / 5) * 5))
                        zmin, zmax, cbar_title = -lim, lim, "°F vs. normal"
                if zmin is None:
                    zmin = grid_cfg.get("zmin", float(np.nanmin(z_color)))
                    zmax = max(float(np.nanmax(z_color)), float(grid_cfg.get("zmax_floor", -np.inf)))
                fig_grid.add_trace(go.Heatmap(
                    x=times, y=y_show, z=z_color, text=format_cells(z_show),
                    colorscale=scale, zmin=zmin, zmax=zmax,
                    xgap=1, ygap=1,
                    texttemplate="%{text}", textfont=cell_font,
                    colorbar=dict(title=cbar_title, len=0.8, y=1, yanchor="top"),
                    **({"customdata": hover_dep, "hovertemplate": "%{customdata}<extra></extra>"} if hover_dep is not None else
                       {"hovertemplate": f"%{{y}}<br>%{{x|%a %b %d, %I:%M %p}}<br><b>%{{z:.{decimals + 1}f}} {unit}</b><extra></extra>"})
                ), row=1, col=1)

                if show_spread:
                    spread = np.nanmax(z, axis=0) - np.nanmin(z, axis=0)
                    fig_grid.add_trace(go.Heatmap(
                        x=times, y=["Spread (max − min)"], z=[spread], text=format_cells([spread]),
                        colorscale="Greys", zmin=0,
                        zmax=max(float(np.nanmax(spread)), float(grid_cfg["spread_floor"])),
                        xgap=1, ygap=1,
                        texttemplate="%{text}", textfont=cell_font,
                        showscale=False,
                        hovertemplate=f"Model spread<br>%{{x|%a %b %d, %I:%M %p}}<br><b>%{{z:.{decimals + 1}f}} {unit}</b><extra></extra>"
                    ), row=2, col=1)

                fig_grid.update_yaxes(autorange="reversed", showline=False, showgrid=False, tickfont=dict(size=12))
                fig_grid.update_xaxes(
                    tickformat="%a %m/%d %I%p", tickangle=-45, automargin=True,
                    dtick=3 * 60 * 60 * 1000, showgrid=False, showline=False
                )
                if show_spread:
                    fig_grid.update_xaxes(showticklabels=False, row=1, col=1)

                # Thin divider at each local midnight (cell edge = half an hour before).
                for t in times:
                    if t.hour == 0 and t != times[0]:
                        edge = t - pd.Timedelta(minutes=30)
                        fig_grid.add_shape(
                            type="line", xref="x", yref="paper", x0=edge, x1=edge, y0=0, y1=1,
                            line=dict(color="rgba(128, 128, 128, 0.9)", width=2)
                        )

                fig_grid.update_layout(
                    height=110 + 42 * (len(models) + (1 if show_spread else 0) + (1 if normals_hourly is not None else 0)) + 90,
                    margin=dict(l=10, r=10, t=20, b=10)
                )
                st.plotly_chart(fig_grid, use_container_width=True)


    # --- TAB 5: TRENDS (TEMPERATURE) ---
    with tab5:
        st.subheader("Temperature Trends")
        st.caption(
            f"How each model's temperature forecast has changed over the last {PREVIOUS_RUN_DAYS} days of runs. "
            "Always air temperature, regardless of the sidebar variable. A run's forecast for a day comes from "
            "Open-Meteo's Previous Runs data: the forecast made N days before each hour, N = days between run and target."
        )
        if not prev_runs:
            st.info(f"Previous-run data isn't available right now. ({prev_runs_err})")
        else:
            today = station_now_local(dev_mode, utc_offset_seconds).normalize()
            day_lbl = lambda d: d.strftime("%a %m/%d")
            target_days = [today + pd.Timedelta(days=i) for i in range(7)]
            run_days = [today - pd.Timedelta(days=i) for i in range(PREVIOUS_RUN_DAYS - 1, -1, -1)]
            has_ens_trend = not archive_daily.empty

            c1, c2 = st.columns([2, 1])
            value_mode = c1.radio("Value", ["High", "Low", "Specific hour"], horizontal=True, key="trend_value")
            how = {"High": "max", "Low": "min"}.get(value_mode)
            if how is None:
                how = c2.selectbox("Local hour", list(range(24)), index=16, key="trend_hour",
                                   format_func=lambda hr: pd.Timestamp(2000, 1, 1, hr).strftime("%I %p").lstrip("0"))
            value_lbl = {"max": "HIGH", "min": "LOW"}.get(how) or pd.Timestamp(2000, 1, 1, how).strftime("%I %p").lstrip("0")
            ens_ok = has_ens_trend and how in ("max", "min")

            # A. Trend grid: target days x runs, with change vs. the previous run.
            model_options = list(prev_runs) + (["Grand Ensemble"] if ens_ok else [])
            trend_model = st.selectbox("Model", model_options, key="trend_model")

            def trend_cell(target_day, run_day):
                if trend_model == "Grand Ensemble":
                    stats = grand_trend_value(archive_daily, target_day, run_day, how)
                    return np.nan if stats is None else stats[0.5]
                return run_forecast_value(prev_runs[trend_model], target_day, run_day, today, how)

            z = np.array([[trend_cell(d, r) for r in run_days] for d in target_days], dtype=float)
            delta = z[:, -1] - z[:, -2]
            arrows = ["" if np.isnan(v) else ("▲" if v > 0.5 else "▼" if v < -0.5 else "•") + f"{abs(v):.0f}" for v in delta]
            fig_trend = make_subplots(rows=1, cols=2, column_widths=[0.88, 0.12], shared_yaxes=True, horizontal_spacing=0.01)
            fig_trend.add_trace(go.Heatmap(
                z=z, x=[day_lbl(r) for r in run_days], y=[day_lbl(d) for d in target_days],
                text=[["" if np.isnan(v) else f"{v:.0f}" for v in row] for row in z], texttemplate="%{text}",
                textfont=dict(size=14), colorscale=WEATHER_VARS["temperature_2m"]["grid"]["colorscale"],
                xgap=2, ygap=2, colorbar=dict(title="°F", x=1.02),
                hovertemplate="Target %{y}<br>Run %{x}<br><b>%{z:.1f}°F</b><extra></extra>"
            ), 1, 1)
            fig_trend.add_trace(go.Heatmap(
                z=delta[:, None], x=["Δ vs prev run"], y=[day_lbl(d) for d in target_days],
                text=[[a] for a in arrows], texttemplate="%{text}", textfont=dict(size=14),
                colorscale="RdBu_r", zmid=0, zmin=-6, zmax=6, showscale=False, xgap=2, ygap=2,
                hovertemplate="%{y}: %{z:+.1f}°F vs previous run<extra></extra>"
            ), 1, 2)
            fig_trend.update_yaxes(autorange="reversed", title_text="Target day", row=1, col=1)
            fig_trend.update_xaxes(side="top", row=1, col=1)
            fig_trend.update_xaxes(side="top", row=1, col=2)
            fig_trend.update_layout(
                title=dict(text=f"{trend_model} forecast {value_lbl} (°F) by run date (columns, latest at right)", font=dict(size=18)),
                height=440, margin=dict(t=110, l=10, r=10, b=10)
            )
            st.plotly_chart(fig_trend, use_container_width=True)

            st.divider()

            # B. dprog/dt: one target day, every model's forecast by run date.
            dprog_days = [today + pd.Timedelta(days=i) for i in range(1, 8)]
            target = st.selectbox("Target day", dprog_days, index=2, format_func=day_lbl, key="trend_target")
            dprog_runs = [target - pd.Timedelta(days=n) for n in range(PREVIOUS_RUN_DAYS, -1, -1)
                          if target - pd.Timedelta(days=n) <= today]
            x_lbls = [day_lbl(r) for r in dprog_runs]
            fig_dprog = go.Figure()
            if ens_ok:
                ens = [grand_trend_value(archive_daily, target, r, how) for r in dprog_runs]
                pts = [(x, e) for x, e in zip(x_lbls, ens) if e is not None]
                if pts:
                    xs = [x for x, _ in pts]
                    fig_dprog.add_trace(go.Scatter(
                        x=xs + xs[::-1], y=[e[0.75] for _, e in pts] + [e[0.25] for _, e in pts][::-1],
                        fill="toself", line=dict(width=0), mode="lines",
                        fillcolor="rgba(255, 255, 255, 0.15)" if is_dark_theme() else "rgba(0, 0, 0, 0.15)",
                        name="Grand Ensemble IQR", hoverinfo="skip"
                    ))
                    fig_dprog.add_trace(go.Scatter(
                        x=xs, y=[e[0.5] for _, e in pts], mode="lines+markers", name="Grand Ensemble (Median)",
                        line=dict(color=model_color("Grand Ensemble"), width=3, dash="dash"),
                        hovertemplate="Grand Ensemble<br>Run %{x}: <b>%{y:.0f}°F</b><extra></extra>"
                    ))
            for model_name, df_runs in prev_runs.items():
                fig_dprog.add_trace(go.Scatter(
                    x=x_lbls, y=[run_forecast_value(df_runs, target, r, today, how) for r in dprog_runs],
                    mode="lines+markers", name=model_name,
                    line=dict(color=model_color(model_name), width=3), marker=dict(size=9),
                    hovertemplate=f"{model_name}<br>Run %{{x}}: <b>%{{y:.0f}}°F</b><extra></extra>"
                ))
            fig_dprog.update_layout(
                title=dict(text=f"Forecast {value_lbl} for {target:%a %m/%d} by run date", font=dict(size=18)),
                # Pin run-date order: the ensemble trace can start later than the
                # deterministic ones, which would otherwise append early runs at the end.
                xaxis=dict(title="Run date", type="category", categoryorder="array", categoryarray=x_lbls, **AXIS_STYLE),
                yaxis=dict(title="Temperature (°F)", **AXIS_STYLE),
                height=450, hovermode="x unified", legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_dprog, use_container_width=True)
            if has_ens_trend and not ens_ok:
                st.caption("The Grand Ensemble trend covers daily highs and lows only (the archive doesn't keep hourly members).")
            elif not has_ens_trend and not dev_mode and loc_mode == "Airport Code" and airport_input in ARCHIVE_STATIONS:
                st.caption("The Grand Ensemble trend will appear once the daily forecast archive has a run or two of history.")


# ==============================================================================
# HAZARDS VIEW (one tab per hazard: fog, drizzle, freezing drizzle)
# ==============================================================================

if view == "Hazards":
    # --- HAZARDS VIEW: FOG TAB ---
    fog_tab, drizzle_tab, fzdz_tab = st.tabs(["🌫️ Fog", "🌦️ Drizzle", "🧊 Freezing Drizzle"])
    with fog_tab:
        st.subheader(f"Fog Potential — next {HAZARD_HOURS} h")
        st.caption(
            f"Visibility tiers: mist < {FOG_TIERS['mist_mi']:g} mi, fog < {FOG_TIERS['fog_mi']:g} mi, "
            f"dense < {FOG_TIERS['dense_mi']:g} mi (NWS Dense Fog Advisory). "
            f"GEFS % = share of members with fog; Signals = share of models/members with dew-point spread ≤ "
            f"{FOG_TIERS['spread_f']:g}°F and wind ≤ {FOG_TIERS['wind_mph']:g} mph (each model family an equal share). "
            "Overall = average of the rows; dense only when two rows agree."
        )
        window = pd.date_range(station_now_local(dev_mode, utc_offset_seconds).floor('h'), periods=HAZARD_HOURS, freq='h')
        fog_rows, fog_ev = build_fog_rows(window, fog_inputs, dict_det, dict_ens)
        missing_rows = [r for r in ("NBM vis", "GEFS %", "RRFS vis", "Signals") if r not in fog_rows]
        if not fog_rows:
            st.info("No fog inputs are available for this location right now.")
        else:
            labels = list(fog_rows)
            z = np.vstack([fog_rows[r] for r in labels])

            def fmt(v, spec, suffix=""):
                return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:{spec}}{suffix}"

            hover = []
            for r in labels:
                row_text = []
                for i, t in enumerate(window):
                    lvl = fog_rows[r][i]
                    parts = [f"<b>{r}</b> · {t:%a %b %d %I %p}", f"Level: {'—' if np.isnan(lvl) else FOG_LEVELS[int(lvl)]}"]
                    if "spread" in fog_ev:
                        parts.append(f"Dew-point spread (median): {fmt(fog_ev['spread'][i], '.1f', '°F')}")
                        parts.append(f"Wind (median): {fmt(fog_ev['wind'][i], '.0f', ' mph')}")
                    if "low_cloud" in fog_ev:
                        parts.append(f"Low cloud (median): {fmt(fog_ev['low_cloud'][i], '.0f', '%')}")
                    if "nbm_vis" in fog_ev:
                        parts.append(f"NBM vis: {fmt(fog_ev['nbm_vis'][i], '.2f', ' mi')}")
                    if "rrfs_vis" in fog_ev:
                        parts.append(f"RRFS vis: {fmt(fog_ev['rrfs_vis'][i], '.2f', ' mi')}")
                    if "gefs_n" in fog_ev and fog_ev["gefs_n"][i]:
                        parts.append(f"GEFS: {int(fog_ev['gefs_fog'][i])} of {int(fog_ev['gefs_n'][i])} members < {FOG_TIERS['fog_mi']:g} mi")
                    if fog_ev.get("type") and fog_ev["type"][i]:
                        parts.append(f"Likely type: {fog_ev['type'][i]}")
                    row_text.append("<br>".join(parts))
                hover.append(row_text)
            cell_text = [[(fog_ev["type"][i][:1] if r == "Overall" and fog_ev.get("type") and fog_ev["type"][i] not in ("", "Unclear") else "")
                          for i in range(len(window))] for r in labels]

            # Discrete 5-step scale: none (faint gray) -> dense (deep red).
            steps = ["rgba(128, 128, 128, 0.12)", "#FFE08A", "#FDAE61", "#F46D43", "#B2182B"]
            colorscale = []
            for k, c in enumerate(steps):
                colorscale += [(k / len(steps), c), ((k + 1) / len(steps), c)]
            fig_fog = go.Figure(go.Heatmap(
                z=z, x=window, y=labels, zmin=-0.5, zmax=4.5, colorscale=colorscale, xgap=1, ygap=3,
                customdata=hover, hovertemplate="%{customdata}<extra></extra>",
                text=cell_text, texttemplate="%{text}", textfont=dict(size=11),
                colorbar=dict(tickvals=list(range(5)), ticktext=FOG_LEVELS, title="Fog")
            ))
            for t in window:
                if t.hour == 0 and t != window[0]:
                    edge = t - pd.Timedelta(minutes=30)
                    fig_fog.add_shape(type="line", xref="x", yref="paper", x0=edge, x1=edge, y0=0, y1=1,
                                      line=dict(color="rgba(128, 128, 128, 0.9)", width=2))
            fig_fog.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=13))
            fig_fog.update_xaxes(tickformat="%a %m/%d %I%p", dtick=6 * 3600 * 1000, tickangle=-45, showgrid=False, automargin=True)
            fig_fog.update_layout(height=140 + 55 * len(labels), margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_fog, use_container_width=True)
            st.caption("Overall row letters: R = likely radiation fog (night, light wind), A = likely advection fog (5–15 mph).")

            episodes = fog_episodes(window, fog_rows["Overall"], fog_ev.get("type", [""] * len(window)))
            if episodes:
                st.markdown("**Fog potential periods**")
                for start, end, peak, kind in episodes:
                    span = f"{start:%a %I %p}" if start == end else f"{start:%a %I %p} – {end:%a %I %p}"
                    st.markdown(f"- {span}: peak **{FOG_LEVELS[peak]}**" + (f", likely {kind.lower()}" if kind != "Unclear" else ""))
            else:
                st.markdown("No fog potential flagged in the next 72 hours.")

        notes = []
        if missing_rows:
            notes.append(f"Not shown (no data for this location/time): {', '.join(missing_rows)}.")
        if fog_errors:
            notes.append("Source errors: " + "; ".join(fog_errors))
        if notes:
            st.caption(" ".join(notes))
        st.caption(
            "These models are 3–25 km grids and fog is notoriously hard to forecast; treat a flag as a prompt to "
            "check soundings and observations, not a forecast."
        )

    # --- HAZARDS VIEW: DRIZZLE AND FREEZING DRIZZLE TABS ---
    def render_drizzle_tab(kind):
        T = DRIZZLE_TIERS
        freezing = kind == "fzdz"
        st.subheader(f"{'Freezing Drizzle' if freezing else 'Drizzle'} Potential — next {HAZARD_HOURS} h")
        criteria = (
            f"Each model/member says yes when: RH ≥ {T['sat_rh']:g}% somewhere between the surface and {T['low_top_hpa']} mb, "
            f"RH < {T['aloft_rh']:g}% at 800/700 mb (no deep moisture, so no ice), and model precip ≤ {T['max_precip_in']:g} in/h"
        )
        if freezing:
            criteria += f", plus a 2 m temperature ≤ {T['fz_sfc_f']:g}°F and a saturated-layer top warmer than {T['fz_top_c']:g}°C (liquid cloud)"
        st.caption(
            criteria + ". Deterministic rows (GFS, ECMWF, RRFS) are yes/no; EPS/AIFS/GEFS rows are the share of members "
            "saying yes. Overall = average of the rows."
        )
        window = pd.date_range(station_now_local(dev_mode, utc_offset_seconds).floor('h'), periods=HAZARD_HOURS, freq='h')
        rows, hover_by_row = build_drizzle_rows(window, drizzle_inputs, kind)
        if not rows:
            st.info("No low-level profile data is available for this location right now.")
        else:
            labels = list(rows)
            z = np.vstack([rows[r] for r in labels])
            hover = [hover_by_row[r] for r in labels]
            steps = (["rgba(128, 128, 128, 0.12)", "#F3C9DD", "#DF65B0", "#980043"] if freezing
                     else ["rgba(128, 128, 128, 0.12)", "#C6DBEF", "#6BAED6", "#08519C"])
            colorscale = []
            for k, c in enumerate(steps):
                colorscale += [(k / len(steps), c), ((k + 1) / len(steps), c)]
            fig = go.Figure(go.Heatmap(
                z=z, x=window, y=labels, zmin=-0.5, zmax=3.5, colorscale=colorscale, xgap=1, ygap=3,
                customdata=hover, hovertemplate="%{customdata}<extra></extra>",
                colorbar=dict(tickvals=list(range(4)), ticktext=DRIZZLE_LEVELS, title="Freezing drizzle" if freezing else "Drizzle")
            ))
            for t in window:
                if t.hour == 0 and t != window[0]:
                    edge = t - pd.Timedelta(minutes=30)
                    fig.add_shape(type="line", xref="x", yref="paper", x0=edge, x1=edge, y0=0, y1=1,
                                  line=dict(color="rgba(128, 128, 128, 0.9)", width=2))
            fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=13))
            fig.update_xaxes(tickformat="%a %m/%d %I%p", dtick=6 * 3600 * 1000, tickangle=-45, showgrid=False, automargin=True)
            fig.update_layout(height=140 + 55 * len(labels), margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

            episodes = fog_episodes(window, rows["Overall"], [""] * len(window))
            word = "freezing drizzle" if freezing else "drizzle"
            if episodes:
                st.markdown(f"**{word.capitalize()} potential periods**")
                for start, end, peak, _kind in episodes:
                    span = f"{start:%a %I %p}" if start == end else f"{start:%a %I %p} – {end:%a %I %p}"
                    st.markdown(f"- {span}: peak **{DRIZZLE_LEVELS[peak]}**")
            else:
                st.markdown(f"No {word} potential flagged in the next 72 hours.")
        notes = []
        if drizzle_errors:
            notes.append("Source errors: " + "; ".join(drizzle_errors))
        if not dev_mode:
            notes.append("RRFS and GFS resolve 975/950/900 mb; ECMWF and the ensembles only 1000/925/850/700 mb, "
                         "so their shallow layers are crude (RRFS is also limited to its first ~3.5 days).")
        if notes:
            st.caption(" ".join(notes))
        st.caption(
            "Models handle drizzle poorly (it forms in layers they barely resolve and the QPF is tiny). "
            "Treat a flag as a prompt to check soundings, ceilings and observations, not a forecast."
        )

    with drizzle_tab:
        render_drizzle_tab("drizzle")
    with fzdz_tab:
        render_drizzle_tab("fzdz")
