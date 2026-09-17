import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone
import time
import re

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
        "chart_title": "Daily High Temperature Spread"
    },
    "dew_point_2m": {
        "label": "Dew Point",
        "unit": "°F",
        "hourly_param": "dew_point_2m"
        # No daily_agg/chart_title: hourly line chart only, no daily
        # box-and-whisker or high/low tracking for this variable.
    },
    "precipitation": {
        "label": "Total Precipitation",
        "unit": "in",
        "hourly_param": "precipitation",
        "daily_agg": "sum",          # 'sum' for total daily rainfall
        "chart_title": "Daily Total Precipitation Spread"
    },
    "wind_speed_10m": {
        "label": "Wind Speed",
        "unit": "mph",
        "hourly_param": "wind_speed_10m",
        "daily_agg": "max",          # 'max' for daily peak wind
        "chart_title": "Daily High Wind Speed Spread"
    },
    "wind_gusts_10m": {
        "label": "Wind Gust",
        "unit": "mph",
        "hourly_param": "wind_gusts_10m",
        "daily_agg": "max",          # 'max' for daily peak gust
        "chart_title": "Daily High Wind Gust Spread"
    }
}

# Color Vision Deficiency (CVD) Safe Palette (Okabe-Ito Inspired)
MODEL_CONFIG = {
    # Deterministic Operational Runs
    "ECMWF Operational": {"color": "#D55E00"},  # Vermilion
    "GFS Operational":   {"color": "#CC79A7"},  # Purple/Magenta
    "NBM Operational":   {"color": "#000000"},
    "Deterministic":     {"color": "#D55E00"},
    
    # Ensemble Model Families
    "EPS":            {"color": "#0072B2"},  # Blue
    "AIFS":           {"color": "#CC79A7"},  # Purple
    "GEFS":           {"color": "#E69F00"},  # Amber
    "WeatherNext":    {"color": "#009E73"},  # Teal
    "Grand Ensemble": {"color": "#888888"}   # Mid-Gray
}

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
    "NBM Operational": "ncep_nbm_conus"
}

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

    return {
        "ECMWF Operational": ecmwf_str,
        "GFS Operational": gfs_str,
        "NBM Operational": nbm_str,
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

    dict_det = {
        "temperature_2m": pd.DataFrame({
            'time': dates,
            'ECMWF Operational': base_temp + 1.0,
            'GFS Operational': base_temp - 1.0
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

    dict_ens = {var_key: {} for var_key in WEATHER_VARS}
    run_cycles = {}

    # AIFS and WeatherNext are AI-based models without gust parameterization,
    # so they're left out of wind_gusts_10m here to mirror the live API's
    # exclusion behavior for Dev Mode testing.
    gust_capable_nicknames = {"EPS", "GEFS"}

    for nickname in ["EPS", "AIFS", "GEFS", "WeatherNext"]:
        df_t = pd.DataFrame({'time': dates})
        df_p = pd.DataFrame({'time': dates})
        df_w = pd.DataFrame({'time': dates})
        df_g = pd.DataFrame({'time': dates})
        df_d = pd.DataFrame({'time': dates})

        # Add synthetic ensemble member variation
        for m in range(1, 31):
            df_t[f"member_{m}"] = base_temp + np.random.normal(0, 2.5, len(dates))
            df_p[f"member_{m}"] = np.maximum(0, np.random.normal(0, 0.05, len(dates)))
            df_w[f"member_{m}"] = np.maximum(0, base_wind + np.random.normal(0, 2.0, len(dates)))
            df_g[f"member_{m}"] = np.maximum(0, base_gust + np.random.normal(0, 3.0, len(dates)))
            df_d[f"member_{m}"] = base_dew_point + np.random.normal(0, 2.0, len(dates))

        dict_ens["temperature_2m"][nickname] = df_t
        dict_ens["precipitation"][nickname] = df_p
        dict_ens["wind_speed_10m"][nickname] = df_w
        if nickname in gust_capable_nicknames:
            dict_ens["wind_gusts_10m"][nickname] = df_g
        dict_ens["dew_point_2m"][nickname] = df_d
        run_cycles[nickname] = "DEV-MOCK 00Z"

    det_run_cycles = {
        "ECMWF Operational": "DEV-MOCK 00Z",
        "GFS Operational": "DEV-MOCK 00Z"
    }

    return dict_det, dict_ens, det_run_cycles, run_cycles

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

# ==============================================================================
# LAYER 1: LIVE DATA INGESTION (CACHED FOR 15 MINUTES)
# ==============================================================================

@st.cache_data(ttl=900)
def fetch_deterministic_data(lat, lon, days=7):
    """Fetches explicit operational deterministic runs for ECMWF IFS, GFS, and NBM."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(cfg["hourly_param"] for cfg in WEATHER_VARS.values()),
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
        "NBM Operational": all_cycles["NBM Operational"]
    }

    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()

        hourly = data["hourly"]
        dict_det = {}
        for var_key, cfg in WEATHER_VARS.items():
            hourly_param = cfg["hourly_param"]
            df_var = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
            for model_name, slug in DET_MODEL_SLUGS.items():
                key = f"{hourly_param}_{slug}"
                if key in hourly:
                    df_var[model_name] = hourly[key]
                # else: this model doesn't return this variable -- leave it
                # out entirely so it's excluded from charts/legends instead
                # of showing an empty/all-NaN series.
            dict_det[var_key] = df_var

        fetch_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return dict_det, fetch_time, det_run_cycles, None
    except Exception as e:
        return {var_key: pd.DataFrame() for var_key in WEATHER_VARS}, "", {}, str(e)

@st.cache_data(ttl=900)
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
            "hourly": ",".join(cfg["hourly_param"] for cfg in WEATHER_VARS.values()),
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
                    df_m_var[col] = hourly[k]

                dict_ens[var_key][nickname] = df_m_var

            time.sleep(0.15)

        except Exception as e:
            errors.append(f"{nickname}: {str(e)}")
            continue

    return dict_ens, run_cycles, errors

# ==============================================================================
# LAYER 2 & 3: PROCESSING & GRAND ENSEMBLE BUILDER
# ==============================================================================

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
            daily_det_highs = getattr(det_grouped, daily_agg)()
            if compute_lows:
                daily_det_lows = det_grouped.min()
    # else: this variable has no daily_agg configured (e.g. dew point) --
    # it only gets the hourly line chart, so daily aggregates stay empty.

    return hourly_summaries, daily_ens_highs, daily_ens_lows, daily_det_highs, daily_det_lows

# ==============================================================================
# STREAMLIT UI & SIDEBAR
# ==============================================================================

st.title("🌤️ Multi-Model Ensemble Weather Consensus Dashboard")
st.markdown("Comparing deterministic operational runs against **197 probabilistic ensemble members** across European (ECMWF/EPS/AIFS), American (GFS/GEFS), and AI (Google WeatherNext 2) forecasting systems.")

with st.sidebar:
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
        
        selected_var_key = st.selectbox(
            "Forecast Parameter",
            options=list(WEATHER_VARS.keys()),
            format_func=lambda x: WEATHER_VARS[x]["label"]
        )
        
        submitted = st.form_submit_button("🚀 Load / Update Forecast", use_container_width=True)

    var_cfg = WEATHER_VARS[selected_var_key]
    
    st.divider()
    if st.button("🔄 Force Clear Cache & Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

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
else:
    with st.spinner("Fetching multi-model ensemble payloads..."):
        dict_det, fetch_time, det_run_cycles, det_err = fetch_deterministic_data(lat, lon, days=forecast_days)
        dict_ens, run_cycles, ens_errs = fetch_ensemble_data(lat, lon, days=forecast_days)

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
# LAYER 4: PLOTLY VISUALIZATIONS
# ==============================================================================

tab1, tab2, tab3 = st.tabs(["📈 Hourly Time-Series", "📊 Daily Distribution Spread", "📋 Consensus Summary Table"])

# --- TAB 1: HOURLY TIME-SERIES ---
with tab1:
    fig_hourly = go.Figure()
    
    # Render all available operational deterministic models
    det_colors = {
        "ECMWF Operational": "#D55E00", 
        "GFS Operational": "#CC79A7", 
        "NBM Operational": "#000000",
        "Deterministic": "#D55E00"
    }
    for det_col in [c for c in df_det_active.columns if c != 'time']:
        color = det_colors.get(det_col, "#D55E00")
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
            color = MODEL_CONFIG[ens_name]["color"]
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
with tab2:
    if var_cfg.get("daily_agg") is None:
        st.info(f"Daily distribution isn't tracked for {var_cfg['label']}.")
    else:
        dates = list(daily_det_highs.index)
        # Display labels add the day of week (e.g. "Tue 08/25"); the underlying
        # trace x-values stay as ISO date strings for correct grouping/sorting.
        date_labels = {d: pd.to_datetime(d).strftime('%a %m/%d') for d in dates}

        axis_style = AXIS_STYLE
    
        # 1. HIGH TEMPERATURE / PRECIPITATION CHART
        fig_daily_high = go.Figure()
    
        for ens_name in ENS_ORDER:
            if ens_name in daily_ens_highs:
                df_m = daily_ens_highs[ens_name]
                color = MODEL_CONFIG[ens_name]["color"]
            
                x_vals = []
                y_vals = []
                for date_str in dates:
                    if date_str in df_m.index:
                        vals = df_m.loc[date_str].values
                        x_vals.extend([date_str] * len(vals))
                        y_vals.extend(vals)
                    
                fig_daily_high.add_trace(go.Box(
                    x=x_vals,
                    y=y_vals,
                    name=ens_name,
                    marker_color=color,
                    line=dict(width=2),
                    whiskerwidth=0.8,
                    boxpoints='outliers',
                    legendgroup=ens_name,
                    hoverinfo="y+name"
                ))

        det_colors = {"ECMWF Operational": "#D55E00", "GFS Operational": "#CC79A7", "NBM Operational": "#000000", "Deterministic": "#D55E00"}
        for det_col in daily_det_highs.columns:
            color = det_colors.get(det_col, "#D55E00")
            fig_daily_high.add_trace(go.Scatter(
                x=daily_det_highs.index,
                y=daily_det_highs[det_col],
                mode='markers',
                name=det_col,
                marker=dict(color=color, size=11, symbol='diamond', line=dict(width=1.5, color='black')),
                hovertemplate=f"<b>{det_col}</b><br>%{{y:.1f}} " + var_cfg['unit'] + "<extra></extra>"
            ))

        chart_a_title = var_cfg["chart_title"]
        fig_daily_high.update_layout(
            title=dict(text=f"{chart_a_title} ({var_cfg['unit']})", font=dict(size=18)),
            xaxis=dict(
                title="Calendar Day",
                type='category',
                categoryorder='array',
                categoryarray=dates,
                tickvals=dates,
                ticktext=[date_labels[d] for d in dates],
                **axis_style
            ),
            yaxis=dict(title=f"{var_cfg['label']} ({var_cfg['unit']})", zeroline=False, **axis_style),
            boxmode='group',
            boxgap=0.3,
            boxgroupgap=0.08,
            height=520,
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_daily_high, use_container_width=True)

        # 2. LOW CHART (only meaningful for variables with a two-sided daily range, e.g. temperature)
        if not daily_det_lows.empty:
            st.divider()
            fig_daily_low = go.Figure()
        
            for ens_name in ENS_ORDER:
                if ens_name in daily_ens_lows:
                    df_m_low = daily_ens_lows[ens_name]
                    color = MODEL_CONFIG[ens_name]["color"]
                
                    x_vals_low = []
                    y_vals_low = []
                    for date_str in dates:
                        if date_str in df_m_low.index:
                            vals = df_m_low.loc[date_str].values
                            x_vals_low.extend([date_str] * len(vals))
                            y_vals_low.extend(vals)
                        
                    fig_daily_low.add_trace(go.Box(
                        x=x_vals_low,
                        y=y_vals_low,
                        name=ens_name,
                        marker_color=color,
                        line=dict(width=2),
                        whiskerwidth=0.8,
                        boxpoints='outliers',
                        legendgroup=ens_name,
                        showlegend=False,
                        hoverinfo="y+name"
                    ))
                
            for det_col in daily_det_lows.columns:
                color = det_colors.get(det_col, "#D55E00")
                fig_daily_low.add_trace(go.Scatter(
                    x=daily_det_lows.index,
                    y=daily_det_lows[det_col],
                    mode='markers',
                    name=det_col,
                    showlegend=False,
                    marker=dict(color=color, size=11, symbol='diamond', line=dict(width=1.5, color='black')),
                    hovertemplate=f"<b>{det_col}</b><br>%{{y:.1f}} " + var_cfg['unit'] + "<extra></extra>"
                ))

            low_title = chart_a_title.replace("High", "Low")
            fig_daily_low.update_layout(
                title=dict(text=f"{low_title} ({var_cfg['unit']})", font=dict(size=18)),
                xaxis=dict(
                    title="Calendar Day",
                    type='category',
                    categoryorder='array',
                    categoryarray=dates,
                    tickvals=dates,
                    ticktext=[date_labels[d] for d in dates],
                    **axis_style
                ),
                yaxis=dict(title=f"Low {var_cfg['label']} ({var_cfg['unit']})", zeroline=False, **axis_style),
                boxmode='group',
                boxgap=0.3,
                boxgroupgap=0.08,
                height=520,
                hovermode="closest"
            )
            st.plotly_chart(fig_daily_low, use_container_width=True)

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
                if selected_var_key == "temperature_2m" and det_col in daily_det_lows.columns and d in daily_det_lows.index:
                    low_val = daily_det_lows.loc[d, det_col]
                    high_val = daily_det_highs.loc[d, det_col]
                    row[f"{det_col} (L/H)"] = f"{low_val:.1f}° / {high_val:.1f}°F"
                else:
                    row[det_col] = round(daily_det_highs.loc[d, det_col], 2)
                
            for ens_name in ["EPS", "AIFS", "GEFS", "WeatherNext"]:
                if ens_name in daily_ens_highs and d in daily_ens_highs[ens_name].index:
                    high_vals = daily_ens_highs[ens_name].loc[d].values
                    if selected_var_key == "temperature_2m" and ens_name in daily_ens_lows and d in daily_ens_lows[ens_name].index:
                        low_vals = daily_ens_lows[ens_name].loc[d].values
                        row[f"{ens_name} Med (L/H)"] = f"{np.median(low_vals):.1f}° / {np.median(high_vals):.1f}°F"
                    else:
                        row[f"{ens_name} Median"] = round(float(np.median(high_vals)), 2)
                
            if "Grand Ensemble" in daily_ens_highs and d in daily_ens_highs["Grand Ensemble"].index:
                g_highs = daily_ens_highs["Grand Ensemble"].loc[d].values
                if selected_var_key == "temperature_2m" and "Grand Ensemble" in daily_ens_lows and d in daily_ens_lows["Grand Ensemble"].index:
                    g_lows = daily_ens_lows["Grand Ensemble"].loc[d].values
                    row["Grand Ens Med (L/H)"] = f"{np.median(g_lows):.1f}° / {np.median(g_highs):.1f}°F"
                    row["High IQR Spread"] = f"{np.percentile(g_highs, 25):.1f}° to {np.percentile(g_highs, 75):.1f}°F"
                else:
                    row["Grand Ens Median"] = round(float(np.median(g_highs)), 2)
                    row["Consensus IQR"] = f"{np.percentile(g_highs, 25):.2f} to {np.percentile(g_highs, 75):.2f} {var_cfg['unit']}"
            
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
