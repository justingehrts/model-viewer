import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time

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
        "chart_title": "Daily Temperature Spread"
    },
    "precipitation": {
        "label": "Total Precipitation",
        "unit": "in",
        "hourly_param": "precipitation",
        "daily_agg": "sum",          # 'sum' for total daily rainfall
        "chart_title": "Daily Total Precipitation Spread"
    }
}

# Color Vision Deficiency (CVD) Safe Palette (Okabe-Ito Inspired)
MODEL_CONFIG = {
    # Deterministic Operational Run
    "Deterministic":  {"color": "#D55E00"},  # Vermilion
    
    # Ensemble Model Families
    "EPS":            {"color": "#0072B2"},  # Blue
    "AIFS":           {"color": "#CC79A7"},  # Purple
    "GEFS":           {"color": "#E69F00"},  # Amber
    "WeatherNext":    {"color": "#009E73"},  # Teal
    "Grand Ensemble": {"color": "#888888"}   # Mid-Gray
}

ENS_ORDER = ["EPS", "AIFS", "GEFS", "WeatherNext", "Grand Ensemble"]

ENS_NAME_MAP = {
    "ecmwf_ifs025": "EPS",
    "ecmwf_aifs025": "AIFS",
    "gfs_seamless": "GEFS",
    "google_weathernext2_ensemble": "WeatherNext"
}

# ==============================================================================
# HELPER 1: DEV MOCK DATA GENERATOR (OFFLINE / 0 API CALLS)
# ==============================================================================

def generate_mock_data(days=7):
    """Generates synthetic hourly weather data so you can test layout/charts offline."""
    now = datetime.now()
    dates = pd.date_range(start=now, periods=days * 24, freq='h')
    
    # Generate realistic diurnal temperature curve (60°F to 80°F)
    base_temp = 70 + 10 * np.sin(np.linspace(0, days * 2 * np.pi, len(dates)))
    
    df_det_temp = pd.DataFrame({'time': dates, 'Deterministic': base_temp})
    df_det_precip = pd.DataFrame({'time': dates, 'Deterministic': np.zeros(len(dates))})
    
    dict_ens_temp = {}
    dict_ens_precip = {}
    run_cycles = {}
    
    for nickname in ["EPS", "AIFS", "GEFS", "WeatherNext"]:
        df_t = pd.DataFrame({'time': dates})
        df_p = pd.DataFrame({'time': dates})
        
        # Add synthetic ensemble member variation
        for m in range(1, 31):
            df_t[f"member_{m}"] = base_temp + np.random.normal(0, 2.5, len(dates))
            df_p[f"member_{m}"] = np.maximum(0, np.random.normal(0, 0.05, len(dates)))
            
        dict_ens_temp[nickname] = df_t
        dict_ens_precip[nickname] = df_p
        run_cycles[nickname] = "DEV-MOCK 00Z"
        
    return df_det_temp, df_det_precip, dict_ens_temp, dict_ens_precip, run_cycles

# ==============================================================================
# HELPER 2: AIRPORT GEOCODING LOOKUP
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
    """Fetches explicit operational deterministic runs for ECMWF IFS and GFS."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation",
        "models": ["ecmwf_ifs025", "gfs_seamless"],  # Explicitly requests Euro (IFS) & GFS
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "auto",
        "forecast_days": days
    }
    
    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        
        hourly = data["hourly"]
        df_temp = pd.DataFrame({
            "time": pd.to_datetime(hourly["time"]),
            "ECMWF Operational": hourly.get("temperature_2m_ecmwf_ifs025"),
            "GFS Operational": hourly.get("temperature_2m_gfs_seamless")
        })
        df_precip = pd.DataFrame({
            "time": pd.to_datetime(hourly["time"]),
            "ECMWF Operational": hourly.get("precipitation_ecmwf_ifs025"),
            "GFS Operational": hourly.get("precipitation_gfs_seamless")
        })
        
        fetch_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return df_temp, df_precip, fetch_time, None
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), "", str(e)

@st.cache_data(ttl=900)
def fetch_ensemble_data(lat, lon, days=7):
    """Fetches 197 probabilistic ensemble members across EPS, AIFS, GEFS, WeatherNext."""
    url = "https://ensemble-api.open-meteo.com/v1/ensemble"
    models = ["ecmwf_ifs025", "ecmwf_aifs025", "gfs_seamless", "google_weathernext2_ensemble"]
    
    dict_temp = {}
    dict_precip = {}
    run_cycles = {}
    errors = []
    
    for m in models:
        nickname = ENS_NAME_MAP.get(m, m)
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,precipitation",
            "models": m,
            "temperature_unit": "fahrenheit",
            "precipitation_unit": "inch",
            "timezone": "auto",
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
                
            if "model_initialization_time" in data and data["model_initialization_time"]:
                init_dt = pd.to_datetime(data["model_initialization_time"])
                run_cycles[nickname] = init_dt.strftime("%m/%d %HZ")
            elif "hourly" in data and "time" in data["hourly"]:
                offset_sec = data.get("utc_offset_seconds", 0)
                first_time_utc = pd.to_datetime(data["hourly"]["time"][0]) - pd.Timedelta(seconds=offset_sec)
                run_cycles[nickname] = first_time_utc.strftime("%m/%d %HZ")
                
            hourly = data["hourly"]
            df_m_temp = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
            df_m_precip = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
            
            temp_keys = [k for k in hourly.keys() if k.startswith("temperature_2m")]
            precip_keys = [k for k in hourly.keys() if k.startswith("precipitation")]
            
            for k in temp_keys:
                col = k.replace("temperature_2m_", "")
                df_m_temp[col] = hourly[k]
                
            for k in precip_keys:
                col = k.replace("precipitation_", "")
                df_m_precip[col] = hourly[k]
                
            dict_temp[nickname] = df_m_temp
            dict_precip[nickname] = df_m_precip

            time.sleep(0.15)  # Micro-pause to prevent rate-limit bursts

        except Exception as e:
            errors.append(f"{nickname}: {str(e)}")
            continue
        
    return dict_temp, dict_precip, run_cycles, errors

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

    df_det_daily = df_det.copy()
    if not df_det_daily.empty and 'time' in df_det_daily.columns:
        df_det_daily['date'] = df_det_daily['time'].dt.strftime('%Y-%m-%d')
        det_cols = [c for c in df_det.columns if c != 'time']

        if selected_var_key == "temperature_2m":
            for name in ENS_ORDER:
                if name in dict_ens:
                    df = dict_ens[name]
                    member_cols = [c for c in df.columns if c != 'time']
                    df_daily = df.copy()
                    df_daily['date'] = df_daily['time'].dt.strftime('%Y-%m-%d')
                    daily_ens_highs[name] = df_daily.groupby('date')[member_cols].max()
                    daily_ens_lows[name] = df_daily.groupby('date')[member_cols].min()
                    
            daily_det_highs = df_det_daily.groupby('date')[det_cols].max()
            daily_det_lows = df_det_daily.groupby('date')[det_cols].min()
        else:
            for name in ENS_ORDER:
                if name in dict_ens:
                    df = dict_ens[name]
                    member_cols = [c for c in df.columns if c != 'time']
                    df_daily = df.copy()
                    df_daily['date'] = df_daily['time'].dt.strftime('%Y-%m-%d')
                    daily_ens_highs[name] = df_daily.groupby('date')[member_cols].sum()
                    
            daily_det_highs = df_det_daily.groupby('date')[det_cols].sum()

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
            auto_lat, auto_lon, station_name = get_coordinates_from_airport(airport_input)
            lat, lon = auto_lat, auto_lon
            st.caption(f"📍 **{station_name}** ({lat:.2f}°, {lon:.2f}°)")
        else:
            lat = st.number_input("Latitude", value=39.97, step=0.01, format="%.2f")
            lon = st.number_input("Longitude", value=-83.00, step=0.01, format="%.2f")
            
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

# ==============================================================================
# ROUTING: DEV MODE VS LIVE FETCH
# ==============================================================================

if dev_mode:
    # 0 API Calls — Instant offline render with synthetic data
    df_det_temp, df_det_precip, dict_ens_temp, dict_ens_precip, run_cycles = generate_mock_data(days=forecast_days)
    fetch_time = "OFFLINE DEV MODE"
    det_err = None
else:
    # Real live network calls to Open-Meteo
    with st.spinner("Fetching multi-model ensemble payloads..."):
        df_det_temp, df_det_precip, fetch_time, det_err = fetch_deterministic_data(lat, lon, days=forecast_days)
        dict_ens_temp, dict_ens_precip, run_cycles, ens_errs = fetch_ensemble_data(lat, lon, days=forecast_days)

# Safety Check (ONLY runs if Dev Mode is OFF)
if not dev_mode and (det_err or df_det_temp.empty):
    st.error(f"⚠️ Unable to fetch weather data from Open-Meteo. Details: `{det_err}`")
    st.info("💡 **Tip:** Switch on '🛠️ Dev Mode' in the sidebar to test layout & charts offline without hitting API rate limits.")
    st.stop()

# Display Run Information in Sidebar
with st.sidebar:
    st.caption(f"🕒 **Last API Fetch:** {fetch_time}")
    st.markdown("**Model Run Cycles Loaded:**")
    for model_name, cycle_str in run_cycles.items():
        st.text(f"• {model_name:<12}: {cycle_str}")

# Select Payload based on Dropdown
if selected_var_key == "temperature_2m":
    df_det_active = df_det_temp
    dict_ens_active = dict_ens_temp
else:
    df_det_active = df_det_precip
    dict_ens_active = dict_ens_precip

# Process Data dynamically
hourly_summaries, daily_ens_highs, daily_ens_lows, daily_det_highs, daily_det_lows = process_ensemble_data(
    dict_ens_active, 
    df_det_active, 
    selected_var_key=selected_var_key
)

# ==============================================================================
# KEY METRICS SUMMARY CARDS
# ==============================================================================

if "Grand Ensemble" in daily_ens_highs and not daily_det_highs.empty:
    grand_daily = daily_ens_highs["Grand Ensemble"]
    dates = list(daily_det_highs.index)
    
    if len(dates) > 0:
        peak_val = grand_daily.loc[dates[0]].median().max()
        max_day_str = dates[0]
        
        for d in dates:
            m_val = np.median(grand_daily.loc[d].values)
            if m_val > peak_val:
                peak_val = m_val
                max_day_str = d
                
        first_day_spread = np.percentile(grand_daily.loc[dates[0]].values, 75) - np.percentile(grand_daily.loc[dates[0]].values, 25)
        last_day_spread = np.percentile(grand_daily.loc[dates[-1]].values, 75) - np.percentile(grand_daily.loc[dates[-1]].values, 25)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Location Coordinates", f"{lat:.2f}°, {lon:.2f}°")
        col2.metric(f"Peak Consensus {var_cfg['label']}", f"{peak_val:.1f} {var_cfg['unit']}", f"Day: {max_day_str}")
        col3.metric("Day 1 Consensus Spread (IQR)", f"{first_day_spread:.1f} {var_cfg['unit']}")
        col4.metric(f"Day {forecast_days} Uncertainty Spread", f"{last_day_spread:.1f} {var_cfg['unit']}", f"+{last_day_spread - first_day_spread:.1f} {var_cfg['unit']} vs Day 1", delta_color="inverse")

st.divider()

# ==============================================================================
# LAYER 4: PLOTLY VISUALIZATIONS
# ==============================================================================

tab1, tab2, tab3 = st.tabs(["📈 Hourly Time-Series", "📊 Daily Distribution Spread", "📋 Consensus Summary Table"])

# --- TAB 1: HOURLY TIME-SERIES ---
with tab1:
    fig_hourly = go.Figure()
    
    if "Deterministic" in df_det_active.columns:
        color = MODEL_CONFIG["Deterministic"]["color"]
        fig_hourly.add_trace(go.Scatter(
            x=df_det_active['time'],
            y=df_det_active["Deterministic"],
            mode='lines',
            name="Deterministic Operational Run",
            line=dict(color=color, width=3),
            hovertemplate=f"%{{x|%a %b %d, %I:%M %p}}<br><b>Deterministic Run</b>: %{{y:.2f}} {var_cfg['unit']}<extra></extra>"
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
        xaxis_title="Date / Time (Local)",
        yaxis_title=f"{var_cfg['label']} ({var_cfg['unit']})",
        hovermode="x unified",
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_hourly, use_container_width=True)


# --- TAB 2: DAILY DISTRIBUTION SPREAD ---
with tab2:
    dates = list(daily_det_highs.index)
    
    # Common Axis Styling Options
    axis_style = dict(
        showgrid=True,
        gridcolor="rgba(128, 128, 128, 0.15)",
        gridwidth=1,
        showline=True,
        linecolor="rgba(128, 128, 128, 0.4)",
        linewidth=1.5,
        tickfont=dict(size=12, family="sans-serif"),
        title_font=dict(size=13, family="sans-serif", color="#555555")
    )
    
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

    if "Deterministic" in daily_det_highs.columns:
        color = MODEL_CONFIG["Deterministic"]["color"]
        fig_daily_high.add_trace(go.Scatter(
            x=daily_det_highs.index,
            y=daily_det_highs["Deterministic"],
            mode='markers',
            name="Deterministic Operational Run",
            marker=dict(color=color, size=11, symbol='diamond', line=dict(width=1.5, color='black')),
            hovertemplate="<b>%{fullData.name}</b><br>%{y:.1f} " + var_cfg['unit'] + "<extra></extra>"
        ))

    chart_a_title = "Daily High Temperature Spread" if selected_var_key == "temperature_2m" else "Daily Total Precipitation Spread"
    fig_daily_high.update_layout(
        title=dict(text=f"{chart_a_title} ({var_cfg['unit']})", font=dict(size=18)),
        xaxis=dict(title="Calendar Day", **axis_style),
        yaxis=dict(title=f"{var_cfg['label']} ({var_cfg['unit']})", zeroline=False, **axis_style),
        boxmode='group',
        boxgap=0.3,
        boxgroupgap=0.08,
        height=520,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_daily_high, use_container_width=True)

    # 2. LOW TEMPERATURE CHART
    if selected_var_key == "temperature_2m" and not daily_det_lows.empty:
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
                
        if "Deterministic" in daily_det_lows.columns:
            color = MODEL_CONFIG["Deterministic"]["color"]
            fig_daily_low.add_trace(go.Scatter(
                x=daily_det_lows.index,
                y=daily_det_lows["Deterministic"],
                mode='markers',
                name="Deterministic Operational Run",
                showlegend=False,
                marker=dict(color=color, size=11, symbol='diamond', line=dict(width=1.5, color='black')),
                hovertemplate="<b>%{fullData.name}</b><br>%{y:.1f} " + var_cfg['unit'] + "<extra></extra>"
            ))

        fig_daily_low.update_layout(
            title=dict(text=f"Daily Low Temperature Spread ({var_cfg['unit']})", font=dict(size=18)),
            xaxis=dict(title="Calendar Day", **axis_style),
            yaxis=dict(title=f"Low Temperature ({var_cfg['unit']})", zeroline=False, **axis_style),
            boxmode='group',
            boxgap=0.3,
            boxgroupgap=0.08,
            height=520,
            hovermode="closest"
        )
        st.plotly_chart(fig_daily_low, use_container_width=True)

# --- TAB 3: SUMMARY DATA TABLE & CSV DOWNLOAD ---
with tab3:
    summary_rows = []
    dates = list(daily_det_highs.index)
    
    for d in dates:
        date_obj = pd.to_datetime(d)
        row = {"Date": date_obj.strftime("%a %b %d, %Y")}
        
        if "Deterministic" in daily_det_highs.columns:
            if selected_var_key == "temperature_2m" and d in daily_det_lows.index:
                row["Deterministic (L/H)"] = f"{daily_det_lows.loc[d, 'Deterministic']:.1f}° / {daily_det_highs.loc[d, 'Deterministic']:.1f}°F"
            else:
                row["Deterministic"] = round(daily_det_highs.loc[d, 'Deterministic'], 2)
                
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
