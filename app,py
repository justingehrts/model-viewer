import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Multi-Model Ensemble Weather Dashboard",
    page_icon="🌤️",
    layout="wide"
)

# ==============================================================================
# CONFIGURATION & METADATA CONFIG (TWEAK B)
# ==============================================================================
WEATHER_VARS = {
    "temperature_2m": {
        "label": "Air Temperature",
        "unit": "°F",
        "hourly_param": "temperature_2m",
        "daily_agg": "max",          # 'max' for daily highs
        "chart_title": "Daily High Temperature Spread"
    },
    "precipitation": {
        "label": "Total Precipitation",
        "unit": "in",
        "hourly_param": "precipitation",
        "daily_agg": "sum",          # 'sum' for total daily rainfall
        "chart_title": "Daily Total Precipitation Spread"
    }
}

MODEL_CONFIG = {
    # Deterministic Operational Runs
    "ECMWF":          {"color": "#1f77b4"},  # Royal Blue
    "GFS":            {"color": "#d62728"},  # Crimson Red
    
    # Ensemble Model Families
    "EPS":            {"color": "#2b5c8f"},  # Deep Blue
    "AIFS":           {"color": "#7b4173"},  # Purple
    "GEFS":           {"color": "#e377c2"},  # Rose / Pink
    "WeatherNext":    {"color": "#2ca02c"},  # Emerald Green
    "Grand Ensemble": {"color": "#111111"}   # Charcoal / Off-Black
}

ENS_ORDER = ["EPS", "AIFS", "GEFS", "WeatherNext", "Grand Ensemble"]

DET_NAME_MAP = {
    "ecmwf_ifs025": "ECMWF",
    "gfs_seamless": "GFS"
}

ENS_NAME_MAP = {
    "ecmwf_ifs025": "EPS",
    "ecmwf_aifs025": "AIFS",
    "gfs_seamless": "GEFS",
    "google_weathernext2_ensemble": "WeatherNext"
}

# ==============================================================================
# LAYER 1: DATA INGESTION (WITH STREAMLIT CACHING)
# ==============================================================================

@st.cache_data(ttl=3600)  # Cache API responses in memory for 1 hour
def fetch_deterministic_data(lat, lon, days=7):
    url = "https://api.open-meteo.com/v1/forecast"
    models = ["ecmwf_ifs025", "gfs_seamless"]
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation",
        "models": ",".join(models),
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "auto",
        "forecast_days": days
    }
    
    res = requests.get(url, params=params)
    res.raise_for_status()
    data = res.json()
    hourly = data["hourly"]
    
    df_temp = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
    df_precip = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
    
    for m in models:
        nickname = DET_NAME_MAP.get(m, m)
        temp_key = f"temperature_2m_{m}"
        precip_key = f"precipitation_{m}"
        
        if temp_key in hourly:
            df_temp[nickname] = hourly[temp_key]
        if precip_key in hourly:
            df_precip[nickname] = hourly[precip_key]
            
    return df_temp, df_precip


@st.cache_data(ttl=3600)
def fetch_ensemble_data(lat, lon, days=7):
    url = "https://ensemble-api.open-meteo.com/v1/ensemble"
    models = ["ecmwf_ifs025", "ecmwf_aifs025", "gfs_seamless", "google_weathernext2_ensemble"]
    
    dict_temp = {}
    dict_precip = {}
    
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
        
        res = requests.get(url, params=params)
        if res.status_code != 200:
            continue
            
        data = res.json()
        if "hourly" not in data:
            continue
            
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
        
    return dict_temp, dict_precip

# ==============================================================================
# LAYER 2 & 3: PROCESSING & GRAND ENSEMBLE BUILDER
# ==============================================================================

def process_ensemble_data(dict_ens, df_det, agg_func="max"):
    # 1. Build Grand Ensemble
    all_member_dfs = []
    for model_name, df_m in dict_ens.items():
        cols_to_rename = {c: f"{model_name}_{c}" for c in df_m.columns if c != 'time'}
        df_renamed = df_m.rename(columns=cols_to_rename).set_index('time')
        all_member_dfs.append(df_renamed)
        
    if all_member_dfs:
        df_grand = pd.concat(all_member_dfs, axis=1).reset_index()
        dict_ens["Grand Ensemble"] = df_grand

    # 2. Compute Hourly Summaries
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

    # 3. Compute Daily Calendar Aggregations
    daily_ens = {}
    for name in ENS_ORDER:
        if name in dict_ens:
            df = dict_ens[name]
            member_cols = [c for c in df.columns if c != 'time']
            df_daily = df.copy()
            df_daily['date'] = df_daily['time'].dt.strftime('%Y-%m-%d')
            daily_ens[name] = df_daily.groupby('date')[member_cols].agg(agg_func)

    # 4. Compute Daily Deterministic Aggregations
    df_det_daily = df_det.copy()
    df_det_daily['date'] = df_det_daily['time'].dt.strftime('%Y-%m-%d')
    det_cols = [c for c in df_det.columns if c != 'time']
    daily_det = df_det_daily.groupby('date')[det_cols].agg(agg_func)

    return hourly_summaries, daily_ens, daily_det

# ==============================================================================
# STREAMLIT UI & SIDEBAR
# ==============================================================================

st.title("🌤️ Multi-Model Ensemble Weather Consensus Dashboard")
st.markdown("Comparing deterministic operational runs against **197 probabilistic ensemble members** across European (ECMWF/EPS/AIFS), American (GFS/GEFS), and AI (Google WeatherNext 2) forecasting systems.")

with st.sidebar:
    st.header("⚙️ Forecast Controls")
    lat = st.number_input("Latitude", value=39.97, step=0.01, format="%.2f")
    lon = st.number_input("Longitude", value=-83.00, step=0.01, format="%.2f")
    forecast_days = st.slider("Forecast Horizon (Days)", min_value=3, max_value=14, value=7)
    
    st.divider()
    
    selected_var_key = st.selectbox(
        "Forecast Parameter",
        options=list(WEATHER_VARS.keys()),
        format_func=lambda x: WEATHER_VARS[x]["label"]
    )
    
    var_cfg = WEATHER_VARS[selected_var_key]
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Fetch Pre-Cached Datasets
with st.spinner("Fetching multi-model ensemble payloads..."):
    df_det_temp, df_det_precip = fetch_deterministic_data(lat, lon, days=forecast_days)
    dict_ens_temp, dict_ens_precip = fetch_ensemble_data(lat, lon, days=forecast_days)

# Select Payload based on Dropdown
if selected_var_key == "temperature_2m":
    df_det_active = df_det_temp
    dict_ens_active = dict_ens_temp
else:
    df_det_active = df_det_precip
    dict_ens_active = dict_ens_precip

# Process Data dynamically
hourly_summaries, daily_ens, daily_det = process_ensemble_data(
    dict_ens_active, 
    df_det_active, 
    agg_func=var_cfg["daily_agg"]
)

# ==============================================================================
# KEY METRICS SUMMARY CARDS
# ==============================================================================

if "Grand Ensemble" in daily_ens and not daily_det.empty:
    grand_daily = daily_ens["Grand Ensemble"]
    dates = list(daily_det.index)
    
    # Calculate key metrics for the 7-day window
    peak_val = grand_daily.loc[dates[0]].median().max() if len(dates) > 0 else 0
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
    
    # Deterministic Operational Runs
    for det_name in ["ECMWF", "GFS"]:
        if det_name in df_det_active.columns:
            color = MODEL_CONFIG[det_name]["color"]
            fig_hourly.add_trace(go.Scatter(
                x=df_det_active['time'],
                y=df_det_active[det_name],
                mode='lines',
                name=f"{det_name} (Det)",
                line=dict(color=color, width=3),
                hovertemplate=f"%{{x|%a %b %d, %I:%M %p}}<br><b>{det_name} (Det)</b>: %{{y:.2f}} {var_cfg['unit']}<extra></extra>"
            ))
            
    # Ensemble Medians
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
        template="plotly_white",
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_hourly, use_container_width=True)


# --- TAB 2: DAILY DISTRIBUTION SPREAD (BOX PLOTS) ---
with tab2:
    fig_daily = go.Figure()
    dates = list(daily_det.index)
    
    # Ensemble Box Plots
    for ens_name in ENS_ORDER:
        if ens_name in daily_ens:
            df_daily_m = daily_ens[ens_name]
            color = MODEL_CONFIG[ens_name]["color"]
            
            for date_str in dates:
                if date_str in df_daily_m.index:
                    vals = df_daily_m.loc[date_str].values
                    fig_daily.add_trace(go.Box(
                        y=vals,
                        x=[date_str] * len(vals),
                        name=ens_name,
                        marker_color=color,
                        boxpoints='outliers',
                        legendgroup=ens_name,
                        showlegend=(date_str == dates[0]),
                        hovertemplate=f"<b>{ens_name}</b><br>Date: {date_str}<br>Val: %{{y:.2f}} {var_cfg['unit']}<extra></extra>"
                    ))

    # Overlay Deterministic Markers
    for det_name in ["ECMWF", "GFS"]:
        if det_name in daily_det.columns:
            color = MODEL_CONFIG[det_name]["color"]
            fig_daily.add_trace(go.Scatter(
                x=daily_det.index,
                y=daily_det[det_name],
                mode='markers',
                name=f"{det_name} (Det)",
                marker=dict(color=color, size=11, symbol='diamond', line=dict(width=1.5, color='black')),
                hovertemplate=f"<b>{det_name} (Det)</b><br>Date: %{{x}}<br>Val: %{{y:.2f}} {var_cfg['unit']}<extra></extra>"
            ))

    fig_daily.update_layout(
        title=dict(text=f"{var_cfg['chart_title']} ({var_cfg['unit']})", font=dict(size=18)),
        xaxis_title="Calendar Day",
        yaxis_title=f"{var_cfg['label']} ({var_cfg['unit']})",
        boxmode='group',
        template="plotly_white",
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_daily, use_container_width=True)


# --- TAB 3: SUMMARY DATA TABLE & CSV DOWNLOAD ---
with tab3:
    summary_rows = []
    dates = list(daily_det.index)
    
    for d in dates:
        date_obj = pd.to_datetime(d)
        row = {"Date": date_obj.strftime("%a %b %d, %Y")}
        
        # Deterministic
        for det_col in ["ECMWF", "GFS"]:
            if det_col in daily_det.columns:
                row[f"Det {det_col}"] = round(daily_det.loc[d, det_col], 2)
                
        # Individual Ensembles
        for ens_name in ["EPS", "AIFS", "GEFS", "WeatherNext"]:
            if ens_name in daily_ens and d in daily_ens[ens_name].index:
                vals = daily_ens[ens_name].loc[d].values
                row[f"{ens_name} Median"] = round(float(np.median(vals)), 2)
                
        # Grand Ensemble
        if "Grand Ensemble" in daily_ens and d in daily_ens["Grand Ensemble"].index:
            g_vals = daily_ens["Grand Ensemble"].loc[d].values
            row["Grand Ens Median"] = round(float(np.median(g_vals)), 2)
            row["Consensus IQR (25-75%)"] = f"{np.percentile(g_vals, 25):.2f} to {np.percentile(g_vals, 75):.2f} {var_cfg['unit']}"
            
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
