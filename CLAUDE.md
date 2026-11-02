# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

A single-page Streamlit dashboard that compares weather forecast models for broadcast meteorologists (not a detailed aviation/METAR/TAF tool — airport
codes are only used as a convenient way to look up a location's lat/lon).

It contrasts:
- **Deterministic** operational runs: ECMWF (IFS), GFS, NBM
- **Ensemble** model families: EPS (ECMWF), AIFS (ECMWF AI), GEFS (NOAA), WeatherNext (Google), plus a synthesized "Grand Ensemble" that pools all members

Data comes from the [Open-Meteo API](https://open-meteo.com/) (forecast and
ensemble endpoints) — free and keyless. Airport-code location lookup falls
back through aviationweather.gov's station-info API, then Open-Meteo's
geocoding API.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

There's no test suite or lint config in this repo. Use the sidebar's "Dev
Mode" toggle to exercise the UI with synthetic data instead of hitting the
live APIs (useful when iterating on charts/layout without burning API calls
or waiting on network latency).

## Structure

Everything lives in `app.py`, organized top-to-bottom as:

1. **Config** — `WEATHER_VARS` (per-variable display/unit/aggregation metadata; an optional `grid` key sets the Hourly Grid color scale and enables it for that variable; `derived` entries aren't requested from the API directly), `SNOW_INPUT_PARAMS` (snowfall, surface pressure, and temperature/RH/vertical velocity/geopotential height on the eight pressure levels the ensemble API returns, 1000–300 mb, fetched in a separate request), `COBB_SR_TEMPS_C`/`COBB_SR_VALUES` (Cobb's temperature→snow ratio curve, read off the 2005 paper's Fig. 5), `MODEL_CONFIG` (per-model color + display config), `ENS_ORDER`, `ENS_NAME_MAP`
2. **Helpers** — live run-cycle estimation, offline mock data generator, airport→coordinates geocoding, model verification (`get_current_conditions` fetches the latest METAR for a station, `calculate_apparent_temperature` computes a comparable feels-like from it)
3. **Data ingestion** (`fetch_deterministic_data`, `fetch_ensemble_data`) — `@st.cache_data(ttl=3600)`-cached calls to Open-Meteo
4. **Processing** — `compute_snowfall` derives the 10:1, Kuchera (`kuchera_slr`: warmest temperature from the surface to 500 mb) and Cobb (`cobb_slr`: Cobb & Waldstreicher 2005 weighting of a temperature-based ratio by upward motion in layers with RH w.r.t. ice ≥ 90%, on 25 mb slices interpolated in log-pressure; falls back to Kuchera in hours with no saturated rising layer) snowfall variables from the model's snow liquid equivalent (Open-Meteo `snowfall` ÷ 7). Below-ground levels are skipped; models lacking the needed upper-air fields are left out of that view. Open-Meteo's `vertical_velocity` is positive-up m/s; the Cobb weights are normalized, so its units cancel. `process_ensemble_data` — builds the Grand Ensemble, hourly median/mean/IQR summaries, and daily high/low aggregates; `get_value_at_time`/`get_grand_ensemble_median_at_time` do single-hour nearest-time lookups for the verification panel
5. **UI** — sidebar controls, a model-verification expander (currently-selected variable vs. the station's latest METAR), and four tabs (hourly time series, daily distribution box plots, summary table + CSV download, and a 48-hour hourly grid heatmap with a model-spread row for the selected variable, with precipitation and snowfall shown as running totals — `build_hourly_grid`) built with Plotly

## Conventions

- **Units**: temperature is always **°F**, wind is always **mph**. Pass
  `temperature_unit=fahrenheit` (and the equivalent wind unit param) when
  calling Open-Meteo rather than converting client-side after the fact.
- **Colors**: model colors follow the Okabe-Ito CVD-safe palette and are
  centralized in `MODEL_CONFIG`. When adding a new model or chart trace,
  read its color from `MODEL_CONFIG` rather than hardcoding a new hex value.
  Note: the Plotly tab code currently has a couple of local `det_colors`
  dicts that duplicate the deterministic-model hex values instead of reading
  `MODEL_CONFIG` — when touching that code, prefer consolidating onto
  `MODEL_CONFIG` rather than adding a third copy.
- **Audience**: this is for broadcast meteorologists. Aviation terminology
  (METAR, station observations, etc.) is fine in labels/copy where relevant
  — this isn't a plain-language-only tool.
- **Secrets**: never hardcode API keys, tokens, or credentials in `app.py`
  or anywhere else in the repo. The current data sources (Open-Meteo,
  aviationweather.gov) don't require auth, but if a future data source does,
  read it via `st.secrets` (e.g. `st.secrets["some_api_key"]`), not a literal
  string or a checked-in `.env`.
- **Model naming**: user-facing model names ("ECMWF Operational", "EPS",
  "GEFS", etc.) are the canonical keys used across `MODEL_CONFIG`,
  `ENS_ORDER`, and DataFrame columns. `ENS_NAME_MAP` is the only place that
  translates Open-Meteo's raw model slugs (e.g. `ecmwf_ifs025`) into those
  display names — extend it there when adding a new ensemble model.
