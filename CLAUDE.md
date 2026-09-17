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

1. **Config** — `WEATHER_VARS` (per-variable display/unit/aggregation metadata), `MODEL_CONFIG` (per-model color + display config), `ENS_ORDER`, `ENS_NAME_MAP`
2. **Helpers** — live run-cycle estimation, offline mock data generator, airport→coordinates geocoding
3. **Data ingestion** (`fetch_deterministic_data`, `fetch_ensemble_data`) — `@st.cache_data(ttl=900)`-cached calls to Open-Meteo
4. **Processing** (`process_ensemble_data`) — builds the Grand Ensemble, hourly median/mean/IQR summaries, and daily high/low aggregates
5. **UI** — sidebar controls, metric cards, and three tabs (hourly time series, daily distribution box plots, summary table + CSV download) built with Plotly

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
- **Audience**: this is for general weather watchers, not pilots/dispatchers.
  Avoid introducing aviation jargon (METAR, TAF, ceiling/visibility, etc.)
  into labels or copy — plain-language forecast terms only.
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
