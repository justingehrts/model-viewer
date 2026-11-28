# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

A single-page Streamlit dashboard that compares weather forecast models for broadcast meteorologists (not a detailed aviation/METAR/TAF tool — airport
codes are only used as a convenient way to look up a location's lat/lon).

It contrasts:
- **Deterministic** operational runs: ECMWF (IFS), GFS, NBM, RRFS (NOAA's 3 km CONUS model, HRRR/NAM successor; 84 h, so it covers only the first ~3.5 days)
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
3. **Data ingestion** (`fetch_deterministic_data`, `fetch_ensemble_data`, `fetch_previous_runs`) — `@st.cache_data(ttl=3600)`-cached calls to Open-Meteo; `fetch_previous_runs` uses the Previous Runs API (`temperature_2m_previous_day1..7` for the deterministic models, columns `n0`..`n7` = forecast made N days before each hour). `fetch_archive_daily` reads per-member ensemble daily highs/lows from the forecast archive (public `forecast-archive` branch via raw.githubusercontent.com) for stations in `ARCHIVE_STATIONS`
4. **Processing** — `compute_snowfall` derives the 10:1, Kuchera (`kuchera_slr`: warmest temperature from the surface to 500 mb) and Cobb (`cobb_slr`: Cobb & Waldstreicher 2005 weighting of a temperature-based ratio by upward motion in layers with RH w.r.t. ice ≥ 90%, on 25 mb slices interpolated in log-pressure; falls back to Kuchera in hours with no saturated rising layer) snowfall variables from the model's snow liquid equivalent (Open-Meteo `snowfall` ÷ 7). Below-ground levels are skipped; models lacking the needed upper-air fields are left out of that view. Open-Meteo's `vertical_velocity` is positive-up m/s; the Cobb weights are normalized, so its units cancel. `process_ensemble_data` — builds the Grand Ensemble (pooled members, but summarized with equal share per model family via `equal_share_weights`/`grand_stats`, so a 64-member family doesn't outvote a 31-member one; Tab 2 draws its box from precomputed weighted stats), hourly median/mean/IQR summaries, and daily high/low aggregates (deterministic daily values need at least `MIN_HOURS_FOR_DAILY` = 20 hours in the day, so RRFS's or anyone's partial last day doesn't produce a bogus high/low); `get_value_at_time`/`get_grand_ensemble_median_at_time` do single-hour nearest-time lookups for the verification panel
5. **UI** — a sidebar **View** switch (Forecast | Hazards, outside the form so it's instant). Forecast view: sidebar controls, a model-verification expander (currently-selected variable vs. the station's latest METAR), and five tabs (hourly time series, daily distribution box plots, summary table + CSV download, a 48-hour hourly grid heatmap with a model-spread row for the selected variable, with precipitation and snowfall shown as running totals — `build_hourly_grid` — and a temperature Trends tab: a trend grid of target days × the last 7 runs with a Δ-vs-previous-run column, plus a dprog/dt chart for one target day; daily high, low or a specific local hour; `run_forecast_value` reads one run's value from Previous Runs data, `grand_trend_value` the Grand Ensemble from the archive) built with Plotly. A sidebar "Show NCEI normals (1991–2020)" toggle (Forecast view, Air Temperature only, outside the form so it's instant; shown only for airports in `NORMALS_STATIONS`, currently KCMH) draws a faint normal line behind the charts: on the hourly chart the real NCEI hourly normal with its 10th–90th percentile band (`fetch_hourly_normals`/`normal_hourly`, no interpolation; NCEI has no hourly standard deviation), on the daily high/low charts the daily normal with ±1σ/±2σ bands (`fetch_daily_normals`, normal high/low and their standard deviations). It also adds departure-from-normal chips under the daily charts (Grand Ensemble median, in °F and σ) and a "NCEI normal" row plus a Temperature | Departure (°F) color switch on the Hourly Grid. There are no NCEI normals for Feels Like, so it isn't offered there. `station_now_local` gives the station-local "now" used by the grid, Trends and Hazards tabs. Hazards view hides the parameter dropdown, tick-interval control and METAR panel and shows four tabs. Three are hazard heatmaps (🌫️ Fog, 🌦️ Drizzle, 🧊 Freezing Drizzle), each a heatmap from 12 h before the latest METAR hour (also the first forecast column, marked by a dashed line) to 72 h after it, with an Observed (METAR) row on top (`fetch_metar_history`, `build_observed_row`, `render_hazard_heatmap`; fog colored by observed visibility on the forecast scale, drizzle/freezing drizzle by DZ/FZDZ in the present weather) and a one-line observed-vs-Overall summary (`observed_summary`). Past model columns are each model's most recent run (hazard requests use `past_days=1`), not the forecast made earlier; the Signals row has no past data. The fourth, 🗺️ Fog Domain, maps fog level over central Ohio from the GRIB extracts described under "Fog domain extract" (`fetch_fog_domain`, `domain_levels`, `fetch_domain_metars`, `FOG_SITES`): a model radio (NBM/GFS/…), fog level or chance of visibility ≤ 1 mi, a valid-time slider with a jump to the peak-fog hour, the latest METARs as dots on the same scale, a share-of-area-with-fog chart per model and a site-by-time heatmap. Dev Mode uses mock domain data unless the `FOG_DOMAIN_URL` environment variable is set (point it at a folder served over http to test real files). The fog hazard heatmap's scoring is described next. The Fog tab flags fog potential over the next 72 h as a heatmap (rows NBM vis, GEFS %, RRFS vis, Signals, Overall; levels None/Low/Moderate/High/Dense likely, thresholds in `FOG_TIERS`: mist < 3 mi, fog ≤ 1 mi, dense ≤ 1/4 mi; the fog and dense limits are inclusive) with hover evidence, a likely type (radiation/advection) and a list of fog periods. `fetch_fog_inputs` adds visibility/low cloud/is_day (GFS, NBM, ECMWF), RRFS 3 km visibility (`RRFS_SLUG`, optional, its own request) and GEFS member visibility; dew-point spread and wind come from the already-fetched `dict_det`/`dict_ens`. Scoring lives in `vis_level`, `pct_level`, `combine_fog_levels`, `fog_type`, `build_fog_rows` and `fog_episodes`. The Drizzle/Freezing Drizzle tabs score each model or member per hour as a tier 0–3 (`drizzle_tiers`, thresholds in `DRIZZLE_TIERS`): drizzle from a saturated surface (2 m temp − dew point ≤ 2 °F), a moist low layer (RH ≥ 95% at 925 mb, ≥ 90% at 850 mb; 975/950 mb near-surface RH where available) and light model QPF (≤ 0.02 in/h); freezing drizzle needs a 2 m temp ≤ 33 °F plus the same moisture/QPF test, graded by the temperature at the top of the saturated column (warmer than −5 °C High, −10 to −5 °C Moderate, colder or surface-only Low). There is deliberately no dry-air-aloft or cloud-depth test: it had no skill in the backtest. Deterministic rows (GFS, ECMWF, RRFS) show the tier; EPS/AIFS/GEFS and Overall average a per-tier chance (`DRIZZLE_PROB`, calibrated on KCMH METARs; `FZDZ_WEIGHT` for freezing drizzle) and band it (`PROB_BANDS`, `prob_band`, `build_drizzle_rows`). `fetch_drizzle_inputs` makes six small requests (one per source, only in the Hazards view) for T/RH on pressure levels, surface pressure, 2 m temp/dew point and precip; GFS/RRFS have 1000–700 mb at fine spacing, ECMWF and the ensembles only 1000/925/850/700 (RH 925/850/700). `research/drizzle/` holds the scripts and README to re-calibrate against METAR history (they score the app's own `drizzle_tiers`).

## Forecast archive

`scripts/archive_forecasts.py`, run daily at 13:37 UTC by
`.github/workflows/archive-forecasts.yml` (also manually via
`workflow_dispatch`), builds the verification history for a future
skill-weighted Grand Ensemble. Scope: KCMH, temperature. It saves every
model's forecast (all ensemble members plus ECMWF/GFS/NBM deterministic) at
00/06/12/18 UTC valid times out to 14 days, KCMH observations from the IEM
ASOS archive, per-member ensemble daily (America/New_York) highs/lows for the Trends tab (`daily/KCMH/*.csv.gz`), and CRPS scores per run, model and lead bucket (day1-2, day3-5,
day6+; deterministic runs score |forecast − obs|). Output goes only to the
orphan `forecast-archive` branch (`forecasts/KCMH/*.csv.gz`,
`observations/KCMH.csv`, `scores/KCMH_temperature.csv`), so daily commits
never trigger a Streamlit Cloud redeploy. It stores raw Open-Meteo slugs plus
a `kind` column (ensemble/deterministic); translate them with `ENS_NAME_MAP` /
`DET_MODEL_SLUGS` when reading. The script is standalone; it must not import
`app.py`.

## Fog domain extract

`scripts/fog_domain.py`, run by `.github/workflows/fog-domain.yml`, crops fog-relevant fields for central Ohio (`DOMAIN`, regridded to a 0.025° grid) out of NOAA GRIB2 output: NBM core (AWS `noaa-nbm-grib2-pds`, hourly to 36 h then 3-hourly to 72 h), GFS 0.25° (AWS `noaa-gfs-bdp-pds`, hourly to 24 h then 3-hourly), RRFS 3 km (NOMADS prod, hourly to 36 h then 3-hourly to 72 h) and REFS (NOMADS `ensprod`: `prob` visibility chances under 1/4, 1 and 2 mi plus `mean` 2 m temperature/dew point; no vis_mi or wind; to 60 h). It downloads only the needed messages (HTTP byte ranges from each file's `.idx`), decodes them with eccodes (the pip wheel bundles the library; no system packages), regrids (nearest for NBM, inverse-distance for GFS) and writes `<model>.npz` plus `latest.json` to the orphan `fog-domain` branch, rebuilt as a single force-pushed commit each run so it never grows. Canonical fields: `vis_mi`, `p_vis_1mi`/`p_vis_3mi` (NBM's probability of visibility under 1/3 miles), `t2m_f`, `spread_f`, `wind_mph`, `cloud_pct`, `ceil_ft`. A model whose newest cycle is already published is skipped, so frequent triggers are cheap. Add a model by adding an entry to `MODELS` (file URL pattern, forecast hours, and which idx message is which field).

Triggering: cron-job.org calls the workflow-dispatch API a few minutes after each model cycle is out (GitHub's own `schedule` ran the daily archive 5–11 hours late); the workflow's `schedule` is only a backup. The token for that call is a fine-grained GitHub token limited to this repo with only the Actions permission, stored in cron-job.org, never in the repo.

Operational RRFS and REFS (now in `MODELS`; the Fog Domain tab offers a metric radio per model: fog level, chance of vis ≤ 1 mi, chance of dense fog; the workflow default is `gfs,nbm,rrfs,refs`, so the cron-job.org body should be just `{"ref":"main"}`): NOMADS `https://nomads.ncep.noaa.gov/pub/data/nccf/com/rrfs/prod/rrfs.YYYYMMDD/CC/` (hourly cycles, 85 hourly files to f084, `rrfs.tCCz.2dfld.3km.fFFF.conus.grib2` with `.idx`; VIS, TMP/DPT/RH 2 m, 10 m wind, LCDC, CEIL) and `.../refs/prod/refs.YYYYMMDD/{00,06}/ensprod/` (`refs.tCCz.{mean,prob,eas,...}.fNN.conus.grib2` with two-digit hours to f60; `prob` has visibility probabilities under 400/800/1600/3200/4829/6400/8049 m and ceiling probabilities). NOMADS keeps ~2 days; the operational archive bucket is `noaa-rrfs-ops-pds` on AWS (the older `noaa-rrfs-pds` is the retired prototype).

## Conventions

- **Units**: temperature is always **°F**, wind is always **mph**. Pass
  `temperature_unit=fahrenheit` (and the equivalent wind unit param) when
  calling Open-Meteo rather than converting client-side after the fact.
- **Colors**: model colors follow the Okabe-Ito CVD-safe palette and are
  centralized in `MODEL_CONFIG`, grouped by modeling center: ECMWF cool (ECMWF blue, EPS sky blue, AIFS bluish green), NOAA warm (GFS vermilion, GEFS orange, RRFS wine), WeatherNext reddish purple, NBM mid-gray, Grand Ensemble black. Read
  them with `model_color(name)` rather than indexing `MODEL_CONFIG` or
  hardcoding a hex value: it turns black into white in dark mode
  (`is_dark_theme()`, via `st.context.theme`), so the Grand Ensemble stays
  visible on a dark chart background.
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

## Project manager (log items for Justin)
Justin tracks work in his project manager at https://justin-project-manager.fly.dev. Log things he
should see there, instead of only mentioning them in chat. This repo is `justingehrts/model-viewer`.

- **At the start of a session,** read the API guide (current fields, kinds and examples), then this
  repo's open follow-ups, and act on those that are in scope:
  `curl -s https://justin-project-manager.fly.dev/api/v1`
  `curl -s "https://justin-project-manager.fly.dev/api/v1/items?repo=justingehrts/model-viewer&open=1&kind=followup"`
- **What to log:** `todo` = something only Justin can do · `idea` = an improvement noticed but not
  asked for · `bug` = a problem spotted but not fixed · `followup` = work a later session should
  pick up. Check the open list first. One item per thing, short titles, details in notes. If you
  made an artifact, PR or doc for an item, attach its link (see the guide).
- **Judgment:** log what Justin would want to find later, not routine progress. No secrets,
  personal data or large pastes. Before ending a session with loose ends, log them as `followup`
  or `todo` and say so in chat.
- **Auth:** the network proxy adds the token for host `justin-project-manager.fly.dev`. If a request
  comes back 401, ask for that credential (request_credential tool, Bearer). Never print, log or
  commit the token.
- **Command shape (so this repo's permission rules allow it):** one line per call, with the full
  `https://justin-project-manager.fly.dev/api/v1...` address written right after `curl -s` (GET,
  quoted when it has `?` or `&`), `curl -s -X POST`, or `curl -s -X PATCH`. No variables for the
  address, no line continuations, nothing chained with `&&` or `;`.
- Sessions can update or finish only items they created; Justin's tasks are read-only and nothing
  can be deleted.

## Fog Domain map lines

`data/fog_map_lines.json` (built by `scripts/build_map_lines.py` from Census county boundaries and TIGER interstates, clipped to the domain and simplified; ~90 KB) is drawn on the Fog Domain map by `load_map_lines`.

## Fog site calibration

`research/fog/calibrate_sites.py` (IEM ASOS 2022–26, 20 `FOG_SITES`) prints `FOG_SITE_PROB`/`FOG_SITE_CLIMO`, pasted into `app.py`: per-site chance of fog (vis ≤ 1 mi or FG) by ingredient tier of dew-point spread and wind (`FOG_SITE_TIERS`), shrunk toward the all-site rate. The Fog Domain tab draws it as "site-calibrated fog chance" (`site_fog_chance`) for models with spread and wind (not REFS). Site ranking repeats across years (Spearman 0.92). Checked on KCMH: GFS-derived tiers gave 24% at the top tier versus 13% from observations, so model-based figures are a bit rarer-but-stronger; model-based per-site calibration is a possible later step.
