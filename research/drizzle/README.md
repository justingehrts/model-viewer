# Drizzle / freezing drizzle calibration

Scripts used to calibrate the tier thresholds in `app.py` (`DRIZZLE_TIERS`, `DRIZZLE_PROB`, `FZDZ_WEIGHT`).
They compare Open-Meteo's historical GFS profiles (historical-forecast API) with METAR present weather
(Iowa Environmental Mesonet ASOS archive) and score the app's own `drizzle_tiers` function, so a threshold
change in `app.py` is tested directly. Everything is keyless. Data files are not committed.

Run from this folder (needs `requests` and `pandas`):

1. `python fz_tasks.py` -- downloads METARs (`iem_<STN>.csv`) for KCMH + DSM/ORD/DTW/DAY/IND and picks the
   30-day windows that contain freezing drizzle (writes `fz_tasks.json`; KCMH is always fetched in full below).
2. `python fetch_history.py` -- GFS profiles for KCMH, every Oct-Apr window since 2022 (`small/`, ~3 requests per window).
3. `python fz_run.py 0`, `1`, `2` (can run in parallel) -- GFS profiles for the freezing-drizzle windows at the other stations.
4. `python backtest.py` -- drizzle tier frequencies per station and freezing drizzle tier precision/POD.

Open-Meteo throttles bulk history requests (429/timeouts); the scripts retry and skip files that already exist.

Findings (Nov 2026): the old "shallow cloud, dry air aloft" rule had no skill (AUC 0.55); a saturated surface
(2 m temp - dew point <= 2 F), RH >= 95% at 925 mb / >= 90% at 850 mb and light model QPF separate drizzle hours
(tier frequencies at KCMH: 0.1% / 2.5% / 11% / 24%), and the temperature at the top of the saturated layer
grades freezing drizzle (warmer than -5 C: ~28% of hours at six stations, colder: 3-8%). Freezing drizzle windows were chosen
for having freezing drizzle, so absolute freezing drizzle percentages are inflated; read them as ratios.
