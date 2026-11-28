import requests, time, os, sys, pandas as pd
# small requests: one variable family at a time, 30-day windows, saved per (station, window, family)
LV = [1000, 975, 950, 925, 900, 850, 800, 700]
FAM = {"rh": [f"relative_humidity_{p}hPa" for p in LV], "t": [f"temperature_{p}hPa" for p in LV],
       "sfc": ["temperature_2m", "precipitation", "surface_pressure", "dew_point_2m", "wind_speed_10m", "cloud_cover_low", "visibility"]}
def get(lat, lon, a, b, vars_, tag):
    fn = f"small/{tag}.csv"
    if os.path.exists(fn): return True
    for attempt in range(5):
        try:
            r = requests.get("https://historical-forecast-api.open-meteo.com/v1/forecast", params={"latitude": lat, "longitude": lon, "start_date": a, "end_date": b,
                "hourly": ",".join(vars_), "models": "gfs_seamless", "temperature_unit": "fahrenheit", "precipitation_unit": "inch", "wind_speed_unit": "mph", "timezone": "UTC"}, timeout=40)
        except requests.RequestException as e:
            time.sleep(3 * (attempt + 1)); continue
        if r.status_code == 200:
            pd.DataFrame(r.json()["hourly"]).to_csv(fn, index=False); return True
        time.sleep(5 * (attempt + 1))
    print("FAILED", tag, flush=True); return False
if __name__ == "__main__":
    os.makedirs("small", exist_ok=True)
    stn, lat, lon = "CMH", 39.997, -82.891
    seasons = [("2022-10-01", "2023-04-30"), ("2023-10-01", "2024-04-30"), ("2024-10-01", "2025-04-30"), ("2025-10-01", "2026-04-30"), ("2026-10-01", "2026-11-26")]
    for a, b in seasons:
        d = pd.Timestamp(a)
        while d <= pd.Timestamp(b):
            e = min(d + pd.Timedelta(days=29), pd.Timestamp(b)); ds, es = d.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")
            for fam, v in FAM.items(): get(lat, lon, ds, es, v, f"{stn}_{ds}_{fam}")
            print(stn, ds, "done", flush=True); d = e + pd.Timedelta(days=1)
    print("DONE", flush=True)
