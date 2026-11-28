#!/usr/bin/env python3
"""Build data/fog_map_lines.json: county borders and interstates for the Fog Domain map.

Counties: Plotly's public county GeoJSON (Census cartographic boundaries).
Interstates: Census TIGER primary roads, RTTYP == 'I'.
Both are clipped to the Fog Domain box and rounded to 3 decimals (about 100 m).

    pip install pyshp requests
    python scripts/build_map_lines.py counties.json tl_2023_us_primaryroads.shp
"""
import json
import sys
import shapefile

LAT0, LON0, LAT1, LON1 = 38.0, -85.3, 41.6, -80.3   # the Fog Domain box (DOMAIN in fog_domain.py)


def inside(p):
    return LON0 <= p[0] <= LON1 and LAT0 <= p[1] <= LAT1


def rdp(pts, eps=0.002):
    """Douglas-Peucker line simplification (eps in degrees, about 200 m)."""
    if len(pts) < 3:
        return pts
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    dx, dy = x1 - x0, y1 - y0
    n = (dx * dx + dy * dy) ** 0.5 or 1e-12
    dmax, k = 0, 0
    for i in range(1, len(pts) - 1):
        d = abs(dy * (pts[i][0] - x0) - dx * (pts[i][1] - y0)) / n
        if d > dmax:
            dmax, k = d, i
    if dmax <= eps:
        return [pts[0], pts[-1]]
    return rdp(pts[:k + 1], eps)[:-1] + rdp(pts[k:], eps)


def runs(points):
    """Split a point list into runs that stay inside the box."""
    cur, out = [], []
    for p in points:
        if inside(p):
            cur.append([round(p[0], 3), round(p[1], 3)])
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return [rdp(r) for r in out if len(r) > 1]


def main(counties_path, roads_path):
    gj = json.load(open(counties_path))
    seen, counties = set(), []
    for f in gj["features"]:
        g = f["geometry"]
        rings = [r for poly in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]) for r in poly]
        if not any(inside(p) for r in rings for p in r):
            continue
        for ring in rings:
            for a, b in zip(ring[:-1], ring[1:]):   # shared borders are drawn once
                key = tuple(sorted((tuple(a), tuple(b))))
                if key in seen:
                    continue
                seen.add(key)
    # re-join the unique segments into lines per ring, dropping duplicates
    seen2, lines = set(), []
    for f in gj["features"]:
        g = f["geometry"]
        rings = [r for poly in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]) for r in poly]
        for ring in rings:
            cur = []
            for a, b in zip(ring[:-1], ring[1:]):
                key = tuple(sorted((tuple(a), tuple(b))))
                if (inside(a) or inside(b)) and key not in seen2:
                    seen2.add(key)
                    if not cur:
                        cur = [a]
                    cur.append(b)
                elif cur:
                    lines.append(cur)
                    cur = []
            if cur:
                lines.append(cur)
    counties = [r for line in lines for r in runs(line)]
    sf = shapefile.Reader(roads_path)
    names = [fl[0] for fl in sf.fields[1:]]
    roads = []
    for sr in sf.iterShapeRecords():
        rec = dict(zip(names, sr.record))
        nm = rec.get("FULLNAME", "")
        if rec.get("RTTYP") != "I" or "Bus" in nm or "Express" in nm:
            continue
        parts = list(sr.shape.parts) + [len(sr.shape.points)]
        for a, b in zip(parts[:-1], parts[1:]):
            for r in runs(sr.shape.points[a:b]):
                roads.append({"name": rec.get("FULLNAME", ""), "xy": r})
    json.dump({"counties": counties, "interstates": roads}, open("data/fog_map_lines.json", "w"), separators=(",", ":"))
    print(len(counties), "county lines,", len(roads), "interstate pieces")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
