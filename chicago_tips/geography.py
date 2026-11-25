"""The community-area assignments and undirected route definitions in map.R and routes.R."""
import json

import numpy as np
import pandas as pd

REGIONS = {
    "Far North": [1, 2, 77, 3, 4, 14, 13, 12, 11, 10, 9],
    "Northwest": [17, 18, 19, 20, 15, 16],
    "North": [5, 6, 7, 21, 22],
    "West": [23, 24, 25, 26, 27, 28, 29, 30, 31],
    "Central": [8, 32, 33],
    "South": [60, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 69],
    "Southwest": [56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68],
    "Far Southwest": [70, 71, 72, 73, 74, 75],
    "Far Southeast": [44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55],
    "Airport": [76],
}
REGION_COLORS = dict(zip(REGIONS, [
    "#3F7FCC", "#C53103", "#065F42", "#FFCF3D", "#93008C",
    "#FC9100", "#CC694E", "#E37C82", "#B0928E", "#234985",
]))
AREA_REGION = {area: region for region, areas in REGIONS.items() for area in areas}
MACRO_REGIONS = {
    "North": ["Far North", "Northwest", "North"],
    "South": ["South", "Far Southeast"],
    "Central": ["Central"],
    "West": ["West", "Southwest", "Far Southwest"],
    "Airport": ["Airport"],
}
ROUTES = [
    "North-North", "South-South", "Central-Central", "West-West", "Airport-Airport",
    "North-South", "North-Central", "Airport-Central", "North-West", "South-West",
    "Central-West", "Central-South", "North-Airport", "South-Airport", "West-Airport",
]
AREA_MACRO = {
    area: index for index, regions in enumerate(MACRO_REGIONS.values())
    for region in regions for area in REGIONS[region]
}
_ZONE = {name: i for i, name in enumerate(MACRO_REGIONS)}
_ROUTE_ID = np.full((6, 6), len(ROUTES), dtype=np.int8)
for route_id, name in enumerate(ROUTES):
    a, b = (_ZONE[zone] for zone in name.split("-"))
    _ROUTE_ID[a, b] = _ROUTE_ID[b, a] = route_id


def trip_routes(pickup, dropoff):
    start = pickup.map(AREA_MACRO).fillna(5).to_numpy(dtype=np.int8)
    end = dropoff.map(AREA_MACRO).fillna(5).to_numpy(dtype=np.int8)
    return pd.Categorical.from_codes(_ROUTE_ID[start, end], categories=ROUTES + ["Other"])


def read_boundaries(path):
    """Read the polygons in a City of Chicago WGS84 GeoJSON, including interior rings."""
    with open(path, encoding="utf-8-sig") as source:
        obj = json.load(source)
    areas = {}
    for feature in obj["features"]:
        props = {k.lower(): v for k, v in feature["properties"].items()}
        area = next((props[k] for k in ("area_num_1", "area_numbe", "area_number", "area_num")
                     if k in props and props[k] is not None), None)
        if area is None:
            raise ValueError(f"No community-area number in boundary properties {list(props)}.")
        geometry = feature["geometry"]
        if geometry["type"] not in {"Polygon", "MultiPolygon"}:
            raise ValueError(f"Expected polygonal boundaries, found {geometry['type']}.")
        polygons = geometry["coordinates"]
        if geometry["type"] == "Polygon":
            polygons = [polygons]
        areas.setdefault(int(area), []).extend(
            [[np.asarray(ring, dtype=float)[:, :2] for ring in polygon] for polygon in polygons]
        )
    missing = set(AREA_REGION) - set(areas)
    if missing:
        raise ValueError(f"Missing community-area boundaries: {sorted(missing)}")
    return areas
