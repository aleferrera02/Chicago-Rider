"""Local R inputs take precedence. The public fallback downloads October 2024 only."""
from contextlib import closing
from datetime import date, timedelta
from io import BytesIO
from http.client import IncompleteRead
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import gzip
import json
import os
import re
import time

import numpy as np
import pandas as pd

from .geography import trip_routes

PORTAL = "https://data.cityofchicago.org/resource/"
TRIPS_ID = "n26f-ihde"
BOUNDARIES_ID = "igwz-8jzy"
START, END = date(2024, 10, 1), date(2024, 11, 1)
NUMERIC = [
    "fare", "trip_miles", "additional_charges", "trip_seconds", "trips_pooled",
    "pickup_community_area", "dropoff_community_area", "pickup_centroid_latitude",
    "pickup_centroid_longitude", "dropoff_centroid_latitude", "dropoff_centroid_longitude",
]
PUBLIC_FIELDS = ["trip_start_timestamp", "tip", "shared_trip_match", *NUMERIC]
DROP_IN_R = {
    "trip_id", "trip_end_timestamp", "pickup_census_tract", "dropoff_census_tract",
    "trip_total", "shared_trip_authorized", "tip", "shared_trip_match",
    "percent_time_chicago", "percent_distance_chicago",
}
# Friday/Saturday/Sunday dates in the original script, plus Halloween.
WEEKEND_DATES = {4, 5, 6, 11, 12, 13, 18, 19, 20, 25, 26, 27, 31}


def _get_bytes(resource, params):
    url = PORTAL + resource + "?" + urlencode(params)
    headers = {"User-Agent": "ChicagoTippingAnalysis/1.0", "Accept-Encoding": "gzip"}
    if token := os.getenv("SOCRATA_APP_TOKEN"):
        headers["X-App-Token"] = token
    for attempt in range(4):
        try:
            with closing(urlopen(Request(url, headers=headers), timeout=120)) as response:
                payload = response.read()
                return gzip.decompress(payload) if response.headers.get("Content-Encoding") == "gzip" else payload
        except HTTPError as error:
            if error.code not in {408, 429, 500, 502, 503, 504} or attempt == 3:
                detail = error.read(600).decode(errors="replace")
                raise RuntimeError(f"Chicago Data Portal returned HTTP {error.code}: {detail}") from error
        except (OSError, URLError, IncompleteRead, EOFError) as error:
            if attempt == 3:
                raise RuntimeError("Could not reach the Chicago Data Portal. Supply the original "
                                   "CSV and GeoJSON in data/ or retry the download.") from error
        time.sleep(2 ** (attempt + 1))


def _download_day(day, destination, expected=None):
    """Stable pagination within one day. Commit only complete, counted daily downloads."""
    next_day = day + timedelta(days=1)
    where = (f"trip_start_timestamp >= '{day}T00:00:00' AND "
             f"trip_start_timestamp < '{next_day}T00:00:00'")
    resource = TRIPS_ID + ".csv"
    if expected is None:
        counts = _get_bytes(resource, {"$select": "count(*) as n", "$where": where})
        expected = int(pd.read_csv(BytesIO(counts))["n"].iloc[0])
    size, written = 250_000, 0
    with NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as temp:
        pending = Path(temp.name)
    try:
        with gzip.open(pending, "wt", encoding="utf-8", newline="") as out:
            if expected == 0:
                raise ValueError(f"The public portal returned no trips for {day}.")
            while written < expected:
                query = {"$select": ",".join(PUBLIC_FIELDS), "$where": where,
                         "$order": "trip_start_timestamp,trip_id", "$limit": size, "$offset": written}
                block = pd.read_csv(BytesIO(_get_bytes(resource, query)), dtype=str)
                if block.columns.tolist() != PUBLIC_FIELDS or block.empty:
                    raise ValueError(f"Incomplete public data for {day}, at row {written:,}.")
                block.to_csv(out, index=False, header=written == 0)
                written += len(block)
        if written != expected:
            raise ValueError(f"The portal row count for {day} changed during download. Rerun.")
        pending.replace(destination)
    finally:
        pending.unlink(missing_ok=True)


def resolve_inputs(data_dir, trips_file=None, boundaries_file=None, download=True):
    """Return input paths and provenance. Completed daily downloads can be reused."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if trips_file is not None and not Path(trips_file).is_file():
        raise FileNotFoundError(trips_file)
    if boundaries_file is not None and not Path(boundaries_file).is_file():
        raise FileNotFoundError(boundaries_file)
    geo = Path(boundaries_file) if boundaries_file is not None else data_dir / "chicago_boundaries.geojson"
    if not geo.exists():
        if not download:
            raise FileNotFoundError(f"Place the community-area GeoJSON at {geo}.")
        payload = _get_bytes(BOUNDARIES_ID + ".geojson", {"$limit": 100})
        if len(json.loads(payload).get("features", [])) != 77:
            raise ValueError("Expected 77 Chicago community-area features.")
        with NamedTemporaryFile(dir=data_dir, delete=False) as tmp:
            tmp.write(payload)
        Path(tmp.name).replace(geo)
    if trips_file is None:
        trips_file = next((p for p in (data_dir / "Chicago_rides_october24.csv",
                                       data_dir / "Chicago_rides_october24.csv.gz") if p.is_file()), None)
    if trips_file is not None:
        return [Path(trips_file)], geo, "Local trip file"
    folder = data_dir / "public_october_2024"
    folder.mkdir(exist_ok=True)
    files = []
    for day in pd.date_range(START, END, inclusive="left").date:
        destination = folder / f"{day}.csv.gz"
        if not destination.exists():
            if not download:
                raise FileNotFoundError("Provide data/Chicago_rides_october24.csv or enable DOWNLOAD.")
            print(f"Downloading rides for {day} ...", flush=True)
            _download_day(day, destination)
        files.append(destination)
    return files, geo, "City of Chicago public October 2024 export (cached by day)"


def _names(columns):
    return [re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_") for c in columns]


def _binary(values):
    if pd.api.types.is_bool_dtype(values):
        return values.astype("float64")
    lookup = {"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0, "t": 1, "f": 0}
    result = values.astype("string").str.strip().str.lower().map(lookup)
    if (result.isna() & values.notna()).any():
        raise ValueError(f"Unrecognised boolean in column {values.name}.")
    return result


def _prepare_chunk(raw):
    raw.columns = _names(raw.columns)
    missing = set(NUMERIC + ["trip_start_timestamp"]) - set(raw)
    if missing:
        raise ValueError(f"Missing trip fields: {sorted(missing)}")
    df = raw[NUMERIC].apply(pd.to_numeric, errors="coerce")
    # The old file already contained the two flags. The public file contains their inputs.
    if "tip_given" in raw:
        df["tip_given"] = _binary(raw["tip_given"])
    elif "tip" in raw:
        tip = pd.to_numeric(raw["tip"], errors="coerce")
        df["tip_given"] = (tip > 0).astype(float).where(tip.notna())
    else:
        raise ValueError("A Tip.Given or Tip column is required.")
    if "trip_pooled" in raw:
        shared = _binary(raw["trip_pooled"])
    elif "shared_trip_match" in raw:
        shared = _binary(raw["shared_trip_match"])
    else:
        shared = (df["trips_pooled"] > 1).where(df["trips_pooled"].notna())
    stamp = pd.to_datetime(raw["trip_start_timestamp"], format="mixed", errors="coerce")
    valid_time = stamp.ge("2024-10-01") & stamp.lt("2024-11-01")
    # R calls na.omit after dropping its explicit column list. Preserve NA checks on
    # additional retained columns in an original extract, even if they are not predictors.
    other = set(raw) - DROP_IN_R - set(NUMERIC) - {"trip_start_timestamp", "tip_given", "trip_pooled"}
    # CSV row indices saved by write.csv are metadata, not trip fields.
    other = [col for col in other if col and not col.startswith("unnamed")]
    keep = valid_time & shared.notna() & np.isfinite(df).all(axis=1)
    if other:
        keep &= raw[other].notna().all(axis=1)
    df = df.loc[keep].copy()
    stamp = stamp.loc[keep]
    hour = stamp.dt.hour.to_numpy(dtype=np.int8)
    df["hour_start"] = hour
    df["day_of_week"] = stamp.dt.dayofweek.astype("int8")
    df["part_of_day"] = np.select([(hour >= 7) & (hour < 15), (hour >= 15) & (hour < 23)],
                                  [1, 2], default=3).astype("int8")
    df["weekend"] = stamp.dt.day.isin(WEEKEND_DATES).astype("int8")
    for col, dtype in {"pickup_community_area": "int16", "dropoff_community_area": "int16",
                       "trips_pooled": "int32", "tip_given": "int8"}.items():
        df[col] = df[col].astype(dtype)
    df["route"] = trip_routes(df.pickup_community_area, df.dropoff_community_area)
    return df, int(valid_time.sum())


def load_trips(paths, chunk_size=100_000):
    """Chunked CSV parsing; retain exact numeric inputs used by the R models."""
    frames, n_read, n_october = [], 0, 0
    for path in paths:
        for raw in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
            n_read += len(raw)
            frame, in_window = _prepare_chunk(raw)
            n_october += in_window
            if len(frame):
                frames.append(frame)
    if not frames:
        raise ValueError("No complete October 2024 trips remained. Check the dates and columns.")
    trips = pd.concat(frames, ignore_index=True)
    # R's as.numeric(as.factor(Trips.Pooled)) yields the one-based rank of each level.
    levels = np.sort(trips.trips_pooled.unique())
    trips["pool_index"] = np.searchsorted(levels, trips.trips_pooled).astype("int32") + 1
    audit = pd.Series({"Rows read": n_read, "October timestamps": n_october,
                       "Complete October trips": len(trips),
                       "Dropped within October": n_october - len(trips),
                       "Trips with a recorded tip": int(trips.tip_given.sum()),
                       "Recorded tip rate": float(trips.tip_given.mean())}, name="Value")
    return trips, audit


class PlotSampler:
    """R's default Mersenne-Twister and rejection sampler, for the two EDA samples.

    Keeping this stream across calls reproduces set.seed(30) followed by the
    two sample(..., replace=FALSE) calls when the cleaned row order is identical.
    """
    def __init__(self, seed=30):
        state = seed & 0xffffffff
        for _ in range(50):
            state = (69069*state + 1) & 0xffffffff
        words = []
        for _ in range(625):
            state = (69069*state + 1) & 0xffffffff
            words.append(state)
        self.mt = np.random.RandomState()
        self.mt.set_state(("MT19937", np.asarray(words[1:], dtype="uint32"), 624, 0, 0.))
        self.short_words = self._random_shorts()

    def _random_shorts(self):
        while True:
            block = self.mt.randint(0, 2**32, size=8192, dtype="uint32") >> 16
            yield from block.tolist()

    def indices(self, population, size):
        if not 0 <= size <= population:
            raise ValueError("Sample size must be between zero and the population size.")
        pool = np.arange(population, dtype=np.intp)
        chosen = np.empty(size, dtype=np.intp)
        for i in range(size):
            remaining = population-i
            bits = (remaining-1).bit_length()
            while True:
                value = 0
                for _ in range(0, bits+1, 16):
                    value = (value << 16) | next(self.short_words)
                index = value & ((1 << bits)-1)
                if index < remaining:
                    break
            chosen[i] = pool[index]
            pool[index] = pool[remaining-1]
        return chosen
