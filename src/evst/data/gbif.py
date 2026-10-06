from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

from evst.paths import RAW

DIR = RAW / "gbif"
DIR.mkdir(parents=True, exist_ok=True)
API = "https://api.gbif.org/v1/occurrence/search"
FIELDS = ["gbifID", "datasetKey", "basisOfRecord", "occurrenceStatus", "kingdom", "phylum", "class", "order",
          "family", "genus", "species", "taxonKey", "speciesKey", "taxonRank", "decimalLatitude",
          "decimalLongitude", "coordinateUncertaintyInMeters", "eventDate", "year", "month",
          "recordedBy", "institutionCode", "countryCode", "issues"]
CLASSES = {"Amphibia": 131, "Squamata": 11592253, "Testudines": 11418114, "Mammalia": 359,
           "Insecta": 216, "Arachnida": 367, "Actinopterygii": 204, "Magnoliopsida": 220, "Liliopsida": 196,
           "Polypodiopsida": 7228684}


def _get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode(params, doseq=True)
    for i in range(6):
        try:
            return json.load(urllib.request.urlopen(url, timeout=60))
        except Exception as e:
            print(f"  retry {i} {type(e).__name__}: {e}", flush=True)
            time.sleep(2 ** i)
    raise RuntimeError(url)


def _page_all(base: dict) -> list[dict]:
    n = _get({**base, "limit": 0})["count"]
    if n == 0:
        return []
    if n > 99_000 and "year" not in base:
        out = []
        for y0, y1 in [(1700, 1979), (1980, 1999)] + [(y, y) for y in range(2000, 2027)]:
            out += _page_all({**base, "year": f"{y0},{y1}"})
        return out
    if n > 99_000:
        out = []
        for m in range(1, 13):
            out += _page_all({**base, "month": m, "year": base["year"] + ""})
        return out
    from concurrent.futures import ThreadPoolExecutor
    pages = lambda off: [{k: rec.get(k) for k in FIELDS} for rec in _get({**base, "limit": 300, "offset": off})["results"]]
    print(f"  {base.get('classKey')} {base.get('year', 'all')} {base.get('month', '')}: {n:,}", flush=True)
    with ThreadPoolExecutor(6) as ex:
        return [rec for page in ex.map(pages, range(0, n, 300)) for rec in page]


def api_pull(bbox: tuple[float, float, float, float]) -> None:
    x0, y0, x1, y1 = bbox
    for name, key in CLASSES.items():
        f = DIR / f"api_{name}.parquet"
        if f.exists():
            continue
        t = time.time()
        recs = _page_all({"decimalLatitude": f"{y0:.3f},{y1:.3f}", "decimalLongitude": f"{x0:.3f},{x1:.3f}",
                          "classKey": key, "hasCoordinate": "true",
                          "hasGeospatialIssue": "false", "occurrenceStatus": "PRESENT"})
        df = pd.DataFrame(recs, columns=FIELDS)
        df["issues"] = df["issues"].astype(str)
        df.to_parquet(f)
        print(f"{name}: {len(df):,} records, {time.time()-t:.0f}s", flush=True)


def fetch(key: str) -> None:
    from pygbif import occurrences as occ
    occ.download_get(key, path=str(DIR))
    print(occ.download_meta(key).get("doi"))


if __name__ == "__main__":
    if sys.argv[1] == "fetch":
        fetch(sys.argv[2])
    else:
        from evst.data.region import region
        api_pull(tuple(region().to_crs(4326).total_bounds))
