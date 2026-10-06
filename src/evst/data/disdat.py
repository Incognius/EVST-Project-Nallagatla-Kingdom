from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

from evst.paths import RAW

OSF_DATA = "https://api.osf.io/v2/nodes/kwc4v/files/osfstorage/5d6e06147215100019a0d2d2/"
DIR = RAW / "disdat"
REGIONS = ["AWT", "CAN", "NSW", "NZ", "SA", "SWI"]
CATEGORICAL = {"AWT": [], "CAN": ["ontveg"], "NSW": ["vegsys"], "NZ": ["age", "toxicats"],
               "SA": [], "SWI": ["calc"]}
GROUPS = {"AWT": ["bird", "plant"], "CAN": [None], "NSW": ["ba", "db", "nb", "ot", "ou", "rt", "ru", "sr"],
          "NZ": [None], "SA": [None], "SWI": [None]}


def _walk(url: str, rel: str = ""):
    while url:
        d = json.load(urllib.request.urlopen(url))
        for x in d["data"]:
            a = x["attributes"]
            if a["kind"] == "folder":
                yield from _walk(x["relationships"]["files"]["links"]["related"]["href"], f"{rel}{a['name']}/")
            else:
                yield f"{rel}{a['name']}", x["links"]["download"], a["size"]
        url = d["links"].get("next")


def download(include_rasters: bool = False) -> None:
    for rel, href, size in _walk(OSF_DATA):
        if rel.startswith("Environment/") and not include_rasters and not rel.endswith(".csv"):
            continue
        out = DIR / rel
        if out.exists() and out.stat().st_size == size:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(href, out)
        print(f"{rel} ({size/1e6:.1f} MB)")


def load_po(region: str) -> pd.DataFrame:
    return pd.read_csv(DIR / "Records/train_po" / f"{region}train_po.csv")


def load_bg(region: str) -> pd.DataFrame:
    return pd.read_csv(DIR / "Records/train_bg" / f"{region}train_bg.csv")


def load_pa(region: str, group: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    suf = f"_{group}" if group else ""
    pa = pd.read_csv(DIR / "Records/test_pa" / f"{region}test_pa{suf}.csv")
    env = pd.read_csv(DIR / "Records/test_env" / f"{region}test_env{suf}.csv")
    return pa, env


if __name__ == "__main__":
    import sys
    download(include_rasters="--rasters" in sys.argv)
