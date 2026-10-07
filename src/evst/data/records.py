from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from evst.data.region import CRS, grid, region
from evst.paths import INTERIM, PROCESSED, RAW

EBIRD = "4fa7b334-ce0d-4e88-aaae-2e0c138d049e"
INAT = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
CLEAN = INTERIM / "wg_records_clean.parquet"


def taxon_group(df: pd.DataFrame) -> pd.Series:
    c, o = df["class"].fillna(""), df["order"].fillna("")
    g = np.select(
        [c == "Aves", c == "Amphibia", c.isin(["Squamata", "Testudines", "Crocodylia", "Reptilia"]),
         c == "Mammalia", o == "Lepidoptera", o == "Odonata", c == "Insecta", c == "Arachnida",
         c.isin(["Magnoliopsida", "Liliopsida", "Polypodiopsida", "Pinopsida", "Lycopodiopsida", "Cycadopsida"]),
         c.isin(["Actinopterygii"])],
        ["birds", "amphibians", "reptiles", "mammals", "butterflies_moths", "odonates", "other_insects",
         "arachnids", "plants", "fishes"], default="other")
    return pd.Series(g, index=df.index)


def source(df: pd.DataFrame) -> pd.Series:
    b = df["basisofrecord"].fillna("")
    s = np.select([df.datasetkey == EBIRD, df.datasetkey == INAT,
                   b.isin(["PRESERVED_SPECIMEN", "FOSSIL_SPECIMEN", "MATERIAL_SAMPLE", "LIVING_SPECIMEN"]),
                   b.isin(["HUMAN_OBSERVATION", "OBSERVATION", "MACHINE_OBSERVATION"])],
                  ["eBird", "iNaturalist", "specimen", "other_observation"], default="other")
    return pd.Series(s, index=df.index)


def _decimals(x: pd.Series) -> pd.Series:
    s = x.abs().map(lambda v: f"{v:.6f}".rstrip("0"))
    return s.str.split(".").str[1].str.len().fillna(0)


def load_inat() -> pd.DataFrame:
    d = RAW / "inat"
    obs = pd.read_csv(d / "wg_observations.tsv", sep="	", dtype={"observer_id": "Int64", "taxon_id": "Int64"})
    obs = obs[obs.quality_grade == "research"]
    tx = pd.read_csv(d / "taxa.csv.gz", sep="	", usecols=["taxon_id", "ancestry", "rank", "name"])
    name = dict(zip(tx.taxon_id, tx.name))
    rank = dict(zip(tx.taxon_id, tx["rank"]))
    anc = dict(zip(tx.taxon_id, tx.ancestry))

    def lineage(t):
        out = {}
        a = anc.get(t)
        for i in ([int(x) for x in a.split("/")] if isinstance(a, str) else []) + [t]:
            if rank.get(i) in ("kingdom", "phylum", "class", "order", "family", "genus", "species"):
                out[rank[i]] = name[i]
        return out

    lin = {t: lineage(t) for t in obs.taxon_id.dropna().unique()}
    L = pd.DataFrame.from_dict(lin, orient="index")
    obs = obs.join(L, on="taxon_id")
    obs["taxonrank"] = obs.taxon_id.map(rank).str.upper()
    obs["taxonrank"] = obs.taxonrank.replace({"SUBSPECIES": "SPECIES", "VARIETY": "SPECIES", "FORM": "SPECIES"})
    return pd.DataFrame({
        "gbifid": obs.observation_uuid, "datasetkey": INAT, "kingdom": obs.get("kingdom"), "phylum": obs.get("phylum"),
        "class": obs.get("class"), "order": obs.get("order"), "family": obs.get("family"), "genus": obs.get("genus"),
        "species": obs.get("species"), "taxonrank": obs.taxonrank, "decimallatitude": obs.latitude,
        "decimallongitude": obs.longitude, "coordinateuncertaintyinmeters": obs.positional_accuracy,
        "eventdate": pd.to_datetime(obs.observed_on, errors="coerce"),
        "year": pd.to_datetime(obs.observed_on, errors="coerce").dt.year, "basisofrecord": "HUMAN_OBSERVATION",
        "institutioncode": "iNaturalist", "recordedby": obs.observer_id.astype(str).map(lambda v: [v]),
        "issue": None, "occurrencestatus": "PRESENT", "license": "mixed"})


def raw_records() -> pd.DataFrame:
    parts = [load_inat()] if (RAW / "inat" / "wg_observations.tsv").exists() else []
    snaps = sorted((RAW / "gbif").glob("wg_bbox_*.parquet"))
    if snaps:
        g = pd.concat([pd.read_parquet(p) for p in snaps], ignore_index=True)
        if parts:
            g = g[g.datasetkey != INAT]
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def clean(df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = raw_records() if df is None else df
    log = {"raw": len(df)}
    df = df[(df.taxonrank.isin(["SPECIES", "SUBSPECIES", "VARIETY", "FORM"])) & df.species.notna()]
    log["species_level"] = len(df)
    df = df[(df.occurrencestatus.fillna("PRESENT") == "PRESENT") & df.year.between(1950, 2026)]
    log["present_1950_2026"] = len(df)
    df = df[df.coordinateuncertaintyinmeters.isna() | (df.coordinateuncertaintyinmeters <= 5000)]
    log["uncertainty_le_5km"] = len(df)
    df = df[(_decimals(df.decimallatitude) >= 2) & (_decimals(df.decimallongitude) >= 2)
            & (df.decimallatitude != df.decimallongitude) & (df.decimallatitude != 0)]
    log["precision_ge_2dp"] = len(df)
    df = df.assign(source=source(df), group=taxon_group(df))
    hot = (df[df.source == "specimen"].groupby(["decimallatitude", "decimallongitude"])
           .agg(n=("gbifid", "size"), s=("species", "nunique")))
    hot = hot[(hot.n > 1000) & (hot.s > 50)].index
    df = df[~df.set_index(["decimallatitude", "decimallongitude"]).index.isin(hot)]
    log["no_institution_hotspots"] = len(df)
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.decimallongitude, df.decimallatitude), crs=4326).to_crs(CRS)
    inside = g.within(region().geometry.iloc[0])
    g = g[inside]
    log["inside_region"] = len(g)
    tr, shape, mask = grid()
    col = ((g.geometry.x - tr.c) / tr.a).astype(int)
    row = ((g.geometry.y - tr.f) / tr.e).astype(int)
    out = pd.DataFrame(g.drop(columns="geometry")).assign(x=g.geometry.x.values, y=g.geometry.y.values,
                                                          row=row.values, col=col.values)
    out = out[(out.row >= 0) & (out.row < shape[0]) & (out.col >= 0) & (out.col < shape[1])]
    out["cell"] = out.row * shape[1] + out.col
    joiner = lambda v: "|".join(map(str, v)) if isinstance(v, (list, np.ndarray)) and len(v) else None
    out["recordedby"] = out["recordedby"].map(joiner)
    out["issue"] = out["issue"].map(joiner)
    out.to_parquet(CLEAN)
    pd.Series(log).to_csv(PROCESSED.parent.parent / "results" / "E01_cleaning_log.csv")
    return out


def load() -> pd.DataFrame:
    return pd.read_parquet(CLEAN) if CLEAN.exists() else clean()


if __name__ == "__main__":
    d = clean()
    print(d.groupby(["group", "source"]).size().unstack(fill_value=0))
