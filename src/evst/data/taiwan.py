from __future__ import annotations

import json
import pickle
import urllib.parse
import urllib.request

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree
from shapely.geometry import box

from evst.data import rasters, records
from evst.data.kba import ACC, ENV
from evst.paths import INTERIM, PROCESSED, RAW

CRS = "EPSG:32651"
D = RAW / "taiwan"
BOUNDS = (120.0, 21.8, 122.1, 25.4)
STACK = PROCESSED / "tw_stack_1km.tif"
CLEAN = INTERIM / "tw_birds_clean.parquet"
POINTS = INTERIM / "tw_bbs_point_visits.parquet"
RADIUS = 2000.0
TRUTH_YEARS = (2017, 2022)
NAMES = INTERIM / "tw_name_map.csv"


def grid():
    g = gpd.GeoSeries([box(*BOUNDS)], crs=4326).to_crs(CRS)
    xmin, ymin, xmax, ymax = g.total_bounds
    xmin, ymin = np.floor(xmin / 1000) * 1000, np.floor(ymin / 1000) * 1000
    xmax, ymax = np.ceil(xmax / 1000) * 1000, np.ceil(ymax / 1000) * 1000
    w, h = int((xmax - xmin) / 1000), int((ymax - ymin) / 1000)
    tr = from_origin(xmin, ymax, 1000, 1000)
    inbox = rasterize([(g.iloc[0], 1)], out_shape=(h, w), transform=tr, fill=0, dtype="uint8").astype(bool)
    elev = rasters._warp(D / "tile_35_wc2.1_30s_elev.tif", 1, tr, (h, w), crs=CRS)
    return tr, (h, w), inbox & ~np.isnan(elev)


def build_stack():
    tr, shape, mask = grid()
    layers = {}
    layers.update(rasters.worldclim(tr, shape, D, "tile_35", CRS))
    layers.update(rasters.worldcover_fractions(tr, shape, tiles=sorted(D.glob("ESA_WorldCover_*_Map.tif")), crs=CRS))
    built = np.nan_to_num(layers["wc_built"])
    layers["pop_log"] = np.log1p(uniform_filter(built, 5) * 1000).astype(np.float32)
    bounds = np.array(BOUNDS) + np.array([-0.3, -0.3, 0.3, 0.3])
    layers.update(rasters.osm(tr, shape, zips=[D / "taiwan-latest-free.shp.zip"], bounds=bounds, crs=CRS, road100=None))
    names = list(layers)
    with rasterio.open(STACK, "w", driver="GTiff", height=shape[0], width=shape[1], count=len(names),
                       dtype="float32", crs=CRS, transform=tr, nodata=np.nan, compress="deflate") as dst:
        for i, n in enumerate(names, 1):
            dst.write(np.where(mask, layers[n], np.nan).astype(np.float32), i)
            dst.set_band_description(i, n)
    return STACK


def load_stack():
    if not STACK.exists():
        build_stack()
    with rasterio.open(STACK) as src:
        arr = src.read()
        names = list(src.descriptions)
        tr = src.transform
    keep = [names.index(k) for k in ENV + ACC]
    mask = ~np.isnan(arr[keep]).any(0)
    return {n: arr[i] for i, n in enumerate(names)}, tr, mask


def birds() -> pd.DataFrame:
    if CLEAN.exists():
        return pd.read_parquet(CLEAN)
    o = records.load_inat(D / "tw_observations.tsv")
    o = o[(o["class"] == "Aves") & (o.taxonrank == "SPECIES") & o.species.notna() & o.year.between(1950, 2026)]
    o = o[o.coordinateuncertaintyinmeters.isna() | (o.coordinateuncertaintyinmeters <= 5000)]
    x, y = Transformer.from_crs(4326, CRS, always_xy=True).transform(o.decimallongitude.values, o.decimallatitude.values)
    _, tr, _ = load_stack()
    o = o.assign(x=x, y=y, row=((y - tr.f) / tr.e).astype(int), col=((x - tr.c) / tr.a).astype(int))
    o["recordedby"] = o.recordedby.map(lambda v: "|".join(v))
    o = o[["gbifid", "species", "decimallatitude", "decimallongitude", "eventdate", "year", "recordedby", "x", "y",
           "row", "col"]]
    o.to_parquet(CLEAN)
    return o


def point_visits() -> pd.DataFrame:
    if POINTS.exists():
        return pd.read_parquet(POINTS)
    e = pd.read_csv(D / "bbs" / "event.txt", sep="\t", dtype=str,
                    usecols=["eventID", "eventDate", "eventTime", "locationID", "decimalLatitude", "decimalLongitude"])
    e = e[(e.eventID.str.count("_") == 4) & ~e.eventID.str.contains(":") & e.decimalLatitude.notna()]
    o = pd.read_csv(D / "bbs" / "occurrence.txt", sep="\t", dtype=str, usecols=["eventID", "scientificName"])
    e = e.assign(lat=e.decimalLatitude.astype(float), lon=e.decimalLongitude.astype(float),
                 year=e.eventID.str.split("_").str[1].astype(int), site=e.locationID.str.split("_").str[0],
                 hour=pd.to_numeric(e.eventTime.str[:2], errors="coerce"))
    e = e[(e.lon >= BOUNDS[0]) & (e.lon <= BOUNDS[2]) & (e.lat >= BOUNDS[1]) & (e.lat <= BOUNDS[3])]
    o = o.dropna(subset=["scientificName"])
    sp = o[o.eventID.isin(e.eventID)].groupby("eventID").scientificName.apply(lambda s: sorted(set(s)))
    e = e.assign(species=e.eventID.map(sp)).drop(columns=["decimalLatitude", "decimalLongitude"])
    e["species"] = e.species.map(lambda v: v if isinstance(v, list) else [])
    e.to_parquet(POINTS)
    return e


def name_map(names) -> dict:
    m = pd.read_csv(NAMES) if NAMES.exists() else pd.DataFrame(columns=["name", "accepted"])
    todo = sorted(set(names) - set(m.name))
    rows = []
    for n in todo:
        acc = None
        if not any(t in n for t in ("(", " sp.", "/", " x ", "idae")) and len(n.split()) >= 2:
            r = json.load(urllib.request.urlopen("https://api.gbif.org/v1/species/match?class=Aves&name=" + urllib.parse.quote(n)))
            if r.get("matchType") != "NONE" and r.get("rank") in ("SPECIES", "SUBSPECIES") and r.get("class") == "Aves":
                acc = r.get("species")
        rows.append({"name": n, "accepted": acc})
    if rows:
        m = pd.concat([m, pd.DataFrame(rows)], ignore_index=True)
        m.to_csv(NAMES, index=False)
    return {k: v for k, v in zip(m.name, m.accepted) if isinstance(v, str)}


def endemics() -> set:
    f = D / "taxon_tw_share.csv"
    if not f.exists():
        return set()
    s = pd.read_csv(f)
    s = s[(s.n_all >= 30) & (s.n_tw / s.n_all >= 0.9)]
    tx = pd.read_csv(RAW / "inat" / "taxa.csv.gz", sep="\t", usecols=["taxon_id", "ancestry", "rank", "name"])
    tx = tx[(tx["rank"] == "species") & tx.ancestry.fillna("").str.contains("/3/")]
    names = tx[tx.taxon_id.isin(s.taxon_id)].name
    nm = name_map(names)
    return {nm[n] for n in names if n in nm}


def build(end: int = 2022, min_cells: int = 20, start: int | None = None, truth_years=TRUTH_YEARS) -> dict:
    f = INTERIM / f"tw_setup_{start or 0}_{end}_{min_cells}_{truth_years[0]}_{truth_years[1]}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    layers, tr, mask = load_stack()
    H, W = mask.shape
    rr, cc = np.nonzero(mask)
    land = rr * W + cc
    ux, uy = tr * (cc + 0.5, rr + 0.5)
    plon, plat = Transformer.from_crs(CRS, 4326, always_xy=True).transform(ux, uy)
    X = np.column_stack([layers[k][rr, cc] for k in ENV]).astype(np.float32)
    A = np.column_stack([np.log1p(layers[k][rr, cc]) if k.startswith("d_") else layers[k][rr, cc] for k in ACC]).astype(np.float32)
    pv = point_visits()
    pv = pv[pv.year.between(*truth_years)]
    yrs = pv.groupby("site").year.nunique()
    pv = pv[pv.site.isin(yrs[yrs >= 2].index)]
    px, py = Transformer.from_crs(4326, CRS, always_xy=True).transform(pv.lon.values, pv.lat.values)
    pts = pd.DataFrame({"site": pv.site.values, "x": px, "y": py}).groupby(["site"]).apply(
        lambda g: g[["x", "y"]].drop_duplicates()).reset_index(level=0)
    tree = cKDTree(pts[["x", "y"]].values)
    dist, pi = tree.query(np.column_stack([ux, uy]))
    site_of_land = np.where(dist <= RADIUS, pts.site.values[pi], None)
    cnt = pd.Series(site_of_land[site_of_land != None]).value_counts()
    ev = pd.Index(sorted(cnt[cnt >= 8].index))
    tp = pv[pv.site.isin(ev)].explode("species").dropna(subset=["species"])
    allb = birds()
    nm = name_map(set(tp.species) | set(allb.species))
    tp = tp.assign(species=tp.species.map(nm)).dropna(subset=["species"])
    truth = tp.groupby("site").species.apply(set)
    ev = ev.intersection(truth.index)
    posl = pd.Series(np.arange(len(land)), index=land)
    b = allb.assign(species=allb.species.map(nm)).dropna(subset=["species"])
    b = b[(b.year <= end) & (b.row >= 0) & (b.row < H) & (b.col >= 0) & (b.col < W)]
    if start is not None:
        b = b[b.year >= start]
    b = b.assign(cell=b.row * W + b.col)
    b = b[b.cell.isin(posl.index)].copy()
    b["li"] = posl[b.cell].to_numpy()
    b["visit"] = b.recordedby.astype(str) + "|" + pd.to_datetime(b.eventdate).dt.normalize().astype(str)
    b["kcell"] = site_of_land[b.li.to_numpy()]
    survey_sp = set().union(*truth.values)
    matched = survey_sp & set(allb.species.map(nm).dropna())
    ncell = b.groupby("species").li.nunique()
    cand = sorted(s for s in matched if s in ncell.index and ncell[s] >= min_cells)
    sidx = pd.Series(np.arange(len(cand)), index=cand)
    kpos = pd.Series(np.arange(len(ev)), index=ev)
    Y = np.zeros((len(ev), len(cand)), np.int8)
    for k in ev:
        for s in truth[k]:
            if s in sidx.index:
                Y[kpos[k], sidx[s]] = 1
    bk = b[b.kcell.isin(ev) & b.species.isin(cand)]
    R = np.zeros_like(Y)
    R[kpos[bk.kcell].to_numpy(), sidx[bk.species].to_numpy()] = 1
    U = b[b.kcell.isin(ev)].groupby("kcell").visit.nunique().reindex(ev, fill_value=0).to_numpy(float)
    in_ev = pd.Series(site_of_land).isin(ev).to_numpy()
    pix_li = np.flatnonzero(in_ev)
    pix_k = kpos[pd.Series(site_of_land[in_ev])].to_numpy()
    cxy = pv[pv.site.isin(ev)].groupby("site")[["lon", "lat"]].mean().loc[ev].to_numpy()
    nl = pv[pv.site.isin(ev)].groupby("site").size().reindex(ev).to_numpy()
    end_set = endemics()
    out = dict(ev=ev, cxy=cxy, X=X, A=A, land=land, lonlat=np.column_stack([plon, plat]).astype(np.float32),
               rc=np.column_stack([rr, cc]), shape=(H, W),
               birds=b[["species", "li", "kcell", "visit", "year", "recordedby"]], cand=cand, Y=Y, R=R, U=U,
               n_lists=nl, pix_li=pix_li, pix_k=pix_k, endemic=np.array([s in end_set for s in cand]), spd=None)
    f.write_bytes(pickle.dumps(out))
    return out
