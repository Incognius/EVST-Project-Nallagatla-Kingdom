"""E19 prep: put GeoLifeCLEF 2024 rasters, PO records and PA surveys on one 1 km grid (EPSG:3035).

Environment channels (model inputs): 14 CHELSA bioclim variables, 3 SoilGrids layers, MODIS land-cover fractions.
Access layers (never model inputs, used to measure bias leakage): roads, population, night lights, built-up,
railways, human footprint index.
"""
import json

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

from evst.paths import INTERIM, RAW

R = RAW / "glc24"
ER = R / "EnvironmentalRasters"
OUT = INTERIM / "glc24"
OUT.mkdir(parents=True, exist_ok=True)
CRS, RES = "EPSG:3035", 1000.0
MIN_PA, MIN_PO = 20, 20

po = pd.read_csv(R / "PresenceOnlyOccurrences" / "GLC24-PO-metadata-train.csv", usecols=["lat", "lon", "speciesId"])
pa = pd.read_csv(R / "PresenceAbsenceSurveys" / "GLC24-PA-metadata-train.csv", usecols=["lat", "lon", "speciesId", "surveyId"])
po, pa = po.dropna(), pa.dropna(subset=["lat", "lon", "surveyId"])
tf = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
po["x"], po["y"] = tf.transform(po.lon.values, po.lat.values)
pa["x"], pa["y"] = tf.transform(pa.lon.values, pa.lat.values)
xs, ys = np.r_[po.x, pa.x], np.r_[po.y, pa.y]
xmin, xmax = np.floor(np.percentile(xs, 0.01) / RES) * RES - 20e3, np.ceil(np.percentile(xs, 99.99) / RES) * RES + 20e3
ymin, ymax = np.floor(np.percentile(ys, 0.01) / RES) * RES - 20e3, np.ceil(np.percentile(ys, 99.99) / RES) * RES + 20e3
W, H = int((xmax - xmin) / RES), int((ymax - ymin) / RES)
TR = from_origin(xmin, ymax, RES, RES)
print(f"grid {H} x {W} = {H * W / 1e6:.1f} M cells; PO {len(po):,}; PA rows {len(pa):,}", flush=True)


def warp(path, resampling=Resampling.bilinear, band=1, src_arr=None):
    out = np.full((H, W), np.nan, np.float32)
    with rasterio.open(path) as src:
        nod = src.nodata
        if src_arr is None:
            src_arr = src.read(band).astype(np.float32)
            if nod is not None:
                src_arr[src_arr == nod] = np.nan
        reproject(src_arr, out, src_transform=src.transform, src_crs=src.crs, dst_transform=TR, dst_crs=CRS,
                  resampling=resampling, src_nodata=np.nan, dst_nodata=np.nan)
    return out


env, names = [], []
for b in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15):
    env.append(warp(ER / "Climate" / "BioClimatic_Average_1981-2010" / f"bio{b}.tif"))
    names.append(f"bio{b}")
for s in ("phh2o", "soc", "sand"):
    env.append(warp(ER / "Soilgrids" / f"crop_proj_{s}_5-15cm_mean_1000.tif"))
    names.append(f"soil_{s}")
lcp = ER / "LandCover" / "LandCover_MODIS_Terra-Aqua_500m.tif"
with rasterio.open(lcp) as src:
    lc = src.read(1)  # band 1 = IGBP classes 1-17, 255 = fill (LC_README)
    lc_nod = 255
classes = [int(c) for c in np.unique(lc) if 1 <= c <= 17]
print("land-cover classes", classes, "nodata", lc_nod, flush=True)
for c in classes:
    m = (lc == c).astype(np.float32)
    if lc_nod is not None:
        m[lc == lc_nod] = np.nan
    f = warp(lcp, Resampling.average, src_arr=m)
    if np.nanmean(f) > 0.002:  # drop classes covering < 0.2% of the grid
        env.append(f)
        names.append(f"lc_{c}")
del lc
E = np.stack(env)
land = np.isfinite(E[0]) & np.isfinite(E[1])
mu = np.array([np.nanmean(e[land]) for e in E])
sd = np.array([np.nanstd(e[land]) for e in E]) + 1e-6
Z = np.nan_to_num((E - mu[:, None, None]) / sd[:, None, None], nan=0.0)
Z[:, ~land] = 0
np.save(OUT / "stack.npy", Z.astype(np.float16))
np.save(OUT / "land_cells.npy", np.flatnonzero(land.ravel()).astype(np.int64))
print(f"stack {Z.shape}, land {land.mean():.1%}", flush=True)
del E, Z

HF = ER / "HumanFootprint"
acc_src = {"roads": HF / "detailed/Roads_WGS84.tif", "popdensity": HF / "detailed/Popdensity2010_WGS84.tif",
           "lights": HF / "detailed/Lights2009_WGS84.tif", "built": HF / "detailed/Built2009_WGS84.tif",
           "railways": HF / "detailed/Railways_WGS84.tif", "hfp": HF / "summarized/HFP2009_WGS84.tif"}
acc = {k: warp(p) for k, p in acc_src.items()}


def to_cell(df):
    col = ((df.x.values - xmin) // RES).astype(np.int64)
    row = ((ymax - df.y.values) // RES).astype(np.int64)
    ok = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    cell = np.where(ok, row * W + col, -1)
    ok &= np.where(ok, land.ravel()[np.clip(cell, 0, None)], False)
    return np.where(ok, cell, -1)


po["cell"], pa["cell"] = to_cell(po), to_cell(pa)
po, pa = po[po.cell >= 0], pa[pa.cell >= 0]
npa = pa.groupby("speciesId").surveyId.nunique()
npo = po.groupby("speciesId").size()
species = np.array(sorted(set(npa[npa >= MIN_PA].index) & set(npo[npo >= MIN_PO].index)))
spos = pd.Series(np.arange(len(species)), index=species)
print(f"species with >= {MIN_PA} PA surveys and >= {MIN_PO} PO records: {len(species)}", flush=True)
pom = po[po.speciesId.isin(species)]
np.savez(OUT / "po_cells.npz", cell=pom.cell.to_numpy(np.int64), sp=spos[pom.speciesId].to_numpy(np.int64),
         cells_all=np.unique(po.cell.to_numpy(np.int64)))
surv = pa.drop_duplicates("surveyId").set_index("surveyId")
spos_s = pd.Series(np.arange(len(surv)), index=surv.index)
Y = np.zeros((len(surv), len(species)), np.uint8)
pm = pa[pa.speciesId.isin(species)]
Y[spos_s[pm.surveyId].to_numpy(), spos[pm.speciesId].to_numpy()] = 1
A = np.column_stack([acc[k].ravel()[surv.cell.to_numpy()] for k in acc])
A = np.where(np.isfinite(A), A, np.nanmedian(A, 0))
A[:, [1, 2]] = np.log1p(np.clip(A[:, [1, 2]], 0, None))  # population and lights are heavy-tailed
A = (A - A.mean(0)) / (A.std(0) + 1e-9)
np.savez(OUT / "pa.npz", cell=surv.cell.to_numpy(np.int64), Y=Y, access=A.astype(np.float32),
         lon=surv.lon.to_numpy(), lat=surv.lat.to_numpy())
np.save(OUT / "species.npy", species)
# access at all PO cells and a land sample, for describing the observer bias itself
cells_all = np.unique(po.cell.to_numpy(np.int64))
samp = np.flatnonzero(land.ravel())
samp = np.random.default_rng(0).choice(samp, 200000, replace=False)
np.savez(OUT / "access_po_land.npz", **{f"po_{k}": v.ravel()[cells_all] for k, v in acc.items()},
         **{f"land_{k}": v.ravel()[samp] for k, v in acc.items()})
json.dump({"shape": [H, W], "transform": list(TR)[:6], "crs": CRS, "env_names": names, "access_names": list(acc),
           "n_surveys": int(len(surv)), "n_species": int(len(species)), "n_po_cells": int(len(cells_all))},
          open(OUT / "meta.json", "w"), indent=1)
print("prevalence in PA surveys: median", np.median(Y.mean(0)).round(4), "| PO cells", len(cells_all), flush=True)
