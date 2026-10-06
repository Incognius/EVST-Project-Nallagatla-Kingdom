from __future__ import annotations

import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy.ndimage import distance_transform_edt

from evst.data.region import CRS, grid, region
from evst.paths import PROCESSED, RAW

STACK = PROCESSED / "stack_1km.tif"
ROAD100 = PROCESSED / "d_road_all_100m.tif"
WC_CLASSES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 70: "snow",
              80: "water", 90: "wetland", 95: "mangrove", 100: "moss"}
MAJOR = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary",
         "secondary_link", "tertiary", "tertiary_link"}


def _warp(src_path, band, tr, shape, resampling=Resampling.bilinear, src_nodata=None):
    out = np.full(shape, np.nan, np.float32)
    with rasterio.open(src_path) as src:
        reproject(rasterio.band(src, band), out, dst_transform=tr, dst_crs=CRS,
                  resampling=resampling, src_nodata=src_nodata if src_nodata is not None else src.nodata,
                  dst_nodata=np.nan)
    return out


def worldclim(tr, shape) -> dict[str, np.ndarray]:
    d = RAW / "worldclim"
    out = {f"bio{b:02d}": _warp(d / "tile_33_wc2.1_30s_bio.tif", b, tr, shape) for b in range(1, 20)}
    out["elev"] = _warp(d / "tile_33_wc2.1_30s_elev.tif", 1, tr, shape)
    return out


def worldpop(tr, shape) -> dict[str, np.ndarray]:
    dens = _warp(RAW / "worldpop" / "ind_ppp_2020_1km.tif", 1, tr, shape, Resampling.average)
    return {"pop_log": np.log1p(np.clip(dens, 0, None)).astype(np.float32)}


def worldcover_fractions(tr, shape, block: int = 100) -> dict[str, np.ndarray]:
    cache = PROCESSED.parent / "interim" / "worldcover_frac"
    cache.mkdir(parents=True, exist_ok=True)
    tiles = sorted((RAW / "worldcover").glob("*_Map.tif"))
    acc = {k: np.zeros(shape, np.float32) for k in WC_CLASSES}
    wsum = np.zeros(shape, np.float32)
    for t in tiles:
        f = cache / (t.stem + "_frac.tif")
        if not f.exists():
            with rasterio.open(t) as src:
                H, W = src.height // block, src.width // block
                frac = np.zeros((len(WC_CLASSES), H, W), np.float32)
                cls = np.array(list(WC_CLASSES))
                col = (np.arange(W * block) // block).astype(np.int64)[None, :] * 256
                for i in range(H):
                    a = src.read(1, window=((i * block, (i + 1) * block), (0, W * block)))
                    counts = np.bincount((col + a).ravel(), minlength=W * 256).reshape(W, 256)
                    frac[:, i] = (counts[:, cls] / block ** 2).T
                prof = dict(driver="GTiff", height=H, width=W, count=len(WC_CLASSES), dtype="float32",
                            crs=src.crs, transform=src.transform * src.transform.scale(block, block),
                            compress="deflate")
            with rasterio.open(f, "w", **prof) as dst:
                dst.write(frac)
        for j, c in enumerate(WC_CLASSES):
            o = _warp(f, j + 1, tr, shape, src_nodata=-1)
            ok = ~np.isnan(o)
            acc[c][ok] += o[ok]
            if j == 0:
                wsum[ok] += 1
    return {f"wc_{n}": np.where(wsum > 0, acc[c] / np.maximum(wsum, 1), np.nan)
            for c, n in WC_CLASSES.items() if c not in (70, 100)}


def _osm_layer(name: str) -> gpd.GeoDataFrame:
    parts = []
    bounds = region().to_crs(4326).total_bounds + np.array([-0.3, -0.3, 0.3, 0.3])
    for z in sorted((RAW / "osm").glob("*-free.shp.zip")):
        with zipfile.ZipFile(z) as zf:
            shp = [n for n in zf.namelist() if n == f"gis_osm_{name}_free_1.shp"]
        if shp:
            parts.append(gpd.read_file(f"zip://{z}!{shp[0]}", bbox=tuple(bounds)))
    return gpd.pd.concat(parts, ignore_index=True).to_crs(CRS)


def _dist_km(geoms, tr, shape, res) -> np.ndarray:
    if len(geoms) == 0:
        return np.full(shape, np.nan, np.float32)
    r = rasterize(((g, 1) for g in geoms), out_shape=shape, transform=tr, fill=0, dtype="uint8",
                  all_touched=True)
    return (distance_transform_edt(r == 0) * res / 1000).astype(np.float32)


def osm(tr, shape) -> dict[str, np.ndarray]:
    res = tr.a
    pad = int(30_000 / res)
    trp = from_origin(tr.c - pad * res, tr.f + pad * res, res, res)
    shp = (shape[0] + 2 * pad, shape[1] + 2 * pad)
    crop = lambda a: a[pad:-pad, pad:-pad]
    roads = _osm_layer("roads")
    places = _osm_layer("places")
    water = _osm_layer("waterways")
    reserves = _osm_layer("protected_areas_a")
    out = {
        "d_road_all": crop(_dist_km(roads.geometry, trp, shp, res)),
        "d_road_major": crop(_dist_km(roads[roads.fclass.isin(MAJOR)].geometry, trp, shp, res)),
        "d_town": crop(_dist_km(places[places.fclass.isin(["city", "town"])].geometry, trp, shp, res)),
        "d_city": crop(_dist_km(places[places.fclass == "city"].geometry, trp, shp, res)),
        "d_river": crop(_dist_km(water[water.fclass == "river"].geometry, trp, shp, res)),
        "d_reserve": crop(_dist_km(reserves.geometry, trp, shp, res)),
    }
    tr100 = from_origin(tr.c, tr.f, 100, 100)
    s100 = (shape[0] * 10, shape[1] * 10)
    r = rasterize(((g, 1) for g in roads.geometry), out_shape=s100, transform=tr100, fill=0, dtype="uint8",
                  all_touched=True)
    d100 = (distance_transform_edt(r == 0) * 0.1).astype(np.float32)
    with rasterio.open(ROAD100, "w", driver="GTiff", height=s100[0], width=s100[1], count=1, dtype="float32",
                       crs=CRS, transform=tr100, compress="deflate") as dst:
        dst.write(d100, 1)
    return out


def build_stack() -> Path:
    tr, shape, mask = grid()
    layers = {}
    for fn in (worldclim, worldcover_fractions, worldpop, osm):
        layers.update(fn(tr, shape))
        print(fn.__name__, "done", flush=True)
    names = list(layers)
    with rasterio.open(STACK, "w", driver="GTiff", height=shape[0], width=shape[1], count=len(names),
                       dtype="float32", crs=CRS, transform=tr, nodata=np.nan, compress="deflate") as dst:
        for i, n in enumerate(names, 1):
            a = np.where(mask, layers[n], np.nan).astype(np.float32)
            dst.write(a, i)
            dst.set_band_description(i, n)
    return STACK


def load_stack():
    with rasterio.open(STACK) as src:
        arr = src.read()
        names = list(src.descriptions)
        tr = src.transform
    mask = ~np.isnan(arr).any(0)
    return {n: arr[i] for i, n in enumerate(names)}, tr, mask


if __name__ == "__main__":
    print(build_stack())
