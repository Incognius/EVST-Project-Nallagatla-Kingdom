from __future__ import annotations

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin

from evst.paths import PROCESSED, RAW

ECOREGIONS = [242, 253, 254, 270, 271]
CRS = "EPSG:32643"
RES = 1000.0
REGION_GPKG = PROCESSED / "western_ghats.gpkg"


def build_region(buffer_m: float = 10_000) -> gpd.GeoDataFrame:
    eco = gpd.read_file(f"zip://{RAW / 'ecoregions' / 'Ecoregions2017.zip'}")
    wg = eco[eco.ECO_ID.isin(ECOREGIONS)].to_crs(CRS)
    land = eco.to_crs(CRS).clip(wg.total_bounds + np.array([-50e3, -50e3, 50e3, 50e3])).union_all()
    region = wg.union_all().buffer(buffer_m).intersection(land)
    out = gpd.GeoDataFrame({"name": ["Western Ghats"]}, geometry=[region], crs=CRS)
    eco_parts = wg[["ECO_ID", "ECO_NAME", "geometry"]]
    out.to_file(REGION_GPKG, layer="region", driver="GPKG")
    eco_parts.to_file(REGION_GPKG, layer="ecoregions", driver="GPKG")
    return out


def region() -> gpd.GeoDataFrame:
    if not REGION_GPKG.exists():
        build_region()
    return gpd.read_file(REGION_GPKG, layer="region")


def grid():
    g = region()
    xmin, ymin, xmax, ymax = g.total_bounds
    xmin, ymin = np.floor(xmin / RES) * RES, np.floor(ymin / RES) * RES
    xmax, ymax = np.ceil(xmax / RES) * RES, np.ceil(ymax / RES) * RES
    w, h = int((xmax - xmin) / RES), int((ymax - ymin) / RES)
    tr = from_origin(xmin, ymax, RES, RES)
    mask = rasterize([(g.geometry.iloc[0], 1)], out_shape=(h, w), transform=tr, fill=0, dtype="uint8").astype(bool)
    return tr, (h, w), mask


if __name__ == "__main__":
    r = build_region()
    tr, shape, mask = grid()
    print(f"area {r.area.iloc[0]/1e6:,.0f} km2, grid {shape}, {mask.sum():,} land cells")
    print("lon/lat bounds", r.to_crs(4326).total_bounds.round(3))


def region_wkt(tol: float = 0.05) -> str:
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient
    g = region().to_crs(4326).geometry.iloc[0].simplify(tol).buffer(tol)
    g = g if g.geom_type == "Polygon" else max(g.geoms, key=lambda x: x.area)
    return orient(Polygon(g.exterior).simplify(tol / 2), sign=1.0).wkt
