from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

from evst.data import records
from evst.data.rasters import load_stack
from evst.paths import INTERIM, RAW

ENV = [f"bio{i:02d}" for i in range(1, 20)] + ["elev", "wc_tree", "wc_shrub", "wc_grass", "wc_crop", "wc_built",
                                               "wc_bare", "wc_water", "wc_wetland", "wc_mangrove"]
ACC = ["d_road_all", "d_road_major", "d_town", "d_reserve", "pop_log"]
HALF = 0.03


def _atlas():
    K = RAW / "kba"
    kba = pd.read_csv(K / "kba_data.csv")
    names = pd.read_csv(K / "kba_names.csv")
    spd = pd.read_csv(K / "kba_species_details.csv")
    names = names[names.Action.fillna("") != "Remove"]
    kba = kba.merge(names[["Common.Name", "Assigned.Name.for.Atlas"]], on="Common.Name").merge(
        spd[["Common.Name", "Scientific.Name"]], left_on="Assigned.Name.for.Atlas", right_on="Common.Name",
        suffixes=("", "_sp"))
    kba = kba[kba["Cell.ID"].notna()]
    xy = kba["Cell.ID"].str.strip("[]").str.split(",", expand=True).astype(float)
    kba["clon"], kba["clat"] = xy[0], xy[1]
    return kba, spd


def build(end: int = 2020, min_cells: int = 20, start: int | None = None) -> dict:
    tag = f"kba_setup_{start or 0}_{end}_{min_cells}.pkl"
    f = INTERIM / tag
    if f.exists():
        return pickle.loads(f.read_bytes())
    kba, spd = _atlas()
    cells = kba.drop_duplicates("Cell.ID")[["Cell.ID", "clon", "clat"]].reset_index(drop=True)
    tree = cKDTree(cells[["clon", "clat"]].values)
    truth = kba.groupby("Cell.ID")["Scientific.Name"].apply(set)
    lists = kba.groupby("Cell.ID")["List.ID"].nunique()
    layers, tr, mask = load_stack()
    H, W = mask.shape
    rr, cc = np.nonzero(mask)
    ux, uy = tr * (cc + 0.5, rr + 0.5)
    plon, plat = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True).transform(ux, uy)
    _, pi = tree.query(np.column_stack([plon, plat]))
    inside = (np.abs(plon - cells.clon.values[pi]) <= HALF) & (np.abs(plat - cells.clat.values[pi]) <= HALF)
    land = rr * W + cc
    kcell_of_land = np.where(inside, cells["Cell.ID"].values[pi], None)
    cnt = pd.Series(kcell_of_land[inside]).value_counts()
    ev = pd.Index(sorted(cnt[cnt >= 20].index.intersection(truth.index)))
    X = np.column_stack([layers[k][rr, cc] for k in ENV]).astype(np.float32)
    A = np.column_stack([np.log1p(layers[k][rr, cc]) if k.startswith("d_") else layers[k][rr, cc] for k in ACC]).astype(np.float32)
    posl = pd.Series(np.arange(len(land)), index=land)
    rec = records.load()
    b = rec[(rec.group == "birds") & (rec.year <= end)]
    if start is not None:
        b = b[b.year >= start]
    b = b[b.cell.isin(posl.index)].copy()
    b["li"] = posl[b.cell].to_numpy()
    b["visit"] = b.recordedby.astype(str) + "|" + pd.to_datetime(b.eventdate).dt.normalize().astype(str)
    b["kcell"] = kcell_of_land[b.li.to_numpy()]
    matched = set(spd["Scientific.Name"]) & set(rec[rec.group == "birds"].species)
    ncell = b.groupby("species").li.nunique()
    cand = sorted(s for s in matched if s in ncell.index and ncell[s] >= min_cells)
    sidx = pd.Series(np.arange(len(cand)), index=cand)
    Y = np.zeros((len(ev), len(cand)), np.int8)
    kpos = pd.Series(np.arange(len(ev)), index=ev)
    for k in ev:
        for s in truth[k]:
            if s in sidx.index:
                Y[kpos[k], sidx[s]] = 1
    bk = b[b.kcell.isin(ev) & b.species.isin(cand)]
    R = np.zeros_like(Y)
    R[kpos[bk.kcell].to_numpy(), sidx[bk.species].to_numpy()] = 1
    U = b[b.kcell.isin(ev)].groupby("kcell").visit.nunique().reindex(ev, fill_value=0).to_numpy(float)
    in_ev = pd.Series(kcell_of_land).isin(ev).to_numpy()
    pix_li = np.flatnonzero(in_ev)
    pix_k = kpos[pd.Series(kcell_of_land[in_ev])].to_numpy()
    cxy = cells.set_index("Cell.ID").loc[ev, ["clon", "clat"]].to_numpy()
    endemic = set(spd.loc[spd.Endemicity.fillna("") == "Western Ghats", "Scientific.Name"])
    out = dict(ev=ev, cxy=cxy, X=X, A=A, land=land, lonlat=np.column_stack([plon, plat]).astype(np.float32),
               rc=np.column_stack([rr, cc]), shape=(H, W), birds=b[["species", "li", "kcell", "visit", "year", "recordedby"]],
               cand=cand, Y=Y, R=R, U=U, n_lists=lists.reindex(ev).to_numpy(), pix_li=pix_li, pix_k=pix_k,
               endemic=np.array([s in endemic for s in cand]), spd=spd)
    f.write_bytes(pickle.dumps(out))
    return out


def training_arrays(S: dict):
    X = S["X"]
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-6)
    b = S["birds"]
    tg = np.sort(b.li.unique())
    tpos = pd.Series(np.arange(len(tg)), index=tg)
    sidx = pd.Series(np.arange(len(S["cand"])), index=S["cand"])
    bc = b[b.species.isin(S["cand"])]
    Ytg = np.zeros((len(tg), len(S["cand"])), np.float32)
    Ytg[tpos[bc.li].to_numpy(), sidx[bc.species].to_numpy()] = 1
    Utg = b.groupby("li").visit.nunique().reindex(tg).to_numpy(np.float32)
    return Xz, tg, Ytg, Utg


def cell_mean(pred_land: np.ndarray, S: dict) -> np.ndarray:
    p = pred_land[S["pix_li"]]
    n = len(S["ev"])
    out = np.zeros((n,) + p.shape[1:], np.float64)
    np.add.at(out, S["pix_k"], p)
    return out / np.bincount(S["pix_k"], minlength=n).reshape((n,) + (1,) * (p.ndim - 1))
