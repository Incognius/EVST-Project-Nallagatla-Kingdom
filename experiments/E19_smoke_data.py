"""Tiny synthetic stand-in for the E19 prep outputs, to smoke-test E19_glc_cnn.py / E19b_glc_brt.py.
Usage: EVST_INTERIM=... python experiments/E19_smoke_data.py  (writes INTERIM/glc24 with fake data)"""
import json

import numpy as np

from evst.paths import INTERIM

OUT = INTERIM / "glc24"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
H, W, C, S = 120, 150, 6, 12
Z = rng.normal(size=(C, H, W)).astype(np.float16)
land = np.flatnonzero(rng.random(H * W) < 0.9)
np.save(OUT / "stack.npy", Z)
np.save(OUT / "land_cells.npy", land)
cells_all = np.unique(rng.choice(land, 3000))
cell = rng.choice(cells_all, 8000)
sp = rng.integers(0, S, 8000)
np.savez(OUT / "po_cells.npz", cell=cell, sp=sp, cells_all=cells_all)
pc = rng.choice(land, 900)
np.savez(OUT / "pa.npz", cell=pc, Y=(rng.random((900, S)) < 0.1).astype(np.uint8),
         access=rng.normal(size=(900, 6)).astype(np.float32), lon=np.zeros(900), lat=np.zeros(900))
np.save(OUT / "species.npy", np.arange(S) + 1000)
json.dump({"shape": [H, W], "env_names": [f"e{i}" for i in range(C)],
           "access_names": ["roads", "popdensity", "lights", "built", "railways", "hfp"]}, open(OUT / "meta.json", "w"))
print("smoke data written to", OUT)
