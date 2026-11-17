from __future__ import annotations

from evst.data import kba

ELEV = kba.ENV.index("elev")


def get(name: str, **kw) -> dict:
    if name == "kba":
        return kba.build(**kw)
    if name == "taiwan":
        from evst.data import taiwan
        return taiwan.build(**kw)
    raise KeyError(name)


def label(name: str) -> str:
    return {"kba": "kba", "taiwan": "twbbs"}[name]


def suffix(name: str) -> str:
    return "" if name == "kba" else f"_{name}"


training_arrays = kba.training_arrays
cell_mean = kba.cell_mean
