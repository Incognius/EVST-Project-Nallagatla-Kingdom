from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "reports" / "figures"

for _p in (RAW, INTERIM, PROCESSED, RESULTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)
