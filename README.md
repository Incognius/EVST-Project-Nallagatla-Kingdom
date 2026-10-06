# biodiv-gaps

EVST Topic 11: can AI identify biodiversity patterns missing from conventional sampling?
Bias-aware species distribution modelling for the Western Ghats.

Team: Ponnambalam V, Nandini C, Mohit Nallagatla, Manali Gupta

```
src/evst/       package: data, bias, models, validate, virtual
experiments/    E01 bias anatomy · E02 disdat benchmark · E03 virtual species · E04 Western Ghats gaps
r_checks/       cross-checks against the original R packages
scripts/        data download helpers
results/        result tables
reports/        mid-evaluation report and figures
```

```bash
python -m venv --system-site-packages .venv
.venv/Scripts/pip install -e . -r requirements.txt
python -m evst.data.disdat
python experiments/E02_disdat_benchmark.py
python experiments/E02_analyse.py
```

Raw data lives in `data/` and is not tracked.
