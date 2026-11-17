"""Parallel, resumable download of the GeoLifeCLEF 2024 subset we use, from the organisers' public Seafile share.

Single connections to the share are throttled (~20 kB/s), so every file is split into byte ranges fetched by a
thread pool. Usage: python scripts/glc24_seafile_download.py OUT_DIR [--workers 24]
"""
import argparse
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SHARE = "https://seafile.plantnet.org/d/bdb829337aa44a9489f6/files/?p={}&dl=1"
BIO = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15]
HF = ["Roads", "Popdensity2010", "Lights2009", "Built2009", "Railways", "croplands2005", "Pasture2009", "NavWater2009"]
FILES = (["/PresenceOnlyOccurrences/GLC24-PO-metadata-train.csv", "/PresenceOnlyOccurrences/ReadMe.txt",
          "/PresenceAbsenceSurveys/GLC24-PA-metadata-train.csv", "/PresenceAbsenceSurveys/ReadMe.txt",
          "/EnvironmentalRasters/LandCover/LandCover_MODIS_Terra-Aqua_500m.tif", "/EnvironmentalRasters/LandCover/LC_README.txt",
          "/EnvironmentalRasters/HumanFootprint/summarized/HFP2009_WGS84.tif", "/EnvironmentalRasters/HumanFootprint/HF_README.txt",
          "/EnvironmentalRasters/Elevation/GLC24-PO-train-elevation.csv", "/EnvironmentalRasters/Elevation/GLC24-PA-train-elevation.csv",
          "/EnvironmentalRasters/Climate/BioClimatic_Average_1981-2010/Readme.txt", "/EnvironmentalRasters/Soilgrids/Readme.txt"]
         + [f"/EnvironmentalRasters/Climate/BioClimatic_Average_1981-2010/bio{b}.tif" for b in BIO]
         + [f"/EnvironmentalRasters/HumanFootprint/detailed/{h}_WGS84.tif" for h in HF]
         + [f"/EnvironmentalRasters/Soilgrids/crop_proj_{s}_5-15cm_mean_1000.tif" for s in ("phh2o", "soc", "sand")])
CHUNK = 4 << 20
lock = threading.Lock()
done_bytes = 0


def resolve(path):
    req = urllib.request.Request(SHARE.format(urllib.parse.quote(path)), method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.url, int(r.headers["Content-Length"])


def fetch(path, dst, start, end, url_box):
    global done_bytes
    for attempt in range(20):
        try:
            req = urllib.request.Request(url_box[0], headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) != end - start + 1:
                raise IOError(f"short read {len(data)}")
            fd = os.open(dst, os.O_WRONLY)
            os.pwrite(fd, data, start)
            os.close(fd)
            with lock:
                done_bytes += len(data)
            return
        except Exception as e:  # token expiry or throttling: re-resolve and back off
            time.sleep(min(60, 2 ** attempt))
            try:
                url_box[0] = resolve(path)[0]
            except Exception:
                pass
    raise RuntimeError(f"failed {path} {start}-{end}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    jobs, total = [], 0
    for p in FILES:
        dst = os.path.join(a.out, p.lstrip("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            continue
        url, size = resolve(p)
        part = dst + ".part"
        donef = part + ".chunks"
        have = set(open(donef).read().split()) if os.path.exists(donef) else set()
        if not os.path.exists(part):
            with open(part, "wb") as f:
                f.truncate(size)
        box = [url]
        chunks = [(s, min(s + CHUNK, size) - 1) for s in range(0, size, CHUNK)]
        jobs.append((p, dst, part, donef, box, [c for c in chunks if str(c[0]) not in have], len(chunks)))
        total += size
        print(f"{p}: {size / 1e6:.1f} MB, {len(chunks) - len(have)} chunks to fetch", flush=True)
    t0 = time.time()

    def run(p, part, donef, box, c):
        fetch(p, part, c[0], c[1], box)
        with lock, open(donef, "a") as f:
            f.write(f"{c[0]}\n")

    with ThreadPoolExecutor(a.workers) as ex:
        futs = [(j, ex.submit(run, j[0], j[2], j[3], j[4], c)) for j in jobs for c in j[5]]
        last = 0
        while any(not f.done() for _, f in futs):
            time.sleep(30)
            el = time.time() - t0
            print(f"{done_bytes / 1e6:8.1f} MB in {el / 60:5.1f} min ({done_bytes / max(el, 1) / 1e3:6.0f} kB/s)", flush=True)
        for _, f in futs:
            f.result()
    for p, dst, part, donef, *_ in jobs:
        os.replace(part, dst)
        os.remove(donef)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
