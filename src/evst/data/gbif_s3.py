import sys
import time

import duckdb

from evst.paths import RAW

SNAP = "2026-10-01"
LO, HI = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (0.0, 1.0)
PARTS = RAW / "gbif" / f"wg_parts_{SNAP}"
OUT = RAW / "gbif" / f"wg_bbox_{SNAP}.parquet"
COLS = """gbifid, datasetkey, publishingorgkey, kingdom, phylum, class, "order", family, genus, species,
          taxonrank, specieskey, taxonkey, countrycode, decimallatitude, decimallongitude,
          coordinateuncertaintyinmeters, eventdate, year, month, basisofrecord, institutioncode,
          recordedby, issue, occurrencestatus, individualcount, license"""
BBOX = (72.5, 7.9, 78.0, 22.1)
BUCKET = "s3://gbif-open-data-eu-central-1"

if __name__ == "__main__":
    PARTS.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; SET s3_region='eu-central-1'; SET threads=8; "
                "SET memory_limit='1500MB'; SET preserve_insertion_order=false; SET http_retries=10; SET http_timeout=120000;")
    files = [r[0] for r in con.execute(f"SELECT file FROM glob('{BUCKET}/occurrence/{SNAP}/occurrence.parquet/*')").fetchall()]
    print(f"{len(files)} snapshot files", flush=True)
    step, t0 = 100, time.time()
    for k in range(int(LO * len(files)) // step * step, int(HI * len(files)), step):
        part = PARTS / f"part_{k:05d}.parquet"
        if part.exists():
            continue
        flist = ", ".join(f"'{f}'" for f in files[k:k + step])
        con.execute(f"""
        COPY (SELECT {COLS} FROM read_parquet([{flist}])
              WHERE countrycode = 'IN'
                AND decimallatitude BETWEEN {BBOX[1]} AND {BBOX[3]}
                AND decimallongitude BETWEEN {BBOX[0]} AND {BBOX[2]})
        TO '{part.as_posix()}.{LO}.tmp' (FORMAT parquet, COMPRESSION zstd)""")
        (PARTS / f"part_{k:05d}.parquet.{LO}.tmp").rename(part)
        print(f"{k + step}/{len(files)} files, {time.time() - t0:.0f}s", flush=True)
    if len(list(PARTS.glob("*.parquet"))) < (len(files) + step - 1) // step:
        sys.exit(0)
    con.execute(f"COPY (SELECT * FROM read_parquet('{PARTS.as_posix()}/*.parquet')) TO '{OUT.as_posix()}' "
                "(FORMAT parquet, COMPRESSION zstd)")
    n = con.execute(f"SELECT count(*) FROM '{OUT.as_posix()}'").fetchone()[0]
    print(f"{n:,} records -> {OUT} in {time.time() - t0:.0f}s")
