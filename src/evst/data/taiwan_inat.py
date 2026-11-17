import sys
import time

import duckdb

from evst.paths import RAW

BBOX = (119.9, 21.8, 122.1, 25.4)
OUT = RAW / "taiwan" / "tw_observations.tsv"
SHARE = RAW / "taiwan" / "taxon_tw_share.csv"
COLS = ("{'observation_uuid':'VARCHAR','observer_id':'BIGINT','latitude':'DOUBLE','longitude':'DOUBLE',"
        "'positional_accuracy':'DOUBLE','taxon_id':'BIGINT','quality_grade':'VARCHAR','observed_on':'VARCHAR',"
        "'anomaly_score':'DOUBLE'}")

if __name__ == "__main__":
    t0 = time.time()
    src = (RAW / "inat" / "observations.csv.gz").as_posix()
    rd = f"read_csv('{src}', delim='\t', header=true, quote='', escape='', columns={COLS}, ignore_errors=true)"
    con = duckdb.connect()
    con.execute("SET threads=8; SET memory_limit='3GB'; SET preserve_insertion_order=false;")
    if "--share" in sys.argv:
        con.execute(f"""
        COPY (SELECT taxon_id, count(*) AS n_all,
                     sum(CASE WHEN longitude BETWEEN 120.0 AND {BBOX[2]} AND latitude BETWEEN {BBOX[1]} AND {BBOX[3]}
                         THEN 1 ELSE 0 END) AS n_tw
              FROM {rd} WHERE quality_grade = 'research' GROUP BY taxon_id HAVING n_tw > 0)
        TO '{SHARE.as_posix()}' (HEADER)""")
    else:
        con.execute(f"""
        COPY (SELECT * FROM {rd}
              WHERE longitude BETWEEN {BBOX[0]} AND {BBOX[2]} AND latitude BETWEEN {BBOX[1]} AND {BBOX[3]})
        TO '{OUT.as_posix()}' (HEADER, DELIMITER '\t')""")
    print(f"done {time.time() - t0:.0f}s", flush=True)
