import os

from pygbif import occurrences as occ
from evst.data.region import region_wkt

wkt = region_wkt()

query = {
    "type": "and",
    "predicates": [
        {"type": "within", "geometry": wkt},
        {"type": "equals", "key": "HAS_COORDINATE", "value": "true"},
        {"type": "equals", "key": "HAS_GEOSPATIAL_ISSUE", "value": "false"},
        {"type": "equals", "key": "OCCURRENCE_STATUS", "value": "PRESENT"},
    ],
}
key, meta = occ.download(query, format="SIMPLE_PARQUET",
                         user=os.environ["GBIF_USER"], pwd=os.environ["GBIF_PWD"], email=os.environ["GBIF_EMAIL"])
print("download key:", key)
