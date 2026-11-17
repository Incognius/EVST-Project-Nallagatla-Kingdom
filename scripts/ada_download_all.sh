#!/bin/bash
#SBATCH -J evst-dl
#SBATCH -p u22
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH -o /home2/mohit.sriram/evst_data/logs/dl_%j.out
# Rebuild every public raw input on the cluster (Kerala Bird Atlas must be added by hand: Dryad needs a login).
set -u
R=/home2/mohit.sriram/evst_data/raw
mkdir -p $R/{ecoregions,worldclim,worldcover,worldpop,osm,grip4,inat}
dl(){ [ -s "$2" ] && { echo "have $2"; return; }; curl -sSL --retry 8 --retry-all-errors -C - -o "$2.part" "$1" && mv "$2.part" "$2" && echo "ok $2 $(stat -c %s "$2")"; }

dl https://storage.googleapis.com/teow2016/Ecoregions2017.zip $R/ecoregions/Ecoregions2017.zip &
dl https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2020/IND/ind_ppp_2020_1km_Aggregated.tif $R/worldpop/ind_ppp_2020_1km.tif &
dl https://geodata.ucdavis.edu/climate/worldclim/2_1/tiles/tile/tile_33_wc2.1_30s_bio.tif $R/worldclim/tile_33_wc2.1_30s_bio.tif &
dl https://geodata.ucdavis.edu/climate/worldclim/2_1/tiles/tile/tile_33_wc2.1_30s_elev.tif $R/worldclim/tile_33_wc2.1_30s_elev.tif &
dl https://download.geofabrik.de/asia/india/western-zone-latest-free.shp.zip $R/osm/western-zone-latest-free.shp.zip &
dl https://download.geofabrik.de/asia/india/southern-zone-latest-free.shp.zip $R/osm/southern-zone-latest-free.shp.zip &
dl https://dataportaal.pbl.nl/data/GRIP4/GRIP4_Region3_vector_shp.zip $R/grip4/GRIP4_Region3_vector_shp.zip &
for lat in N06 N09 N12 N15 N18 N21; do
  ( for lon in E072 E075; do
      dl https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_${lat}${lon}_Map.tif $R/worldcover/ESA_WorldCover_10m_2021_v200_${lat}${lon}_Map.tif
    done ) &
done
dl https://inaturalist-open-data.s3.amazonaws.com/taxa.csv.gz $R/inat/taxa.csv.gz &

# iNaturalist: stream the 13.7 GB export and keep only the Western Ghats bounding box (same filter as inat_download_wg.sh)
if [ ! -s $R/inat/wg_observations.tsv ]; then
  for i in 1 2 3; do
    curl -sSL --retry 8 --retry-all-errors https://inaturalist-open-data.s3.amazonaws.com/observations.csv.gz \
      | gzip -dc | awk -F'\t' 'NR==1 || ($3>=7.9 && $3<=22.1 && $4>=72.5 && $4<=78.0)' > $R/inat/wg_observations.tsv.part \
      && [ "${PIPESTATUS[0]}" = 0 ] && mv $R/inat/wg_observations.tsv.part $R/inat/wg_observations.tsv && break
    echo "inat attempt $i failed"
  done
  wc -l $R/inat/wg_observations.tsv
fi
wait
ls -laR $R | grep -v '^total'
echo ALLDONE
