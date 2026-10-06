R=data/raw
dl(){ [ -s "$2" ] && return; curl -sL --retry 5 -C - -o "$2.part" "$1" && mv "$2.part" "$2" && echo "ok $2"; }
dl https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2020/IND/ind_ppp_2020_1km_Aggregated.tif $R/worldpop/ind_ppp_2020_1km.tif &
dl https://geodata.ucdavis.edu/climate/worldclim/2_1/tiles/tile/tile_33_wc2.1_30s_bio.tif $R/worldclim/tile_33_wc2.1_30s_bio.tif &
dl https://geodata.ucdavis.edu/climate/worldclim/2_1/tiles/tile/tile_33_wc2.1_30s_elev.tif $R/worldclim/tile_33_wc2.1_30s_elev.tif &
dl https://download.geofabrik.de/asia/india/western-zone-latest-free.shp.zip $R/osm/western-zone-latest-free.shp.zip &
dl https://download.geofabrik.de/asia/india/southern-zone-latest-free.shp.zip $R/osm/southern-zone-latest-free.shp.zip &
for lat in N06 N09 N12 N15 N18 N21; do
  ( for lon in E072 E075; do
      dl https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_${lat}${lon}_Map.tif $R/worldcover/ESA_WorldCover_10m_2021_v200_${lat}${lon}_Map.tif
    done ) &
done
wait
echo ALLDONE
