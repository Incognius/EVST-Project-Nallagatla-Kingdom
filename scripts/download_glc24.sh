set -e
O=data/raw/glc24
for f in GLC24_PA_metadata_train.csv GLC24_P0_metadata_train.csv GLC24_PA_metadata_test.csv \
  "EnvironmentalRasters/EnvironmentalRasters/Climate/Average 1981-2010/GLC24-PA-train-bioclimatic.csv" \
  "EnvironmentalRasters/EnvironmentalRasters/Elevation/GLC24-PA-train-elevation.csv" \
  "EnvironmentalRasters/EnvironmentalRasters/Human Footprint/GLC24-PA-train-human_footprint.csv" \
  "EnvironmentalRasters/EnvironmentalRasters/LandCover/GLC24-PA-train-landcover.csv" \
  "EnvironmentalRasters/EnvironmentalRasters/SoilGrids/GLC24-PA-train-soilgrids.csv"; do
  .venv/Scripts/kaggle competitions download -c geolifeclef-2024 -f "$f" -p $O -q || echo "FAILED $f"
done
ls -la $O
