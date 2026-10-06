O=data/raw/inat
U=https://inaturalist-open-data.s3.amazonaws.com/observations.csv.gz
SIZE=$(curl -sI $U | grep -i content-length | tr -dc 0-9)
for i in $(seq 1 30); do
  have=$(stat -c %s $O/observations.csv.gz 2>/dev/null || echo 0)
  [ "$have" = "$SIZE" ] && break
  curl -sL --retry 5 -C - -o $O/observations.csv.gz $U
  echo "attempt $i: $(stat -c %s $O/observations.csv.gz) / $SIZE"
done
gzip -dc $O/observations.csv.gz | awk -F'\t' 'NR==1 || ($3>=7.9 && $3<=22.1 && $4>=72.5 && $4<=78.0)' > $O/wg_observations.tsv.part \
  && mv $O/wg_observations.tsv.part $O/wg_observations.tsv && wc -l $O/wg_observations.tsv && echo DONE
