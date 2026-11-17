#!/bin/bash
#SBATCH -J e19
#SBATCH -p u22
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=4-00:00:00
#SBATCH -o /home2/mohit.sriram/evst_data/logs/e19_%j.out
# E19 on Ada: (1) smoke test on synthetic data in a scratch copy, (2) GLC24 prep, (3) GPU runs in priority order,
# (4) BRT reference on the CPUs. Usage: sbatch scripts/ada_e19.sh [steps]
set -eo pipefail
STEPS=${1:-30000}
PY=~/venvs/evst/bin/python
cd ~/evst
nvidia-smi -L

# 1. smoke test in an isolated copy so fake data never touches the real interim dir
SM=/scratch/$USER/evst_smoke
rm -rf $SM && mkdir -p $SM && cp -r src experiments pyproject.toml $SM/
(cd $SM && PYTHONPATH=src $PY experiments/E19_smoke_data.py \
  && for m in mlp cnn; do for b in random tgb; do PYTHONPATH=src $PY experiments/E19_glc_cnn.py --model $m --bg $b --steps 30 --batch 64; done; done \
  && PYTHONPATH=src $PY experiments/E19_glc_cnn.py --model cnn --bg tgb --split block --steps 30 --batch 64 \
  && PYTHONPATH=src $PY experiments/E19b_glc_brt.py --n-species 4 --jobs 2 --n-bg 500 \
  && PYTHONPATH=src $PY experiments/E19_analyse.py)
echo "SMOKE OK"

# 2. prep (idempotent)
[ -f data/interim/glc24/meta.json ] || $PY experiments/E19_glc_prep.py
echo "PREP OK"

# 3. BRT reference on CPU in the background while the GPU trains
$PY experiments/E19b_glc_brt.py --n-species 300 --jobs 5 > ~/evst_data/logs/e19b_$SLURM_JOB_ID.out 2>&1 &
BRT=$!

# 4. GPU runs: seed 0 first (enough for the main comparison), then spatial-block split, then extra seeds
run(){ f=results/E19/E19_$1_$2_$3_s$4.csv; [ -f $f ] && { echo "have $f"; return; }; $PY experiments/E19_glc_cnn.py --model $1 --bg $2 --split $3 --seed $4 --steps $STEPS; }
for m in mlp cnn; do for b in random tgb; do run $m $b random 0; done; done
for m in mlp cnn; do for b in random tgb; do run $m $b block 0; done; done
for s in 1 2; do for m in mlp cnn; do for b in random tgb; do run $m $b random $s; done; done; done
wait $BRT || echo "BRT failed"
$PY experiments/E19_analyse.py
echo E19 ALLDONE
