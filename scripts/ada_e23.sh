#!/bin/bash
#SBATCH -J e23
#SBATCH -p u22
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH -o /home2/mohit.sriram/evst_data/logs/e23_%j.out
# E23 on Ada: retrain the seed-0 E19 models with --save-pa (survey-plot predictions), then the survey-budget analysis.
# Usage: sbatch scripts/ada_e23.sh [steps]
set -eo pipefail
STEPS=${1:-30000}
PY=~/venvs/evst/bin/python
cd ~/evst
nvidia-smi -L
run(){ f=data/interim/glc24/preds/E19_$1_$2_random_s0_pred.npy; [ -f $f ] && { echo "have $f"; return; }
  PYTHONPATH=src $PY experiments/E19_glc_cnn.py --model $1 --bg $2 --seed 0 --steps $STEPS --tag _pred --save-pa; }
for m in mlp cnn; do for b in tgb random; do run $m $b; done; done
PYTHONPATH=src $PY experiments/E23_survey_budget.py --system glc
echo E23 ALLDONE
