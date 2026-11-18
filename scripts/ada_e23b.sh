#!/bin/bash
#SBATCH -J e23b
#SBATCH -p u22
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=12:00:00
#SBATCH -o /home2/mohit.sriram/evst_data/logs/e23b_%j.out
# E23 analysis only (the predictions already exist from scripts/ada_e23.sh). Usage: sbatch scripts/ada_e23b.sh [reps]
set -eo pipefail
cd ~/evst
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHONPATH=src ~/venvs/evst/bin/python experiments/E23_survey_budget.py --system glc --reps ${1:-200}
echo E23 ALLDONE
