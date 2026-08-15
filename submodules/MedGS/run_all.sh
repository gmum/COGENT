#!/bin/bash
#SBATCH --job-name=medgs-all
#SBATCH --account=plgunhype-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=medgs-all-%J.out

source ~/setup_medgs.sh
cd ~/Sybil-Interpretability/submodules/MedGS

for p in patient_4 patient_11 patient_12; do
    echo "=== Training $p ==="
    mkdir -p $SCRATCH/sybil_data/LUNA16/$p/mirror
    python3 -u train.py \
      -s $SCRATCH/sybil_data/LUNA16/$p \
      -m $SCRATCH/sybil_data/output/$p
done
