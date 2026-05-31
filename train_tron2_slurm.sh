#!/bin/bash
#SBATCH --job-name=tron2_envb
#SBATCH --account=wellman98
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/tron2_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kshinde@umich.edu

# ENV_TAG and NE_STRATEGY can be overridden via --export:
#   sbatch --export=ALL,ENV_TAG=C,NE_STRATEGY=8,TAG=envc_s8_v1 train_tron2_slurm.sh
ENV_TAG=${ENV_TAG:-B}
NE_STRATEGY=${NE_STRATEGY:-8}
TAG=${TAG:-envb_s8_v1}
EPISODES=${EPISODES:-3000000}

cd /home/kshinde/ondemand/pymarketsim
source venv/bin/activate

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $CUDA_VISIBLE_DEVICES"
echo "Env:    $ENV_TAG | Strategy: S$NE_STRATEGY | Tag: $TAG | Episodes: $EPISODES"

python train_tron2.py \
    --env-tag "$ENV_TAG" \
    --ne-strategy "$NE_STRATEGY" \
    --tag "$TAG" \
    --episodes "$EPISODES"
