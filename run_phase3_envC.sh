#!/usr/bin/env bash
# 3-phase TRONformer experiment for Env C, 3 seeds
# Phase 1: NE search (24 bg agents, 1 deviator)
# Phase 2: Train skewed + uniform TRONformer vs identified NE strategy
# Phase 3: Evaluate both models in same environment
#
# Env C params: lam=0.012, shock-var=2e4, pv-var=2e6, n-bg=24 (matched arrival rates)

PYTHON="venv/bin/python"
LOG="runs_phase3_envC.log"

ENV_ARGS="--lam 0.012 --shock-var 2e4 --pv-var 2e6"
NE_ENV_ARGS="--lam 0.012 --shock-var 2e4 --pv-var 2e6 --n-bg 24"
TRAIN_COMMON="--n-bg 24 --lam 0.012 --lam-zi 0.012 --shock-var 2e4 --pv-var 2e6 --n-layers 2 --batch-size 512"
EVAL_COMMON="--num-runs 500 --n-bg 24 --lam-market 0.012 --lam-zi 0.012 --shock-var 2e4 --pv-var 2e6 --n-layers 2 --bg-only"

for seed in 123 456; do
  echo "" | tee -a $LOG
  echo "######################################################################" | tee -a $LOG
  echo "### SEED $seed — Phase 1: NE Search                                ###" | tee -a $LOG
  echo "######################################################################" | tee -a $LOG
  echo "=== Phase 1 start: $(date) ===" | tee -a $LOG

  PREFIX="results/ne_search/ne_envC_seed${seed}_nb24"
  $PYTHON -u ne_experiment.py \
    --n-bg 24 --base-seed $seed $NE_ENV_ARGS \
    --prefix $PREFIX \
    2>&1 | tee ${PREFIX}_experiment.log
  echo "=== Phase 1 done: $(date) ===" | tee -a $LOG

  # Parse lowest-index NE candidate from summary CSV
  NE_STRAT=$(python3 -c "
import csv, sys
with open('${PREFIX}_ne_summary.csv') as f:
    for row in csv.DictReader(f):
        if row['is_ne_candidate'].strip().lower() == 'true':
            print(row['bg_strategy'].strip().lstrip('S'))
            sys.exit(0)
print('8')  # fallback
")
  echo "=== Identified NE strategy: S${NE_STRAT} ===" | tee -a $LOG

  echo "" | tee -a $LOG
  echo "### SEED $seed — Phase 2: Train TRONformer (skewed + uniform)      ###" | tee -a $LOG

  for spec in \
    "ne24_envC_rope_seed${seed}_skewed:--skew-bins" \
    "ne24_envC_rope_seed${seed}_uniform:--shade-max 1000 --n-bins 100"
  do
    tag=$(echo $spec | cut -d: -f1)
    bins=$(echo $spec | cut -d: -f2)
    echo "=== Phase 2 ($tag) start: $(date) ===" | tee -a $LOG
    $PYTHON -u train_tronformer.py \
      --tag $tag \
      --bg-strategy $NE_STRAT \
      $TRAIN_COMMON --seed $seed $bins \
      2>&1 | tee runs_${tag}_train.log
    echo "=== Phase 2 ($tag) done: $(date) ===" | tee -a $LOG
  done

  echo "" | tee -a $LOG
  echo "### SEED $seed — Phase 3: Evaluate both models                     ###" | tee -a $LOG

  BASELINE="${PREFIX}_abs_mean.csv"
  EVAL_DIR="results/equilibrium/envC_rope_seed${seed}"
  mkdir -p $EVAL_DIR

  for spec in \
    "ne24_envC_rope_seed${seed}_skewed:--skew-bins" \
    "ne24_envC_rope_seed${seed}_uniform:--shade-max 1000 --n-bins 100"
  do
    tag=$(echo $spec | cut -d: -f1)
    bins=$(echo $spec | cut -d: -f2)
    echo "=== Phase 3 ($tag) start: $(date) ===" | tee -a $LOG
    $PYTHON -u equilibrium_tronformer.py \
      --model runs/tronformer_${tag}/best_model.pt \
      $EVAL_COMMON $NE_STRAT \
      --baseline $BASELINE \
      $bins \
      --output ${EVAL_DIR}/${tag}_results.csv \
      2>&1 | tee ${EVAL_DIR}/${tag}_eval.log
    echo "=== Phase 3 ($tag) done: $(date) ===" | tee -a $LOG
  done

  echo "=== Seed $seed complete: $(date) ===" | tee -a $LOG
done

echo "" | tee -a $LOG
echo "=== ALL 3 SEEDS COMPLETE ===" | tee -a $LOG
