#!/bin/bash
# Waits for train_mlp_ne.py to finish, then runs Phase 3 automatically

TRAINING_PID=3533719
MODEL_DIR="/home/kshinde/ondemand/pymarketsim/runs/sac_zi_ne_s0_ne_s0"
LOG="/home/kshinde/ondemand/pymarketsim/eval_mlp_ne_s0.log"

echo "=== Phase 3 watcher started $(date) ===" | tee "$LOG"
echo "Waiting for training PID $TRAINING_PID to finish..." | tee -a "$LOG"

# Wait for training to complete
while kill -0 $TRAINING_PID 2>/dev/null; do
    sleep 60
done

echo "=== Training finished at $(date) ===" | tee -a "$LOG"
echo "Starting Phase 3 evaluation..." | tee -a "$LOG"

cd /home/kshinde/ondemand/pymarketsim

# Check which model to use: best_model if exists, else final_model
if [ -f "$MODEL_DIR/best_model.zip" ]; then
    MODEL="$MODEL_DIR/best_model"
    echo "Using best_model" | tee -a "$LOG"
else
    MODEL="$MODEL_DIR/final_model"
    echo "best_model not found, using final_model" | tee -a "$LOG"
fi

venv/bin/python -u eval_mlp_ne.py \
    --ne-strategy 0 \
    --model "$MODEL" \
    --num-runs 2000 \
    --output eval_mlp_ne_s0_results.csv \
    2>&1 | tee -a "$LOG"

echo "=== Phase 3 complete at $(date) ===" | tee -a "$LOG"
