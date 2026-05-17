#!/bin/bash
# Wait for the s8-bins queue script (PID 2462290) to finish, then run burnin=4 + 42-bin model.
S8BINS_WRAPPER_PID=2462290

echo "[$(date)] Waiting for s8-bins wrapper (PID $S8BINS_WRAPPER_PID) to finish..."
while kill -0 $S8BINS_WRAPPER_PID 2>/dev/null; do
    sleep 60
done
echo "[$(date)] Previous job done. Starting s8-bins burnin=4 run..."
cd /home/kshinde/ondemand/pymarketsim
/home/kshinde/ondemand/pymarketsim/venv/bin/python -u train_tron.py \
    --timesteps 3000000 \
    --lam 0.012 \
    --pv-var 2e7 \
    --bg-strategy 8 \
    --burnin 4 \
    --s8-bins \
    --tag envc_nb24_s8_42bins_burnin4 \
    --seed 45
echo "[$(date)] s8-bins burnin=4 training complete."
