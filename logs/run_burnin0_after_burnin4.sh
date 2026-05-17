#!/bin/bash
# Wait for burnin=4 job (PID 2203084) to finish, then run burnin=0
BURNIN4_PID=2203084
echo "[$(date)] Waiting for burnin=4 job (PID $BURNIN4_PID) to finish..."
while kill -0 $BURNIN4_PID 2>/dev/null; do
    sleep 60
done
echo "[$(date)] burnin=4 done. Starting burnin=0..."
cd /home/kshinde/ondemand/pymarketsim
/home/kshinde/ondemand/pymarketsim/venv/bin/python -u train_tron.py \
    --timesteps 3000000 \
    --lam 0.012 \
    --pv-var 2e7 \
    --bg-strategy 8 \
    --burnin 0 \
    --skew-bins \
    --tag envc_nb24_s8_burnin0 \
    --seed 43
echo "[$(date)] burnin=0 training complete."
