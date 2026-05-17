#!/bin/bash
# Wait for burnin=0 queue script (PID 2204634) to finish, then run s8-bins model.
# burnin=0 is driven by its wrapper; poll until it too is gone.
BURNIN4_PID=2203084
BURNIN0_WRAPPER_PID=2204634

echo "[$(date)] Waiting for burnin=4 (PID $BURNIN4_PID) and burnin=0 wrapper (PID $BURNIN0_WRAPPER_PID) to finish..."
while kill -0 $BURNIN4_PID 2>/dev/null || kill -0 $BURNIN0_WRAPPER_PID 2>/dev/null; do
    sleep 60
done
echo "[$(date)] Both previous jobs done. Starting s8-bins run..."
cd /home/kshinde/ondemand/pymarketsim
/home/kshinde/ondemand/pymarketsim/venv/bin/python -u train_tron.py \
    --timesteps 3000000 \
    --lam 0.012 \
    --pv-var 2e7 \
    --bg-strategy 8 \
    --burnin 0 \
    --s8-bins \
    --tag envc_nb24_s8_42bins \
    --seed 44
echo "[$(date)] s8-bins training complete."
