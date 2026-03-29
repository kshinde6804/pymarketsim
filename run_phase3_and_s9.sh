#!/bin/bash
# run_phase3_and_s9.sh — Phase 3 eval for S6 + S9 generalization training
#
# Run this after the 1M-step S6 training completes.
# Assumes venv is at /home/kshinde/ondemand/pymarketsim/venv/

set -e
cd /home/kshinde/ondemand/pymarketsim

PYTHON=venv/bin/python
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "======================================================================"
echo "Phase 3: Evaluate S6 MLP model (2000 runs)"
echo "======================================================================"
$PYTHON eval_mlp_ne_large.py \
    --ne-strategy 6 \
    --model runs/sac_zi_ne_large_s6_v2/best_model \
    --num-runs 2000 \
    --output eval_mlp_ne_large_s6_v2_results.csv \
    2>&1 | tee eval_s6_v2.log

echo ""
echo "======================================================================"
echo "Saving experiment summary"
echo "======================================================================"
$PYTHON - <<'PYEOF'
import pandas as pd, numpy as np, os, datetime

df = pd.read_csv("eval_mlp_ne_large_s6_v2_results.csv")
ne_row  = df[df.deviator == "ZI-S6"].iloc[0]
mlp_row = df[df.deviator == "MLP-SAC"].iloc[0]
best_zi = df[df.deviator.str.startswith("ZI-")].sort_values("mean_profit").iloc[-1]

summary = f"""# MLP NE Experiment — S6 Background — 1M Steps (Generalization Run 1/2)

Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Experiment Configuration
- NE background strategy: S6  shade=[380,420], eta=0.5
- Background agents: 24 (n_bg=24)
- Training steps: 1,000,000 (600K resumed + 400K)
- Training seed: 42
- SAC hyperparams: lr=1e-4, buffer=500K, batch=256, gamma=0.99, ent_coef=auto
- Evaluation runs (Phase 3): 2,000 per deviator
- ENV: lam=0.005, mean=1e5, r=0.01, shock_var=1e6, pv_var=5e6, q_max=10, sim_time=2000
- Model path: runs/sac_zi_ne_large_s6_v2/best_model.zip
- Results CSV: eval_mlp_ne_large_s6_v2_results.csv

## Phase 1 NE Result (ZI×ZI, 24 BG, 2000 runs/cell)
- NE = S6  diagonal profit = 168.4 ± 285.6 (SE)
- S9 also strict NE: diagonal = 35.0 (weak)

## Phase 3 Results
| Deviator | Mean Profit | ±SE |
|----------|-------------|-----|
"""
for _, row in df.sort_values("mean_profit", ascending=False).iterrows():
    summary += f"| {row.deviator} | {row.mean_profit:.1f} | {row.se:.1f} |\n"

mlp_vs_ne   = mlp_row.mean_profit - ne_row.mean_profit
mlp_vs_best = mlp_row.mean_profit - best_zi.mean_profit
sig = mlp_vs_best > 2 * mlp_row.se

summary += f"""
## Key Metrics
- NE diagonal (ZI-S6 vs ZI-S6): {ne_row.mean_profit:.1f} ± {ne_row.se:.1f}
- Best ZI deviator ({best_zi.deviator}):         {best_zi.mean_profit:.1f} ± {best_zi.se:.1f}
- MLP-SAC deviator:                {mlp_row.mean_profit:.1f} ± {mlp_row.se:.1f}
- MLP Δ vs NE diagonal:            {mlp_vs_ne:+.1f}
- MLP Δ vs best ZI:                {mlp_vs_best:+.1f}
- Statistically significant (>2×SE): {'YES' if sig else 'NO'}

## Next Run
- NE background strategy: S9  shade=[950,1050], eta=0.5
- Training steps: 1,000,000
- Training seed: 123  (different from S6 run seed=42)
- Run dir: runs/sac_zi_ne_large_s9_v1/
- Results CSV: eval_mlp_ne_large_s9_v1_results.csv
"""

out = "results/mlp_ne_s6_v2_summary.md"
os.makedirs("results", exist_ok=True)
with open(out, "w") as f:
    f.write(summary)
print(f"Summary saved → {out}")
print(summary)
PYEOF

echo ""
echo "======================================================================"
echo "Phase 2 (S9, seed=123): Train MLP 1M steps against NE=S9 background"
echo "======================================================================"
$PYTHON train_mlp_ne_large.py \
    --ne-strategy 9 \
    --timesteps 1000000 \
    --seed 123 \
    --tag v1 \
    2>&1 | tee train_s9_v1.log

echo ""
echo "======================================================================"
echo "Phase 3 (S9): Evaluate S9 MLP model (2000 runs)"
echo "======================================================================"
$PYTHON eval_mlp_ne_large.py \
    --ne-strategy 9 \
    --model runs/sac_zi_ne_large_s9_v1/best_model \
    --num-runs 2000 \
    --seed 123 \
    --output eval_mlp_ne_large_s9_v1_results.csv \
    2>&1 | tee eval_s9_v1.log

echo ""
echo "======================================================================"
echo "Saving S9 experiment summary"
echo "======================================================================"
$PYTHON - <<'PYEOF'
import pandas as pd, numpy as np, os, datetime

df = pd.read_csv("eval_mlp_ne_large_s9_v1_results.csv")
ne_row  = df[df.deviator == "ZI-S9"].iloc[0]
mlp_row = df[df.deviator == "MLP-SAC"].iloc[0]
best_zi = df[df.deviator.str.startswith("ZI-")].sort_values("mean_profit").iloc[-1]

mlp_vs_ne   = mlp_row.mean_profit - ne_row.mean_profit
mlp_vs_best = mlp_row.mean_profit - best_zi.mean_profit
sig = mlp_vs_best > 2 * mlp_row.se

summary = f"""# MLP NE Experiment — S9 Background — 1M Steps (Generalization Run 2/2)

Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Experiment Configuration
- NE background strategy: S9  shade=[950,1050], eta=0.5
- Background agents: 24 (n_bg=24)
- Training steps: 1,000,000
- Training seed: 123  (different from S6 run seed=42)
- SAC hyperparams: lr=1e-4, buffer=500K, batch=256, gamma=0.99, ent_coef=auto
- Evaluation runs (Phase 3): 2,000 per deviator, seed=123
- ENV: lam=0.005, mean=1e5, r=0.01, shock_var=1e6, pv_var=5e6, q_max=10, sim_time=2000
- Model path: runs/sac_zi_ne_large_s9_v1/best_model.zip
- Results CSV: eval_mlp_ne_large_s9_v1_results.csv

## Phase 3 Results
| Deviator | Mean Profit | ±SE |
|----------|-------------|-----|
"""
for _, row in df.sort_values("mean_profit", ascending=False).iterrows():
    summary += f"| {row.deviator} | {row.mean_profit:.1f} | {row.se:.1f} |\n"

summary += f"""
## Key Metrics
- NE diagonal (ZI-S9 vs ZI-S9): {ne_row.mean_profit:.1f} ± {ne_row.se:.1f}
- Best ZI deviator ({best_zi.deviator}):         {best_zi.mean_profit:.1f} ± {best_zi.se:.1f}
- MLP-SAC deviator:                {mlp_row.mean_profit:.1f} ± {mlp_row.se:.1f}
- MLP Δ vs NE diagonal:            {mlp_vs_ne:+.1f}
- MLP Δ vs best ZI:                {mlp_vs_best:+.1f}
- Statistically significant (>2×SE): {'YES' if sig else 'NO'}
"""

out = "results/mlp_ne_s9_v1_summary.md"
os.makedirs("results", exist_ok=True)
with open(out, "w") as f:
    f.write(summary)
print(f"Summary saved → {out}")
print(summary)
PYEOF

echo ""
echo "All done. Results in results/mlp_ne_s6_v2_summary.md and results/mlp_ne_s9_v1_summary.md"
