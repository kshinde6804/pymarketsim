"""
combine_ne_results.py
=====================
Reads the 9 ne_summary CSVs (3 envs × 3 seeds, n_bg=24) and produces a single
combined table with environment labels along the y axis.

Output columns: Environment, Seed, S0–S9 (1 = NE candidate, 0 = not)
Also prints a human-readable table to stdout.
"""

import pandas as pd
import os

ENVS   = ['A', 'B', 'C']
SEEDS  = [42, 123, 456]
PREFIX = "results/ne_search/ne_env{env}_nb24_s{seed}_ne_summary.csv"
OUT    = "results/ne_search/ne_nb24_combined_table.csv"

STRATEGIES = [f"S{i}" for i in range(10)]

rows = []
for env in ENVS:
    for seed in SEEDS:
        path = PREFIX.format(env=env, seed=seed)
        if not os.path.exists(path):
            print(f"  WARNING: missing {path}")
            continue
        df = pd.read_csv(path, index_col=0)
        ne_set = set(df.index[df['is_ne_candidate'] == True].tolist())
        row = {'Environment': f'Env {env}', 'Seed': seed}
        for s in STRATEGIES:
            row[s] = 1 if s in ne_set else 0
        rows.append(row)

combined = pd.DataFrame(rows)
combined.to_csv(OUT, index=False)
print(f"Saved combined table to {OUT}\n")

# --- pivot table: rows=env, cols=seed, cells=NE strategy list ---
pivot_rows = []
for env in ENVS:
    row = {'Environment': f'Env {env}'}
    for seed in SEEDS:
        sub = combined[(combined['Environment'] == f'Env {env}') & (combined['Seed'] == seed)]
        if sub.empty:
            row[f'Seed {seed}'] = '—'
        else:
            ne_strats = [s for s in STRATEGIES if sub.iloc[0][s] == 1]
            row[f'Seed {seed}'] = ', '.join(ne_strats) if ne_strats else '(none)'
    pivot_rows.append(row)

pivot = pd.DataFrame(pivot_rows).set_index('Environment')
pivot_out = OUT.replace('combined_table', 'pivot_table')
pivot.to_csv(pivot_out)
print(f"Saved pivot table to {pivot_out}\n")

# pretty print
seed_cols = [f'Seed {s}' for s in SEEDS]
col_w = max(max(len(str(v)) for v in pivot[c]) for c in seed_cols) + 2
header = f"{'':10}" + "".join(f"{c:>{col_w}}" for c in seed_cols)
print(header)
print("─" * len(header))
for env in ENVS:
    row = pivot.loc[f'Env {env}']
    cells = "".join(f"{row[c]:>{col_w}}" for c in seed_cols)
    print(f"{'Env '+env:<10}{cells}")

