"""
Nash Equilibrium Experiment (ne_experiment.py)
===============================================
Replicates the NE strategy search from Wah & Wellman (2016) — 1 deviator
agent + 15 background ZI agents — with two key improvements over the
existing equilibrium_experiment.py:

1. **Relative advantage metric** instead of raw deviator profit.
   Each run records both dev_profit and mean(bg_profits).
   relative_advantage = dev_profit - mean(bg_profits).
   Because all agents share the same fundamental path (CRN), subtracting
   the background mean removes common market noise, giving ~2-4× lower SE
   and making the NE question directly answerable: "does the deviator
   benefit by deviating?"

2. **SE-based NE detection** instead of exact argmax equality.
   For each row i, we compute gap = rel_mean[i, best_j] - rel_mean[i, i]
   and pooled_se = sqrt(SE[i,i]^2 + SE[i,best_j]^2), then
   t_stat = gap / pooled_se.  Strategy Si is a NE candidate iff
   t_stat < epsilon_multiple (default 2.0).

Outputs five CSVs and a console summary with ASCII t-stat bar chart.
"""

import argparse
import math
import random
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from marketsim.simulator.simulator import Simulator
from marketsim.agent.zero_intelligence_agent import ZIAgent


# ---------------------------------------------------------------------------
# Strategies (identical to equilibrium_experiment.py)
# ---------------------------------------------------------------------------

STRATEGIES = {
    0: {'shade': [0, 450],    'eta': 0.5},
    1: {'shade': [0, 600],    'eta': 0.5},
    2: {'shade': [90, 110],   'eta': 0.5},
    3: {'shade': [140, 160],  'eta': 0.5},
    4: {'shade': [190, 210],  'eta': 0.5},
    5: {'shade': [280, 320],  'eta': 0.5},
    6: {'shade': [380, 420],  'eta': 0.5},
    7: {'shade': [380, 420],  'eta': 1.0},
    8: {'shade': [460, 540],  'eta': 0.5},
    9: {'shade': [950, 1050], 'eta': 0.5},
}

STRATEGY_LABELS = [f"S{i}" for i in range(len(STRATEGIES))]

# ---------------------------------------------------------------------------
# Market environment (matches equilibrium_experiment.py)
# ---------------------------------------------------------------------------

ENV = {
    'lam':       0.005,
    'mean':      1e5,
    'r':         0.01,
    'shock_var': 1e6,
    'pv_var':    5e6,
    'q_max':     10,
    'sim_time':  2000,
    'n_bg':      15,
}

NUM_RUNS              = 500
BASE_SEED             = 52
SEED_BG_MULTIPLIER    = 1_000_000   # more headroom than the 100_000 used previously
EPSILON_MULTIPLE      = 2.0         # ~2.3% one-sided significance threshold


# ---------------------------------------------------------------------------
# Worker function (runs in a subprocess via mp.Pool)
# ---------------------------------------------------------------------------

def _run_cell(args):
    """
    Run NUM_RUNS simulations for one (bg_idx, dev_idx) cell and return
    aggregate statistics for both relative advantage and absolute deviator
    profit.

    Agent layout (created in this order to fix np.random stream):
        Agent 0          → deviator  (plays dev strategy)
        Agents 1 – n_bg  → background (all play bg strategy)

    CRN seeding: seed = base_seed + bg_idx * SEED_BG_MULTIPLIER + run_idx.
    dev_idx is intentionally excluded so all deviator strategies in the same
    row face identical fundamental paths and background agent sequences.
    This Common Random Numbers design collapses within-row variance and makes
    small between-strategy differences detectable.

    Returns:
        (bg_idx, dev_idx, mean_rel, std_rel, mean_abs, std_abs)
    """
    bg_idx, dev_idx, num_runs, base_seed, n_bg, lam, shock_var, pv_var = args
    bg  = STRATEGIES[bg_idx]
    dev = STRATEGIES[dev_idx]

    rel_advantages = []   # dev_profit - mean(bg_profits)
    abs_dev_profits = []  # raw deviator profit (backward compat)

    for run_idx in range(num_runs):
        # CRN seed: no dev_idx → same market conditions for all deviators in
        # this row.
        run_seed = base_seed + bg_idx * SEED_BG_MULTIPLIER + run_idx
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)

        sim = Simulator(
            num_background_agents=0,   # agents added manually below
            sim_time=ENV['sim_time'],
            num_assets=1,
            lam=lam,
            mean=ENV['mean'],
            r=ENV['r'],
            shock_var=shock_var,
            q_max=ENV['q_max'],
            pv_var=pv_var,
        )
        sim.agents = {}

        # Deviator FIRST so it always occupies np.random positions 0–19,
        # regardless of dev strategy.  This keeps CRN clean across dev variants.
        sim.agents[0] = ZIAgent(
            agent_id=0,
            market=sim.markets[0],
            q_max=ENV['q_max'],
            shade=dev['shade'],
            eta=dev['eta'],
            pv_var=pv_var,
        )

        # Background agents 1..n_bg
        for i in range(1, n_bg + 1):
            sim.agents[i] = ZIAgent(
                agent_id=i,
                market=sim.markets[0],
                q_max=ENV['q_max'],
                shade=bg['shade'],
                eta=bg['eta'],
                pv_var=pv_var,
            )

        # Market.event_queue is created with no seed (random.Random(None) → OS
        # entropy).  Reseed it deterministically so order-shuffle is reproducible.
        sim.markets[0].event_queue.rand = random.Random(run_seed + 1)

        sim.run()

        fv = sim.markets[0].get_final_fundamental()

        def profit(agent):
            return agent.get_pos_value() + agent.position * fv + agent.cash

        dev_profit = profit(sim.agents[0])
        bg_profits  = [profit(sim.agents[i]) for i in range(1, n_bg + 1)]
        mean_bg     = float(np.mean(bg_profits))

        abs_dev_profits.append(dev_profit)
        rel_advantages.append(dev_profit - mean_bg)

    mean_rel = float(np.mean(rel_advantages))
    std_rel  = float(np.std(rel_advantages, ddof=1))
    mean_abs = float(np.mean(abs_dev_profits))
    std_abs  = float(np.std(abs_dev_profits, ddof=1))

    return bg_idx, dev_idx, mean_rel, std_rel, mean_abs, std_abs


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_ne_experiment(num_runs=NUM_RUNS, n_processes=None, base_seed=BASE_SEED):
    """
    Run the full 10×10 strategy matrix.

    Returns a dict with four (10, 10) DataFrames:
        'rel_mean' : mean relative advantage  (dev_profit - mean_bg_profit)
        'rel_se'   : standard error of relative advantage
        'abs_mean' : mean absolute deviator profit
        'abs_se'   : standard error of absolute deviator profit
    """
    n = len(STRATEGIES)
    n_bg = ENV['n_bg']
    lam = ENV['lam']
    shock_var = ENV['shock_var']
    pv_var = ENV['pv_var']
    tasks = [(bg, dev, num_runs, base_seed, n_bg, lam, shock_var, pv_var)
             for bg in range(n) for dev in range(n)]

    if n_processes is None:
        n_processes = min(mp.cpu_count(), len(tasks))

    rel_mean_arr = np.zeros((n, n))
    rel_std_arr  = np.zeros((n, n))
    abs_mean_arr = np.zeros((n, n))
    abs_std_arr  = np.zeros((n, n))

    total_sims = len(tasks) * num_runs
    print(
        f"\nStrategy matrix: {n}×{n} cells  |  {num_runs} runs/cell  |  "
        f"{total_sims:,} total simulations  |  {n_processes} processes  |  "
        f"base_seed={base_seed}  |  n_bg={n_bg}"
    )

    with mp.Pool(n_processes) as pool:
        for bg_idx, dev_idx, m_rel, s_rel, m_abs, s_abs in tqdm(
            pool.imap_unordered(_run_cell, tasks),
            total=len(tasks),
            desc="cells completed",
        ):
            rel_mean_arr[bg_idx, dev_idx] = m_rel
            rel_std_arr[bg_idx, dev_idx]  = s_rel
            abs_mean_arr[bg_idx, dev_idx] = m_abs
            abs_std_arr[bg_idx, dev_idx]  = s_abs

    idx  = pd.Index(STRATEGY_LABELS, name="BG Strategy")
    cols = pd.Index(STRATEGY_LABELS, name="Deviator")

    sqrt_n = math.sqrt(num_runs)
    return {
        'rel_mean': pd.DataFrame(rel_mean_arr,              index=idx, columns=cols),
        'rel_se':   pd.DataFrame(rel_std_arr  / sqrt_n,     index=idx, columns=cols),
        'abs_mean': pd.DataFrame(abs_mean_arr,              index=idx, columns=cols),
        'abs_se':   pd.DataFrame(abs_std_arr  / sqrt_n,     index=idx, columns=cols),
    }


# ---------------------------------------------------------------------------
# NE detection
# ---------------------------------------------------------------------------

def detect_nash_equilibria(rel_mean: pd.DataFrame, rel_se: pd.DataFrame,
                           epsilon_multiple: float = EPSILON_MULTIPLE) -> pd.DataFrame:
    """
    For each row i (background strategy Si), determine whether Si is a Nash
    equilibrium candidate.

    A strategy Si is a NE candidate iff the best deviator strategy (argmax of
    the row) is Si itself — i.e. no alternative strategy earns a higher mean
    relative advantage against a Si background.

    t_stat is computed as a confidence measure: how many pooled SEs does the
    best non-diagonal strategy beat the diagonal by?  For NE candidates
    (diagonal is the argmax) t_stat = 0.  For non-NE rows it quantifies how
    strongly the diagonal is rejected.

    Method:
        best_j    = argmax_j rel_mean[i, j]
        gap       = rel_mean[i, best_j] - rel_mean[i, i]   (≥ 0 by definition)
        pooled_se = sqrt(rel_se[i,i]^2 + rel_se[i, best_j]^2)
        t_stat    = gap / pooled_se   (0 when diagonal is argmax)
        is_ne     = (best_j == i)     ← diagonal must be the argmax

    epsilon_multiple is unused for NE classification; it controls only the
    threshold line in the t-stat bar chart.

    Returns a DataFrame with columns:
        bg_strategy, best_deviator, rel_diag, rel_best,
        gap, pooled_se, t_stat, is_ne_candidate
    """
    rows = []
    for i, bg_label in enumerate(rel_mean.index):
        row_vals = rel_mean.loc[bg_label]
        row_se   = rel_se.loc[bg_label]

        best_label = row_vals.idxmax()
        best_j     = STRATEGY_LABELS.index(best_label)

        rel_diag = float(row_vals[bg_label])
        rel_best = float(row_vals[best_label])
        gap      = rel_best - rel_diag

        se_diag  = float(row_se[bg_label])
        se_best  = float(row_se[best_label])

        # pooled SE (conservative under CRN)
        pooled_se = math.sqrt(se_diag**2 + se_best**2) if (se_diag > 0 or se_best > 0) else 0.0

        # t_stat = 0 when diagonal is already the best (no gap to test)
        if best_label == bg_label:
            t_stat = 0.0
        else:
            t_stat = gap / pooled_se if pooled_se > 0 else float('inf')

        # NE requires the diagonal to be the argmax (best response = own strategy)
        is_ne = (best_label == bg_label)

        rows.append({
            'bg_strategy':   bg_label,
            'best_deviator': best_label,
            'rel_diag':      rel_diag,
            'rel_best':      rel_best,
            'gap':           gap,
            'pooled_se':     pooled_se,
            't_stat':        t_stat,
            'is_ne_candidate': is_ne,
        })

    return pd.DataFrame(rows).set_index('bg_strategy')


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_results(results: dict, ne_df: pd.DataFrame,
                    num_runs: int, epsilon_multiple: float) -> None:
    """Print a console summary in five sections."""

    rel_mean = results['rel_mean']
    rel_se   = results['rel_se']
    abs_mean = results['abs_mean']
    abs_se   = results['abs_se']

    # ── Section 1: Relative advantage matrix ────────────────────────────────
    print("\n" + "=" * 80)
    print("SECTION 1: RELATIVE ADVANTAGE MATRIX  (dev_profit - mean_bg_profit)")
    print("Rows = background strategy (what the 15 BG agents play)")
    print("Cols = deviator strategy   (what the 1 deviator agent plays)")
    print("★  = best deviator column for that row")
    print("=" * 80)
    _print_matrix(rel_mean, rel_se, mark_max=True)

    # ── Section 2: Absolute deviator profit (backward compat) ───────────────
    print("\n" + "=" * 80)
    print("SECTION 2: ABSOLUTE DEVIATOR PROFIT  (raw deviator profit)")
    print("=" * 80)
    _print_matrix(abs_mean, abs_se, mark_max=False)

    # ── Section 3: NE detection table ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("SECTION 3: NE DETECTION")
    print("NE candidate iff best_deviator == bg_strategy (diagonal is argmax)")
    print(f"t_stat = gap / pooled_SE  (confidence measure; threshold line at {epsilon_multiple})")
    print("=" * 80)
    header = (f"{'BG':>4}  {'Best Dev':>8}  {'Diag':>10}  {'Best':>10}  "
              f"{'Gap':>8}  {'SE':>8}  {'t-stat':>7}  {'NE?':>5}")
    print(header)
    print("─" * len(header))
    for bg_label in ne_df.index:
        row = ne_df.loc[bg_label]
        ne_str = " ✓" if row['is_ne_candidate'] else "  "
        print(
            f"{bg_label:>4}  {row['best_deviator']:>8}  "
            f"{row['rel_diag']:>10.1f}  {row['rel_best']:>10.1f}  "
            f"{row['gap']:>8.1f}  {row['pooled_se']:>8.1f}  "
            f"{row['t_stat']:>7.2f}  {ne_str}"
        )

    # ── Section 4: ASCII t-stat bar chart ───────────────────────────────────
    print("\n" + "=" * 80)
    print("SECTION 4: T-STAT BAR CHART  (measures how decisively best≠diagonal; 0 = NE)")
    print(f"Threshold line at t = {epsilon_multiple}  (informational only)")
    print("=" * 80)
    max_t     = max(ne_df['t_stat'].replace(float('inf'), 20.0))
    bar_width = 40
    threshold_pos = int(epsilon_multiple / max(max_t, epsilon_multiple) * bar_width)
    for bg_label in ne_df.index:
        t = ne_df.loc[bg_label, 't_stat']
        t_capped = min(t, max_t)
        filled   = int(t_capped / max_t * bar_width)
        bar      = "█" * filled + "░" * (bar_width - filled)
        marker   = "▲" if ne_df.loc[bg_label, 'is_ne_candidate'] else " "
        print(f"  {bg_label:>3}  |{bar}|  t={t:.2f} {marker}")
    # print threshold guide
    guide = " " * 7 + "|" + " " * threshold_pos + "^" + f" ε={epsilon_multiple}"
    print(guide)

    # ── Section 5: NE candidate strategy details ─────────────────────────────
    print("\n" + "=" * 80)
    print("SECTION 5: NASH EQUILIBRIUM CANDIDATES")
    print("=" * 80)
    ne_rows = ne_df[ne_df['is_ne_candidate']]
    if ne_rows.empty:
        print("  No pure-strategy Nash equilibrium found among the 10 strategies.")
    else:
        for bg_label in ne_rows.index:
            idx  = int(bg_label[1:])
            s    = STRATEGIES[idx]
            row  = ne_rows.loc[bg_label]
            print(f"  {bg_label}  shade={s['shade']}  eta={s['eta']}")
            print(f"       Diagonal is argmax  (best deviator = {row['best_deviator']}, "
                  f"gap = {row['gap']:.1f}, t_stat = {row['t_stat']:.2f})")
    print("=" * 80)


def _print_matrix(mean_df: pd.DataFrame, se_df: pd.DataFrame,
                  mark_max: bool) -> None:
    """Pretty-print a 10×10 matrix with optional ★ on the row maximum."""
    cols  = list(mean_df.columns)
    width = 11
    header_row = "     " + "".join(f"{c:>{width}}" for c in cols)
    print(header_row)
    print("─" * len(header_row))
    for bg_label in mean_df.index:
        row_m = mean_df.loc[bg_label]
        row_s = se_df.loc[bg_label]
        best  = row_m.idxmax() if mark_max else None
        cells = []
        for col in cols:
            val = row_m[col]
            se  = row_s[col]
            tag = "★" if (mark_max and col == best) else " "
            cells.append(f"{val:>7.0f}{tag}±{se:<3.0f}")
        print(f"{bg_label:>4}  " + "  ".join(cells))


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_results(results: dict, ne_df: pd.DataFrame, prefix: str = "ne") -> None:
    """Save five CSV files: rel_mean, rel_se, abs_mean, abs_se, ne_summary."""
    import os
    outdir = os.path.dirname(prefix)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    files = {
        f"{prefix}_rel_mean.csv": results['rel_mean'],
        f"{prefix}_rel_se.csv":   results['rel_se'],
        f"{prefix}_abs_mean.csv": results['abs_mean'],
        f"{prefix}_abs_se.csv":   results['abs_se'],
        f"{prefix}_ne_summary.csv": ne_df,
    }
    for fname, df in files.items():
        df.to_csv(fname)
        print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Nash Equilibrium experiment with relative-advantage metric and SE-based detection."
    )
    parser.add_argument("--num-runs",   type=int,   default=NUM_RUNS,
                        help=f"Simulation runs per cell (default: {NUM_RUNS})")
    parser.add_argument("--processes",  type=int,   default=None,
                        help="Worker processes (default: cpu_count)")
    parser.add_argument("--base-seed",  type=int,   default=BASE_SEED,
                        help=f"Base random seed (default: {BASE_SEED})")
    parser.add_argument("--epsilon",    type=float, default=EPSILON_MULTIPLE,
                        help=f"t-stat threshold for NE candidacy (default: {EPSILON_MULTIPLE})")
    parser.add_argument("--prefix",     type=str,   default="results/ne_search/ne",
                        help="Output CSV filename prefix (default: 'results/ne_search/ne')")
    parser.add_argument("--n-bg",       type=int,   default=ENV['n_bg'],
                        help=f"Number of background agents (default: {ENV['n_bg']})")
    parser.add_argument("--lam",        type=float, default=None,
                        help="Market arrival rate lambda (default: ENV default 0.005)")
    parser.add_argument("--shock-var",  type=float, default=None,
                        help="Fundamental shock variance (default: ENV default 1e6)")
    parser.add_argument("--pv-var",     type=float, default=None,
                        help="Private value variance (default: ENV default 5e6)")
    args = parser.parse_args()

    ENV['n_bg'] = args.n_bg
    if args.lam is not None:
        ENV['lam'] = args.lam
    if args.shock_var is not None:
        ENV['shock_var'] = args.shock_var
    if args.pv_var is not None:
        ENV['pv_var'] = args.pv_var

    results = run_ne_experiment(
        num_runs   = args.num_runs,
        n_processes= args.processes,
        base_seed  = args.base_seed,
    )

    ne_df = detect_nash_equilibria(
        rel_mean         = results['rel_mean'],
        rel_se           = results['rel_se'],
        epsilon_multiple = args.epsilon,
    )

    display_results(results, ne_df, num_runs=args.num_runs, epsilon_multiple=args.epsilon)

    print("\nSaving results...")
    save_results(results, ne_df, prefix=args.prefix)
    print("Done.")


if __name__ == "__main__":
    main()
