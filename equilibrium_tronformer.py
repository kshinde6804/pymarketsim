"""
Equilibrium Experiment — with TRONformer Transformer DQN Deviator
==================================================================
Extends the 10×10 ZI strategy matrix from equilibrium_experiment.py to a
10×11 matrix by adding the TRONformer-trained agent as an additional deviator.

For each of the 10 background ZI strategies, the TRONformer agent competes as
the deviator alongside the 10 existing ZI deviator strategies.  The experiment
loads pre-computed ZI×ZI results from equilibrium_results.csv and only runs
the 10 new TRONformer-deviator cells.

Usage
-----
    python equilibrium_tronformer.py
    python equilibrium_tronformer.py --model runs/tronformer_v1/best_model.pt
    python equilibrium_tronformer.py --num-runs 500 --processes 4
    python equilibrium_tronformer.py --num-runs 2 --processes 1   # smoke test
"""

import argparse
import collections

import numpy as np
import pandas as pd
import multiprocessing as mp
import torch
from tqdm import tqdm

from marketsim.agent.tronformer_agent import TRONformerPolicy, SEQ_LEN
from marketsim.wrappers.tron_env import TRONEnv

# ---------------------------------------------------------------------------
# Shade bin options (must match train_tronformer.py)
# ---------------------------------------------------------------------------

UNIFORM_SHADE_BINS = np.linspace(0, 600, 42)

SKEWED_SHADE_BINS = np.concatenate([
    np.linspace(0, 300, 11)[:-1],
    np.linspace(300, 600, 32),
])

# ---------------------------------------------------------------------------
# Strategy / environment constants (inlined from equilibrium_experiment.py)
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

STRATEGY_LABELS = {
    0: 'S0 [0,450] η=0.5',
    1: 'S1 [0,600] η=0.5',
    2: 'S2 [90,110] η=0.5',
    3: 'S3 [140,160] η=0.5',
    4: 'S4 [190,210] η=0.5',
    5: 'S5 [280,320] η=0.5',
    6: 'S6 [380,420] η=0.5',
    7: 'S7 [380,420] η=1.0',
    8: 'S8 [460,540] η=0.5',
    9: 'S9 [950,1050] η=0.5',
}

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL   = "runs/tronformer_v1/best_model.pt"
DEFAULT_RUNS    = 500
DEFAULT_N_BG    = 24
DEFAULT_LAM_ZI  = 0.02
TRONFORMER_LABEL = "TRONformer"
NORMALIZERS      = {"fundamental": 1e5, "invt": 10, "reward": 1e3, "pv": 5e5}
BASELINE_CSV     = "results/equilibrium/equilibrium_results.csv"


# ---------------------------------------------------------------------------
# Worker: TRONEnv rollout with rolling obs buffer (no LSTM state)
# ---------------------------------------------------------------------------

def _run_tronformer_cell(args):
    """Run num_runs TRONEnv episodes with TRONformer vs one fixed ZI background.

    Uses TRONEnv (matching the training environment exactly).
    Maintains a rolling obs_buffer (deque of maxlen seq_len) per rollout;
    stacks to (1, len, 14) for each forward pass and takes Q at last position.

    Args:
        args: (bg_idx, weights_path, num_runs, n_bg, lam_zi, n_layers, seq_len)

    Returns:
        (bg_idx, mean_deviator_profit)
    """
    bg_idx, weights_path, num_runs, n_bg, lam_zi, n_layers, seq_len, shade_bins = args

    bg = STRATEGIES[bg_idx]
    reward_norm = NORMALIZERS["reward"]

    policy = TRONformerPolicy(input_dim=14, n_layers=n_layers, shade_bins=shade_bins)
    policy.load_state_dict(torch.load(weights_path, map_location="cpu"))
    policy.eval()

    env_kwargs = dict(
        num_background_agents=n_bg,
        sim_time=ENV["sim_time"],
        lam=ENV["lam"],
        lam_zi=lam_zi,
        mean=ENV["mean"],
        r=ENV["r"],
        shock_var=ENV["shock_var"],
        q_max=ENV["q_max"],
        pv_var=ENV["pv_var"],
        shade=[250, 500],
        normalizers=NORMALIZERS,
        bg_strategies=[{'shade': bg['shade'], 'eta': bg['eta']}],
        warmup_fraction=0.0,
    )
    if shade_bins is not None:
        env_kwargs["shade_bins"] = shade_bins
    env = TRONEnv(**env_kwargs)

    dev_profits = []
    for _ in range(num_runs):
        obs, _ = env.reset()
        obs_buffer: collections.deque = collections.deque(maxlen=seq_len)
        ep_reward = 0.0
        done = False

        while not done:
            obs_buffer.append(obs.astype(np.float32))
            obs_seq = np.stack(list(obs_buffer))          # (cur_len, 14)
            obs_t = torch.tensor(
                obs_seq, dtype=torch.float32
            ).unsqueeze(0)                                # (1, cur_len, 14)

            with torch.no_grad():
                Q_shade, Q_eta = policy(obs_t)            # (1, cur_len, n_*)

            shade_idx = int(Q_shade[0, -1, :].argmax().item())
            eta_idx   = int(Q_eta[0, -1, :].argmax().item())

            obs, r, terminated, truncated, _ = env.step(
                np.array([shade_idx, eta_idx])
            )
            ep_reward += r
            done = terminated or truncated

        dev_profits.append(ep_reward * reward_norm)

    return bg_idx, float(np.mean(dev_profits))


# ---------------------------------------------------------------------------
# Run the TRONformer deviator column
# ---------------------------------------------------------------------------

def run_tronformer_column(
    weights_path: str,
    num_runs: int = DEFAULT_RUNS,
    n_processes: int = None,
    lam_zi: float = DEFAULT_LAM_ZI,
    n_layers: int = 1,
    seq_len: int = SEQ_LEN,
    shade_bins: np.ndarray = None,
) -> np.ndarray:
    """Run num_runs TRONEnv episodes for each of the 10 ZI background strategies.

    Returns:
        tronformer_column: float array of shape (10,) — mean TRONformer
                           deviator profit for each ZI background strategy.
    """
    n_strats = len(STRATEGIES)
    n_bg = ENV['n_bg']
    tasks = [(bg, weights_path, num_runs, n_bg, lam_zi, n_layers, seq_len, shade_bins) for bg in range(n_strats)]

    if n_processes is None:
        n_processes = min(mp.cpu_count(), n_strats)

    tronformer_column = np.zeros(n_strats)

    total_sims = n_strats * num_runs
    print(
        f"TRONformer deviator column:  {n_strats} cells  |  {num_runs} runs/cell  |  "
        f"{total_sims:,} total simulations  |  {n_processes} processes"
    )
    print(f"  Model: {weights_path}")
    print(f"  ENV: lam={ENV['lam']}, lam_zi={lam_zi}, sim_time={ENV['sim_time']}, "
          f"n_bg={n_bg}.\n")

    with mp.Pool(n_processes) as pool:
        for bg_idx, tf_mean in tqdm(
            pool.imap_unordered(_run_tronformer_cell, tasks),
            total=n_strats,
            desc="TRONformer cells completed",
        ):
            tronformer_column[bg_idx] = tf_mean

    return tronformer_column


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_extended_results(
    zi_df: pd.DataFrame,
    tronformer_column: np.ndarray,
    model_label: str = TRONFORMER_LABEL,
) -> pd.DataFrame:
    """Display 10×11 advantage table and best-deviation analysis."""
    tf_series = pd.Series(tronformer_column, index=zi_df.index, name=model_label)
    df = pd.concat([zi_df, tf_series], axis=1)

    # ── Advantage matrix ──────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("DEVIATOR PROFIT TABLE  (extended with TRONformer deviator)")
    print("Rows = background strategy  |  Cols = deviator strategy")
    print(f"Value = mean deviator profit  |  Last col ({model_label}) = TRONformer transformer DQN")
    print("=" * 90)
    print(df.round(1).to_string())

    # ── Best deviator per row ─────────────────────────────────────────────
    best_dev = df.idxmax(axis=1)
    best_adv = df.max(axis=1)

    print("\n" + "=" * 90)
    print("BEST DEVIATOR STRATEGY FOR EACH BACKGROUND STRATEGY")
    print("=" * 90)
    print(
        f"{'BG Strategy':>12} │ {'Best Deviation':>14} │ {'Advantage':>10} │ NE? │ "
        f"TRONformer better than ZI best?"
    )
    print("─" * 80)

    ne_candidates = []
    tf_beats_zi = []

    for bg_label in df.index:
        best    = best_dev[bg_label]
        adv     = best_adv[bg_label]
        zi_best = zi_df.idxmax(axis=1)[bg_label]
        zi_adv  = zi_df.max(axis=1)[bg_label]
        tf_val  = tronformer_column[list(df.index).index(bg_label)]
        is_ne   = (bg_label == best)
        tf_wins = (best == model_label)

        ne_str  = " ✓" if is_ne else ""
        win_str = (
            f"Yes (+{tf_val - zi_adv:+.0f})"
            if tf_wins
            else f"No  ({zi_best} +{zi_adv:.0f})"
        )

        if is_ne:
            ne_candidates.append(bg_label)
        if tf_wins:
            tf_beats_zi.append(bg_label)

        print(f"{bg_label:>12} │ {best:>14} │ {adv:>10.1f}{ne_str:3s} │     │ {win_str}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(
        f"  TRONformer is the best deviator against {len(tf_beats_zi)}/10 "
        f"background strategies:"
    )
    for bg in tf_beats_zi:
        bg_idx  = list(df.index).index(bg)
        s       = STRATEGIES[int(bg[1:])]
        zi_best = zi_df.idxmax(axis=1)[bg]
        zi_adv  = zi_df.max(axis=1)[bg]
        print(
            f"    {bg} (bg={s['shade']} η={s['eta']})  "
            f"TRONformer={tronformer_column[bg_idx]:.1f}  "
            f"best-ZI={zi_best} ({zi_adv:.1f})  "
            f"Δ={tronformer_column[bg_idx] - zi_adv:+.1f}"
        )

    if ne_candidates:
        print(f"\n  Nash equilibrium candidates (pure-strategy): {ne_candidates}")
    else:
        print("\n  No pure-strategy NE found among the 11 strategies.")

    print("=" * 90)
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Path to TRONformerPolicy .pt state_dict (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--num-runs", type=int, default=DEFAULT_RUNS,
        help=f"Simulations per TRONformer-deviator cell (default: {DEFAULT_RUNS})",
    )
    p.add_argument(
        "--processes", type=int, default=None,
        help="Worker processes (default: cpu_count)",
    )
    p.add_argument(
        "--output", default="results/equilibrium/equilibrium_tronformer_results.csv",
        help="Output CSV path (default: results/equilibrium/equilibrium_tronformer_results.csv)",
    )
    p.add_argument(
        "--n-bg", type=int, default=DEFAULT_N_BG,
        help=f"Background agents per simulation (default: {DEFAULT_N_BG})",
    )
    p.add_argument(
        "--baseline", default=BASELINE_CSV,
        help=f"ZI×ZI baseline CSV path (default: {BASELINE_CSV})",
    )
    p.add_argument(
        "--lam-zi", type=float, default=DEFAULT_LAM_ZI,
        help=f"RL agent arrival rate (default: {DEFAULT_LAM_ZI})",
    )
    p.add_argument(
        "--n-layers", type=int, default=1,
        help="Number of Pre-LN transformer blocks in the loaded model (default: 1)",
    )
    p.add_argument(
        "--seq-len", type=int, default=SEQ_LEN,
        help=f"Rolling context window used during evaluation (default: {SEQ_LEN})",
    )
    p.add_argument(
        "--skew-bins", action="store_true",
        help="Use skewed shade bins (must match how the model was trained)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    ENV['n_bg'] = args.n_bg

    shade_bins = SKEWED_SHADE_BINS if args.skew_bins else None

    # ── Load ZI×ZI baseline ───────────────────────────────────────────────
    print(f"Loading ZI×ZI baseline from {args.baseline} ...")
    zi_df = pd.read_csv(args.baseline, index_col=0)
    print(f"  Loaded {zi_df.shape[0]}×{zi_df.shape[1]} matrix.\n")

    # ── Run TRONformer deviator column ────────────────────────────────────
    tronformer_column = run_tronformer_column(
        weights_path=args.model,
        num_runs=args.num_runs,
        n_processes=args.processes,
        lam_zi=args.lam_zi,
        n_layers=args.n_layers,
        seq_len=args.seq_len,
        shade_bins=shade_bins,
    )

    # ── Display extended results ──────────────────────────────────────────
    ext_df = display_extended_results(zi_df, tronformer_column)

    # ── Save ─────────────────────────────────────────────────────────────
    ext_df.to_csv(args.output)
    print(f"\nExtended advantage matrix saved to {args.output}")


if __name__ == "__main__":
    main()
