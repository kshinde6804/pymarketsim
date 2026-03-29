"""
eval_mlp_ne.py — Evaluate a trained MLP against 15 NE-strategy background traders.

Phase 3 of the large-scale MLP experiment:
  1. Load a pre-trained SAC model (from train_mlp_ne.py)
  2. Run it as a deviator against 15 background agents all playing strategy S*
  3. Also run ZI strategies (including S* itself) as benchmarks
  4. Report profits and advantage over ZI NE baseline

The evaluation uses the same Simulator setup as the equilibrium experiment
(not ZIEnv), so profits are directly comparable to the ZI×ZI advantage table.

Usage
-----
    python eval_mlp_ne.py --ne-strategy 5 --model runs/sac_zi_ne_s5_TAG/best_model
    python eval_mlp_ne.py --ne-strategy 5 --model runs/sac_zi_ne_s5_TAG/best_model --num-runs 2000
"""

import argparse
import math
import random
from typing import List

import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

from marketsim.simulator.simulator import Simulator
from marketsim.agent.zero_intelligence_agent import ZIAgent
from marketsim.agent.mlp_agent import MLPAgent
from marketsim.fourheap.constants import BUY, SELL
from marketsim.fourheap.order import Order
from marketsim.wrappers.metrics import (
    midprice_move,
    queue_imbalance,
    realized_volatility,
    relative_strength_index,
    volume_imbalance,
)


# ---------------------------------------------------------------------------
# Strategies (inlined)
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

DEFAULT_RUNS  = 2000
BASE_SEED     = 42
SHADE_RANGE   = [0, 600]
NORMALIZERS   = {"fundamental": 1e5, "invt": 10, "reward": 1e3, "pv": 5e5}


# ---------------------------------------------------------------------------
# SACDeviator (same as equilibrium_mlp.py)
# ---------------------------------------------------------------------------

class SACDeviator(MLPAgent):
    """MLP agent driven by a pre-trained SB3 SAC policy."""

    def __init__(self, agent_id: int, market, q_max: int, pv_var: float,
                 shade_range: List, normalizers: dict, sac_model):
        super().__init__(
            agent_id=agent_id,
            market=market,
            q_max=q_max,
            pv_var=pv_var,
            shade=[250, 500],
            shade_range=shade_range,
            normalizers=normalizers,
        )
        self.sac_model = sac_model

    def build_obs(self) -> np.ndarray:
        t        = self.market.get_time()
        sim_time = self.market.end_time

        fundamental_value = self.market.fundamental.get_value_at(t)
        best_ask  = self.market.order_book.get_best_ask()
        best_bid  = self.market.order_book.get_best_bid()
        inventory = self.position

        mp_delta = midprice_move(self.market)
        vol_imb  = volume_imbalance(self.market)
        que_imb  = queue_imbalance(self.market)
        rv       = realized_volatility(self.market)
        rsi      = relative_strength_index(self.market)

        est_fund = self.estimate_fundamental()
        pv_buy   = self.pv.value_for_exchange(self.position, BUY)
        pv_sell  = self.pv.value_for_exchange(self.position, SELL)

        N    = self.normalizers
        pv_n = N.get("pv", 5e5)

        time_left_n = (sim_time - t) / sim_time if sim_time > 0 else 0.0
        fund_n      = fundamental_value / N["fundamental"]
        ask_n       = 1.0 if math.isinf(best_ask) else best_ask / N["fundamental"]
        bid_n       = 0.0 if math.isinf(best_bid) else best_bid / N["fundamental"]
        inv_n       = np.clip(inventory / N["invt"], -1.0, 1.0)
        mp_n        = np.clip(mp_delta / 1e2, -1.0, 1.0)
        rsi_n       = rsi / 100.0
        est_n       = est_fund / N["fundamental"]
        pv_buy_n    = np.clip(pv_buy  / pv_n, -1.0, 1.0)
        pv_sell_n   = np.clip(pv_sell / pv_n, -1.0, 1.0)

        obs = np.array(
            [time_left_n, fund_n, ask_n, bid_n, inv_n, mp_n,
             vol_imb, que_imb, np.clip(rv, 0.0, 1.0), rsi_n,
             est_n, pv_buy_n, pv_sell_n],
            dtype=np.float64,
        )
        if np.isnan(obs[9]):
            obs[9] = 0.5
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    def take_action(self) -> List[Order]:
        obs = self.build_obs()
        action, _ = self.sac_model.predict(obs, deterministic=True)
        shade_norm = float(action[0])
        eta        = float(action[1])

        d_min, d_max = self.shade_range
        shade_val    = shade_norm * (d_max - d_min) + d_min

        t        = self.market.get_time()
        side     = random.choice([BUY, SELL])
        estimate = self.estimate_fundamental()
        pv_value = self.pv.value_for_exchange(self.position, side)

        if side == BUY:
            price = estimate + pv_value - shade_val
        else:
            price = estimate + pv_value + shade_val

        if eta < 1.0:
            base_price = estimate + pv_value
            if side == BUY:
                best = self.market.order_book.get_best_ask()
                if (base_price - best) > eta * shade_val and not math.isinf(best):
                    price = best
            else:
                best = self.market.order_book.get_best_bid()
                if (best - base_price) > eta * shade_val and not math.isinf(best):
                    price = best

        self._order_counter += 1
        order_id = self.agent_id * 1000000 + self._order_counter
        return [Order(price=price, quantity=1, agent_id=self.agent_id,
                      time=t, order_type=side, order_id=order_id)]


# ---------------------------------------------------------------------------
# Worker: run MLP deviator vs NE background agents
# ---------------------------------------------------------------------------

def _run_mlp_vs_ne(args):
    """Run num_runs sims with MLP deviator vs NE background strategy."""
    ne_idx, model_path, num_runs, base_seed = args

    from stable_baselines3 import SAC
    sac_model = SAC.load(model_path, device="cpu")
    sac_model.policy.set_training_mode(False)

    ne_strat   = STRATEGIES[ne_idx]
    dev_profits = []

    for run_idx in range(num_runs):
        run_seed = base_seed + run_idx
        random.seed(run_seed)
        np.random.seed(run_seed)

        sim = Simulator(
            num_background_agents=0,
            sim_time=ENV["sim_time"],
            num_assets=1,
            lam=ENV["lam"],
            mean=ENV["mean"],
            r=ENV["r"],
            shock_var=ENV["shock_var"],
            q_max=ENV["q_max"],
            pv_var=ENV["pv_var"],
        )
        sim.agents = {}

        sim.agents[0] = SACDeviator(
            agent_id=0,
            market=sim.markets[0],
            q_max=ENV["q_max"],
            pv_var=ENV["pv_var"],
            shade_range=SHADE_RANGE,
            normalizers=NORMALIZERS,
            sac_model=sac_model,
        )

        for i in range(1, ENV["n_bg"] + 1):
            sim.agents[i] = ZIAgent(
                agent_id=i,
                market=sim.markets[0],
                q_max=ENV["q_max"],
                shade=ne_strat["shade"],
                eta=ne_strat["eta"],
                pv_var=ENV["pv_var"],
            )

        sim.run()

        fv  = sim.markets[0].get_final_fundamental()
        dev = sim.agents[0]
        dev_profits.append(dev.get_pos_value() + dev.position * fv + dev.cash)

    return float(np.mean(dev_profits)), float(np.std(dev_profits, ddof=1))


# ---------------------------------------------------------------------------
# Worker: run ZI deviator vs NE background agents (CRN)
# ---------------------------------------------------------------------------

def _run_zi_vs_ne(args):
    """Run num_runs sims with a ZI deviator vs NE background strategy.

    Uses CRN so all deviators share the same fundamental + BG arrivals.
    """
    ne_idx, dev_idx, num_runs, base_seed = args
    ne_strat  = STRATEGIES[ne_idx]
    dev_strat = STRATEGIES[dev_idx]
    dev_profits = []

    for run_idx in range(num_runs):
        run_seed = base_seed + run_idx
        random.seed(run_seed)
        np.random.seed(run_seed)

        sim = Simulator(
            num_background_agents=0,
            sim_time=ENV["sim_time"],
            num_assets=1,
            lam=ENV["lam"],
            mean=ENV["mean"],
            r=ENV["r"],
            shock_var=ENV["shock_var"],
            q_max=ENV["q_max"],
            pv_var=ENV["pv_var"],
        )
        sim.agents = {}

        sim.agents[0] = ZIAgent(
            agent_id=0,
            market=sim.markets[0],
            q_max=ENV["q_max"],
            shade=dev_strat["shade"],
            eta=dev_strat["eta"],
            pv_var=ENV["pv_var"],
        )

        for i in range(1, ENV["n_bg"] + 1):
            sim.agents[i] = ZIAgent(
                agent_id=i,
                market=sim.markets[0],
                q_max=ENV["q_max"],
                shade=ne_strat["shade"],
                eta=ne_strat["eta"],
                pv_var=ENV["pv_var"],
            )

        sim.run()

        fv  = sim.markets[0].get_final_fundamental()
        dev = sim.agents[0]
        dev_profits.append(dev.get_pos_value() + dev.position * fv + dev.cash)

    return dev_idx, float(np.mean(dev_profits)), float(np.std(dev_profits, ddof=1))


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation(ne_idx: int, model_path: str, num_runs: int = DEFAULT_RUNS,
                   n_processes: int = None, base_seed: int = BASE_SEED):
    ne_strat = STRATEGIES[ne_idx]
    n_strats = len(STRATEGIES)

    if n_processes is None:
        n_processes = min(mp.cpu_count(), n_strats + 1)

    print(f"\nPHASE 3: MLP vs NE Strategy Evaluation")
    print(f"  NE strategy    : S{ne_idx}  {ne_strat}")
    print(f"  Model          : {model_path}")
    print(f"  Runs per cell  : {num_runs:,}")
    print(f"  Total sims     : {(n_strats + 1) * num_runs:,}  ({n_strats} ZI + 1 MLP)")
    print(f"  Processes      : {n_processes}")
    print()

    # ── Run all ZI deviators vs NE background ────────────────────────────
    print("Running ZI deviators vs NE background...")
    zi_tasks    = [(ne_idx, dev_idx, num_runs, base_seed) for dev_idx in range(n_strats)]
    zi_means    = np.zeros(n_strats)
    zi_stds     = np.zeros(n_strats)

    with mp.Pool(n_processes) as pool:
        for dev_idx, mean_profit, std_profit in tqdm(
            pool.imap_unordered(_run_zi_vs_ne, zi_tasks),
            total=n_strats,
            desc="ZI cells",
        ):
            zi_means[dev_idx] = mean_profit
            zi_stds[dev_idx]  = std_profit

    # ── Run MLP deviator vs NE background ────────────────────────────────
    print("\nRunning MLP deviator vs NE background...")
    # Run in a single process (model loaded once)
    mlp_mean, mlp_std = _run_mlp_vs_ne((ne_idx, model_path, num_runs, base_seed))

    return zi_means, zi_stds, mlp_mean, mlp_std


def display_results(ne_idx: int, zi_means: np.ndarray, zi_stds: np.ndarray,
                    mlp_mean: float, mlp_std: float, num_runs: int):
    ne_strat = STRATEGIES[ne_idx]
    n        = len(STRATEGIES)
    se_scale = 1.0 / np.sqrt(num_runs)

    print("\n" + "=" * 80)
    print(f"RESULTS: Deviator profit vs S{ne_idx} background  ({ne_strat})")
    print(f"{'Deviator':>14} │ {'Mean Profit':>12} │ {'±SE':>10} │ Best?")
    print("─" * 55)

    best_zi_idx = int(np.argmax(zi_means))
    ne_diag     = zi_means[ne_idx]

    for i in range(n):
        is_best = (i == best_zi_idx)
        is_ne   = (i == ne_idx)
        marker  = " ← best ZI" if is_best else (" (NE diag)" if is_ne else "")
        print(f"  ZI S{i:>2} {STRATEGIES[i]['shade']}  │ {zi_means[i]:>12.1f} │ "
              f"{zi_stds[i]*se_scale:>10.1f} │{marker}")

    mlp_se = mlp_std * se_scale
    best_zi_se = zi_stds[best_zi_idx] * se_scale
    mlp_vs_best = mlp_mean - zi_means[best_zi_idx]
    mlp_vs_ne   = mlp_mean - ne_diag

    print("─" * 55)
    print(f"  {'MLP-SAC':>14}  │ {mlp_mean:>12.1f} │ {mlp_se:>10.1f} │  ← MLP")
    print("=" * 80)

    print(f"\nSUMMARY")
    print(f"  NE diagonal (ZI S{ne_idx} vs ZI S{ne_idx}): {ne_diag:.1f}  ±{zi_stds[ne_idx]*se_scale:.1f}")
    print(f"  Best ZI deviator (S{best_zi_idx}):           {zi_means[best_zi_idx]:.1f}  ±{best_zi_se:.1f}")
    print(f"  MLP deviator:                      {mlp_mean:.1f}  ±{mlp_se:.1f}")
    print(f"  MLP vs NE ZI diagonal:             Δ={mlp_vs_ne:+.1f}")
    print(f"  MLP vs best ZI deviator:           Δ={mlp_vs_best:+.1f}")

    if mlp_vs_best > 2 * mlp_se:
        print(f"\n  ✓ MLP OUTPERFORMS best ZI deviator by {mlp_vs_best:+.1f} (>{2*mlp_se:.1f} = 2×SE)")
    elif mlp_vs_best > 0:
        print(f"\n  ~ MLP slightly above best ZI deviator ({mlp_vs_best:+.1f}) but within 2×SE")
    else:
        print(f"\n  ✗ MLP does not outperform best ZI deviator ({mlp_vs_best:+.1f})")

    if mlp_mean > ne_diag:
        print(f"  ✓ MLP outperforms ZI NE strategy when playing as deviator (Δ={mlp_vs_ne:+.1f})")
    else:
        print(f"  ~ MLP does not outperform ZI NE strategy (Δ={mlp_vs_ne:+.1f})")

    print("=" * 80)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ne-strategy", type=int, required=True,
                   help="Index (0–9) of the NE background strategy")
    p.add_argument("--model", type=str, required=True,
                   help="Path to trained SAC model zip")
    p.add_argument("--num-runs", type=int, default=DEFAULT_RUNS,
                   help=f"Simulation runs (default: {DEFAULT_RUNS})")
    p.add_argument("--processes", type=int, default=None,
                   help="Worker processes (default: cpu_count)")
    p.add_argument("--seed", type=int, default=BASE_SEED,
                   help=f"Base random seed (default: {BASE_SEED})")
    p.add_argument("--output", type=str, default=None,
                   help="Output CSV path (default: eval_mlp_ne_s<idx>.csv)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.ne_strategy not in STRATEGIES:
        raise ValueError(f"--ne-strategy must be 0–9; got {args.ne_strategy}")

    zi_means, zi_stds, mlp_mean, mlp_std = run_evaluation(
        ne_idx=args.ne_strategy,
        model_path=args.model,
        num_runs=args.num_runs,
        n_processes=args.processes,
        base_seed=args.seed,
    )

    display_results(args.ne_strategy, zi_means, zi_stds, mlp_mean, mlp_std, args.num_runs)

    # Save results to CSV
    out = args.output or f"eval_mlp_ne_s{args.ne_strategy}.csv"
    rows = []
    for i in range(len(STRATEGIES)):
        rows.append({
            "deviator": f"ZI-S{i}",
            "mean_profit": zi_means[i],
            "std": zi_stds[i],
            "se": zi_stds[i] / np.sqrt(args.num_runs),
        })
    rows.append({
        "deviator": "MLP-SAC",
        "mean_profit": mlp_mean,
        "std": mlp_std,
        "se": mlp_std / np.sqrt(args.num_runs),
    })
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
