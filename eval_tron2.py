"""
eval_tron2.py — Evaluate a trained TRON2 model against the ZI baseline.

Reports:
  - TRON2 mean ± std profit over --num-runs episodes
  - ZI baseline mean ± std profit (all 25 agents play ne_strategy)
  - Advantage = TRON2 - ZI, with Welch t-test p-value

Usage:
    python eval_tron2.py --model runs/tron2_v1/best_model.pt --env-tag B
    python eval_tron2.py --model runs/tron2_v1/best_model.pt --env-tag C --num-runs 500
"""

import argparse
import random

import numpy as np
import torch
from scipy import stats

from marketsim.agent.tron2_agent import TRONPolicy2
from marketsim.agent.zero_intelligence_agent import ZIAgent
from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from marketsim.market.market import Market
from marketsim.wrappers.tron2_env import TRONEnv2, ENV_CONFIGS, STRATEGIES

# Prevent torch fork deadlock when used alongside multiprocessing
torch.set_num_threads(1)

HIDDEN_DIM = 128


# ── TRON2 evaluation ───────────────────────────────────────────────────────────

def run_tron2_episodes(policy, env_tag, ne_strategy, n_runs, device):
    """Run n_runs greedy episodes with the trained TRON2 policy.

    Returns array of raw (unnormalised) profits.
    """
    env = TRONEnv2(env_tag=env_tag, ne_strategy=ne_strategy)
    policy.eval()
    profits = []

    for _ in range(n_runs):
        obs, _ = env.reset()
        h_c    = None
        done   = False

        while not done:
            obs_t = (
                torch.tensor(obs, dtype=torch.float32)
                .unsqueeze(0).unsqueeze(0).to(device)
            )
            with torch.no_grad():
                Q_shade, Q_eta, h_c = policy(obs_t, h_c)
            shade_idx = int(Q_shade.squeeze().argmax().item())
            eta_idx   = int(Q_eta.squeeze().argmax().item())
            obs, _, terminated, truncated, _ = env.step([shade_idx, eta_idx])
            done = terminated or truncated

        profits.append(env.get_episode_profit())

    return np.array(profits)


# ── ZI baseline evaluation ─────────────────────────────────────────────────────

def run_zi_baseline_episodes(env_tag, ne_strategy, n_runs):
    """Run n_runs simulations with 25 ZI agents all playing strategy ne_strategy.

    Returns array of profits for agent 0 (the 'test' ZI agent).
    """
    cfg   = ENV_CONFIGS[env_tag]
    strat = STRATEGIES[ne_strategy]
    lam       = cfg["lam"]
    shock_var = cfg["shock_var"]
    pv_var    = cfg["pv_var"]
    mean      = 1e5
    r         = 0.01
    q_max     = 10
    sim_time  = 2000
    n_agents  = 25   # 25 total ZI agents (paper §5.2)

    profits = []

    for _ in range(n_runs):
        fundamental = LazyGaussianMeanReverting(
            mean=mean, final_time=sim_time + 1, r=r, shock_var=shock_var
        )
        fundamental.prefetch_all()
        final_f = fundamental.get_final_fundamental()
        market  = Market(fundamental=fundamental, time_steps=sim_time)

        agents = {
            i: ZIAgent(
                agent_id=i,
                market=market,
                q_max=q_max,
                shade=strat["shade"],
                pv_var=pv_var,
                eta=strat["eta"],
            )
            for i in range(n_agents)
        }

        # Simple geometric arrival simulation (same model as TRONEnv2)
        import torch.distributions as tdist

        arrivals: dict = {}
        arr_times = tdist.Geometric(torch.tensor([lam])).sample((10000,)).squeeze()
        idx = 0
        for agent_id in range(n_agents):
            t = int(arr_times[idx].item())
            arrivals.setdefault(t, []).append(agent_id)
            idx += 1

        def schedule_next(agent_id, current_t):
            nonlocal idx
            if idx >= len(arr_times):
                arr_times2 = tdist.Geometric(torch.tensor([lam])).sample((10000,)).squeeze()
                for k in range(len(arr_times)):
                    arr_times[k] = arr_times2[k]
                idx = 0
            gap = int(arr_times[idx].item())
            idx += 1
            arrivals.setdefault(current_t + 1 + gap, []).append(agent_id)

        for t in range(sim_time):
            if t in arrivals:
                market.event_queue.set_time(t)
                for agent_id in arrivals[t]:
                    agent = agents[agent_id]
                    market.withdraw_all(agent_id)
                    market.add_orders(agent.take_action())
                    schedule_next(agent_id, t)
            market.event_queue.set_time(t)
            new_orders = market.step()
            for matched in new_orders:
                aid  = matched.order.agent_id
                qty  = matched.order.order_type * matched.order.quantity
                cash = -matched.price * matched.order.quantity * matched.order.order_type
                agents[aid].update_position(qty, cash)

        profit0 = (
            agents[0].position * final_f
            + agents[0].cash
            + agents[0].get_pos_value()
        )
        profits.append(profit0)

    return np.array(profits)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load TRON2 policy
    policy = TRONPolicy2(input_dim=14, hidden_dim=HIDDEN_DIM).to(device)
    policy.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))

    print(f"Model:       {args.model}")
    print(f"Env:         {args.env_tag}  |  strategy: S{args.ne_strategy}")
    print(f"Runs:        {args.num_runs}\n")

    print("Running TRON2 episodes …")
    tron_profits = run_tron2_episodes(policy, args.env_tag, args.ne_strategy,
                                      args.num_runs, device)

    print("Running ZI baseline episodes …")
    zi_profits = run_zi_baseline_episodes(args.env_tag, args.ne_strategy, args.num_runs)

    tron_mean, tron_std = tron_profits.mean(), tron_profits.std()
    zi_mean,   zi_std   = zi_profits.mean(),   zi_profits.std()
    advantage           = tron_mean - zi_mean

    t_stat, p_val = stats.ttest_ind(tron_profits, zi_profits, equal_var=False)

    pct_gain = 100.0 * advantage / abs(zi_mean) if zi_mean != 0 else float("nan")

    print("─" * 55)
    print(f"TRON2 profit:    {tron_mean:>10.2f}  ±  {tron_std:.2f}")
    print(f"ZI baseline:     {zi_mean:>10.2f}  ±  {zi_std:.2f}")
    print(f"Advantage:       {advantage:>10.2f}  ({pct_gain:+.1f}%)")
    print(f"t-test p-value:  {p_val:.4f}")
    print("─" * 55)

    # Paper targets for reference
    targets = {"A": (114, 106), "B": (170, 152), "C": (1402, 1259)}
    if args.env_tag in targets:
        tgt_tron, tgt_zi = targets[args.env_tag]
        print(f"\nPaper Table 3 target (Env {args.env_tag}):")
        print(f"  TRON={tgt_tron},  ZI={tgt_zi},  gain={100*(tgt_tron-tgt_zi)/tgt_zi:.0f}%")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate TRON2 vs ZI baseline")
    p.add_argument("--model",        required=True,
                   help="Path to saved TRONPolicy2 .pt weights")
    p.add_argument("--env-tag",      default="B", choices=["A", "B", "C"])
    p.add_argument("--ne-strategy",  default=8,   type=int,
                   help="Background ZI strategy index 0–9")
    p.add_argument("--num-runs",     default=500, type=int)
    p.add_argument("--seed",         default=0,   type=int)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
