"""
evaluate_vs_paper.py — Compare TRON vs ZI background agents in the same market.

The paper's comparison: TRON (as deviator) earns more than the average ZI agent
playing in the same market episode. We replicate this by reading background agent
profits from the TRONEnv after each episode.

Paper results (N=25 total, T=2000, mixed ZI background):
  Env A (lam=0.0005): ZI = 106,  TRON = 114  (+7.5%)
  Env B (lam=0.005):  ZI = 152,  TRON = 170  (+11.8%)
  Env C (lam=0.012):  ZI = 1259, TRON = 1402 (+11.4%)

Usage
-----
    python evaluate_vs_paper.py --model runs/tron_v9_ne_fast/best_model.pt \\
        --skew-bins --hidden-dim 256 --episodes 200
"""

import argparse
import random
import numpy as np
import torch

from marketsim.agent.tron_agent import TRONPolicy
from marketsim.wrappers.tron_env import TRONEnv

# ── Strategy set ───────────────────────────────────────────────────────────────

_STRATEGIES = {
    0: {'shade': [0,   450],  'eta': 0.5},
    1: {'shade': [0,   600],  'eta': 0.5},
    2: {'shade': [90,  110],  'eta': 0.5},
    3: {'shade': [140, 160],  'eta': 0.5},
    4: {'shade': [190, 210],  'eta': 0.5},
    5: {'shade': [280, 320],  'eta': 0.5},
    6: {'shade': [380, 420],  'eta': 0.5},
    7: {'shade': [380, 420],  'eta': 1.0},
    8: {'shade': [460, 540],  'eta': 0.5},
    9: {'shade': [950, 1050], 'eta': 0.5},
}
BG_STRATEGIES = [{'shade': s['shade'], 'eta': s['eta']} for s in _STRATEGIES.values()]

UNIFORM_SHADE_BINS = np.linspace(0, 600, 42)
SKEWED_SHADE_BINS  = np.concatenate([
    np.linspace(0, 300, 11)[:-1],
    np.linspace(300, 600, 32),
])

# ── Paper benchmarks ───────────────────────────────────────────────────────────

PAPER = {
    'A': {'lam': 0.0005, 'shock_var': 1e6, 'pv_var': 5e6,  'zi': 106,  'tron': 114},
    'B': {'lam': 0.005,  'shock_var': 1e6, 'pv_var': 5e6,  'zi': 152,  'tron': 170},
    'C': {'lam': 0.012,  'shock_var': 2e4, 'pv_var': 2e7,  'zi': 1259, 'tron': 1402},
}


def make_env_kwargs(env_name: str, lam_zi: float, shade_bins: np.ndarray) -> dict:
    p = PAPER[env_name]
    return dict(
        num_background_agents=24,
        sim_time=2000,
        lam=p['lam'],
        lam_zi=lam_zi,
        mean=1e5,
        r=0.01,
        shock_var=p['shock_var'],
        q_max=10,
        pv_var=p['pv_var'],
        normalizers={"fundamental": 1e5, "invt": 10, "reward": 1e3, "pv": 5e5},
        bg_strategies=BG_STRATEGIES,
        warmup_fraction=0.0,
        shade_bins=shade_bins,
    )


def bg_profits(env: TRONEnv) -> np.ndarray:
    """Compute absolute profit for each background ZI agent after episode end.

    Profit = cash + position * final_fundamental + private_value_at_position.
    get_final_fundamental() is safe to call here because end_sim() already
    cached the value in fundamental_values dict; this just returns it.
    """
    final_fund = env.market.get_final_fundamental()
    profits = []
    for agent in env.agents.values():
        val = agent.cash + agent.position * final_fund + agent.get_pos_value()
        profits.append(val)
    return np.array(profits)


def run_episode(policy: TRONPolicy, env: TRONEnv) -> tuple[float, np.ndarray]:
    """Run one episode. Returns (tron_absolute_profit, bg_profits_array)."""
    normalizer = env.normalizers['reward']
    obs, _ = env.reset()
    h = torch.zeros(1, 1, policy.hidden_dim)
    c = torch.zeros(1, 1, policy.hidden_dim)
    ep_reward = 0.0
    done = False

    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            Q_shade, Q_eta, (h, c) = policy(obs_t, (h, c))
        shade_idx = int(Q_shade.argmax(dim=-1).item())
        eta_idx   = int(Q_eta.argmax(dim=-1).item())
        obs, r, terminated, truncated, _ = env.step(np.array([shade_idx, eta_idx]))
        ep_reward += r
        done = terminated or truncated

    tron_profit = ep_reward * normalizer
    zi_profits  = bg_profits(env)
    return tron_profit, zi_profits


def evaluate_env(policy: TRONPolicy, env_name: str, lam_zi: float,
                 shade_bins: np.ndarray, n_episodes: int):
    env = TRONEnv(**make_env_kwargs(env_name, lam_zi, shade_bins))
    tron_ep, zi_mean_ep = [], []

    for _ in range(n_episodes):
        t_profit, zi_arr = run_episode(policy, env)
        tron_ep.append(t_profit)
        zi_mean_ep.append(np.mean(zi_arr))

    return np.array(tron_ep), np.array(zi_mean_ep)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--envs", nargs="+", default=["A", "B", "C"])
    p.add_argument("--skew-bins", action="store_true")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    shade_bins = SKEWED_SHADE_BINS if args.skew_bins else UNIFORM_SHADE_BINS
    bins_desc  = "skewed" if args.skew_bins else "uniform"

    print(f"\nLoading: {args.model}")
    policy = TRONPolicy(input_dim=14, hidden_dim=args.hidden_dim, shade_bins=shade_bins)
    policy.load_state_dict(torch.load(args.model, map_location="cpu"))
    policy.eval()
    print(f"  hidden_dim={args.hidden_dim}, shade_bins={bins_desc}, episodes={args.episodes}\n")

    hdr = (f"{'Env':<4} {'lam':<7} {'TRON':>8} {'±SE':>6}  {'ZI':>8} {'±SE':>6}  "
           f"{'Ratio':>7}  {'Paper ZI':>9} {'Paper TRON':>11} {'Paper ratio':>12}")
    print(hdr)
    print("-" * len(hdr))

    for env_name in args.envs:
        p   = PAPER[env_name]
        lam = p['lam']
        # Fair: lam_zi = lam (equal participation)
        tron_profits, zi_profits = evaluate_env(
            policy, env_name, lam_zi=lam, shade_bins=shade_bins, n_episodes=args.episodes
        )
        n = args.episodes
        tm, ts = np.mean(tron_profits), np.std(tron_profits) / np.sqrt(n)
        zm, zs = np.mean(zi_profits),   np.std(zi_profits)   / np.sqrt(n)
        ratio       = tm / zm if zm != 0 else float('nan')
        paper_ratio = p['tron'] / p['zi']

        print(f"{env_name:<4} {lam:<7} {tm:>+8.1f} {ts:>6.1f}  {zm:>+8.1f} {zs:>6.1f}  "
              f"{ratio:>7.3f}  {p['zi']:>9} {p['tron']:>11} {paper_ratio:>12.3f}")

    print()
    print("TRON and ZI profits are from the SAME episode (TRON as deviator,")
    print("ZI = mean profit of the 24 background agents in that episode).")
    print("Ratio > 1.0 means TRON outearns the average ZI agent.")
    print("Paper target ratio: ~1.118 (env B).")


if __name__ == "__main__":
    main()
