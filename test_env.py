"""
test_env.py — Smoke-test and integration demo for ZIEnv and MLPAgent.

Sections
--------
1. ZIEnv random-action loop
   Verifies the Gymnasium API (reset/step shapes, dtypes, no NaNs).

2. MLPAgent standalone simulation
   Plugs MLPAgent directly into a raw market simulation (no gym wrapper).

3. Observation consistency check
   Confirms ZIEnv.get_obs() and MLPAgent.build_obs() produce identical
   13-feature vectors when they share the same market and agent state.

Run
---
    python test_env.py
"""

import random
from collections import defaultdict

import numpy as np
import torch
import torch.distributions as dist

from marketsim.agent.mlp_agent import MLPAgent
from marketsim.agent.zero_intelligence_agent import ZIAgent
from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from marketsim.market.market import Market
from marketsim.wrappers.zi_env import ZIEnv

# ── Shared defaults ──────────────────────────────────────────────────────────

NORMALIZERS = {"fundamental": 1e5, "invt": 10, "reward": 1e4, "pv": 5e5}

ENV_KWARGS = dict(
    num_background_agents=10,
    sim_time=500,
    lam=0.1,
    lam_zi=0.1,
    mean=1e5,
    r=0.05,
    shock_var=5e6,
    q_max=10,
    pv_var=5e6,
    shade=[250, 500],
    shade_range=[10, 500],
    normalizers=NORMALIZERS,
)


# ── Section 1 ─────────────────────────────────────────────────────────────────

def test_zienv_random_loop():
    print("=" * 60)
    print("Section 1: ZIEnv random-action loop")

    env = ZIEnv(**ENV_KWARGS)

    obs, info = env.reset()
    assert obs.shape == (14,), f"Expected obs shape (14,), got {obs.shape}"
    assert obs.dtype == np.float64, f"Expected float64, got {obs.dtype}"
    print(f"  obs shape:  {obs.shape}")
    print(f"  obs sample: {np.round(obs, 4)}")

    total_reward = 0.0
    n_steps = 0
    n_episodes = 0

    for global_step in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        n_steps += 1

        assert not np.any(np.isnan(obs)), f"NaN in obs at step {global_step}"
        assert not np.isnan(reward), f"NaN reward at step {global_step}"

        if terminated or truncated:
            n_episodes += 1
            print(
                f"  Episode {n_episodes} ended at global step {global_step} | "
                f"steps={n_steps}  total_reward={total_reward:.4f}"
            )
            obs, info = env.reset()
            total_reward = 0.0
            n_steps = 0

    print("PASSED: Section 1\n")


# ── Section 2 ─────────────────────────────────────────────────────────────────

def _sample_arrivals(p, n):
    geo = dist.Geometric(torch.tensor([p]))
    return geo.sample((n,)).squeeze()


def test_mlp_agent_standalone():
    print("=" * 60)
    print("Section 2: MLPAgent standalone simulation")

    sim_time = 300
    q_max = 10
    pv_var = 5e6
    mean = 1e5
    lam = 0.1

    fundamental = LazyGaussianMeanReverting(
        mean=mean, final_time=sim_time + 1, r=0.05, shock_var=5e6
    )
    market = Market(fundamental=fundamental, time_steps=sim_time)

    # Mix 5 background ZI agents with 1 MLP agent
    agents = {}
    for i in range(5):
        agents[i] = ZIAgent(
            agent_id=i, market=market, q_max=q_max, shade=[250, 500], pv_var=pv_var
        )

    mlp_agent = MLPAgent(
        agent_id=5,
        market=market,
        q_max=q_max,
        pv_var=pv_var,
        shade=[250, 500],
        shade_range=[10, 500],
        normalizers=NORMALIZERS,
    )
    agents[5] = mlp_agent
    num_agents = 6

    # Set up Poisson arrivals
    arrivals_sampled = 2000
    arrival_times = _sample_arrivals(lam, arrivals_sampled)
    idx = [0]  # mutable counter

    arrivals = defaultdict(list)
    for aid in range(num_agents):
        arrivals[arrival_times[idx[0]].item()].append(aid)
        idx[0] += 1

    def _next_arrival(t):
        if idx[0] >= arrivals_sampled:
            arrival_times[:] = _sample_arrivals(lam, arrivals_sampled)
            idx[0] = 0
        gap = arrival_times[idx[0]].item() + 1 + t
        idx[0] += 1
        return gap

    # Manual simulation loop
    for t in range(sim_time):
        market.event_queue.set_time(t)
        for aid in arrivals[t]:
            agent = agents[aid]
            market.withdraw_all(aid)
            orders = agent.take_action()
            market.add_orders(orders)
            arrivals[_next_arrival(t)].append(aid)

        new_orders = market.step()
        for matched_order in new_orders:
            aid = matched_order.order.agent_id
            qty = matched_order.order.order_type * matched_order.order.quantity
            cash = (
                -matched_order.price
                * matched_order.order.quantity
                * matched_order.order.order_type
            )
            agents[aid].update_position(qty, cash)

    final_fund = market.get_final_fundamental()
    mlp_value = (
        mlp_agent.position * final_fund
        + mlp_agent.cash
        + mlp_agent.get_pos_value()
    )
    print(
        f"  MLPAgent final value : {mlp_value:>10.2f}"
        f"  |  position: {mlp_agent.position:>3d}"
        f"  |  cash: {mlp_agent.cash:>10.2f}"
    )
    print("PASSED: Section 2\n")


# ── Section 3 ─────────────────────────────────────────────────────────────────

def test_obs_consistency():
    print("=" * 60)
    print("Section 3: Observation consistency (ZIEnv vs MLPAgent.build_obs)")

    env = ZIEnv(**ENV_KWARGS)

    # Build an MLPAgent that shares env.market (same underlying state)
    mlp = MLPAgent(
        agent_id=env.zi_agent_id,
        market=env.market,
        q_max=ENV_KWARGS["q_max"],
        pv_var=ENV_KWARGS["pv_var"],
        shade=ENV_KWARGS["shade"],
        shade_range=ENV_KWARGS["shade_range"],
        normalizers=NORMALIZERS,
    )

    # Hot-swap: replace env's zi_agent with mlp so reset() initialises mlp too
    env.zi_agent = mlp

    obs_env, _ = env.reset()

    # After reset, env.zi_agent IS mlp, so build_obs() uses identical state
    # Pass env.current_side so the side feature matches between ZIEnv and MLPAgent
    obs_agent = mlp.build_obs(env.current_side)

    diff = np.abs(obs_env - obs_agent)
    print(f"  ZIEnv obs    : {np.round(obs_env,   4)}")
    print(f"  MLPAgent obs : {np.round(obs_agent, 4)}")
    print(f"  Max abs diff : {diff.max():.8f}")

    assert diff.max() < 1e-6, (
        f"Feature mismatch between ZIEnv and MLPAgent!\n"
        f"  ZIEnv:    {obs_env}\n"
        f"  MLPAgent: {obs_agent}\n"
        f"  diff:     {diff}"
    )
    print("PASSED: Section 3\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    test_zienv_random_loop()
    test_mlp_agent_standalone()
    test_obs_consistency()

    print("=" * 60)
    print("All tests PASSED.")
    print()
    print("Next steps:")
    print("  # SB3 env check:")
    print("  from stable_baselines3.common.env_checker import check_env")
    print("  from marketsim.wrappers.zi_env import ZIEnv")
    print("  check_env(ZIEnv(num_background_agents=5, sim_time=300, mean=1e5,")
    print("                  normalizers={'fundamental':1e5,'invt':10,'reward':1e4,'pv':5e5}))")
    print()
    print("  # Quick SAC training:")
    print("  from stable_baselines3 import SAC")
    print("  model = SAC('MlpPolicy', env, verbose=1)")
    print("  model.learn(total_timesteps=50000)")
