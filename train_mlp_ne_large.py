"""
train_mlp_ne_large.py — Train MLP agent against a fixed NE background strategy.

Large-scale version with 24 background agents, 3M training steps, and seed=42.

Usage
-----
    # After finding NE candidate S0 from run_large_zi_zi.py (24-bg run):
    python train_mlp_ne_large.py --ne-strategy 0 --timesteps 3000000 --tag v1

    # With custom timesteps and tag:
    python train_mlp_ne_large.py --ne-strategy 0 --timesteps 1000000 --tag smoke

    # Eval only:
    python train_mlp_ne_large.py --ne-strategy 0 --eval-only --load runs/sac_zi_ne_large_s0_v1/best_model

Outputs (under runs/sac_zi_ne_large_s<idx>_<tag>/)
---------------------------------------------------
    best_model.zip      SB3 SAC policy (best eval reward)
    final_model.zip     final checkpoint
    checkpoints/        periodic saves every 150k steps
    monitor.csv         per-episode training rewards
    eval_rewards.csv    eval checkpoints
    learning_curve.png  plot
    zi_baseline.txt     ZI baseline comparison
"""

import argparse
import csv
import os
import random
import time
from datetime import datetime
from typing import List

import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from marketsim.wrappers.zi_env import ZIEnv


# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

BASE_SEED = 42  # default; overridden by --seed

# ---------------------------------------------------------------------------
# All 10 strategies (for ZI baseline comparison)
# ---------------------------------------------------------------------------

ALL_STRATEGIES = {
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

# ---------------------------------------------------------------------------
# Base environment kwargs (24 background agents to match equilibrium experiment)
# ---------------------------------------------------------------------------

BASE_ENV_KWARGS = dict(
    num_background_agents=24,
    sim_time=2000,
    lam=0.005,
    lam_zi=0.005,
    mean=1e5,
    r=0.01,
    shock_var=1e6,
    q_max=10,
    pv_var=5e6,
    shade=[250, 500],          # fallback; overridden each episode
    shade_range=[0, 600],
    normalizers={"fundamental": 1e5, "invt": 10, "reward": 1e3, "pv": 5e5},
    warmup_fraction=0.0,
)

SAC_KWARGS = dict(
    policy="MlpPolicy",
    learning_rate=1e-4,
    buffer_size=500_000,
    learning_starts=1_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=-1,
    ent_coef="auto",
    verbose=0,
)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(ne_strategy: dict, seed: int = 0):
    """Return a Monitor-wrapped ZIEnv with fixed NE background strategy."""
    ne_bg = [ne_strategy]   # single-element list → always this strategy
    env_kwargs = {**BASE_ENV_KWARGS, "bg_strategies": ne_bg}

    def _init():
        env = ZIEnv(**env_kwargs)
        env = Monitor(env)
        return env
    return _init


# ---------------------------------------------------------------------------
# ZI Baseline evaluation
# ---------------------------------------------------------------------------

def _run_zi_episode(shade_lo: float, shade_hi: float, eta: float, ne_strategy: dict) -> float:
    ne_bg = [ne_strategy]
    env_kwargs = {**BASE_ENV_KWARGS, "bg_strategies": ne_bg}
    env = ZIEnv(**env_kwargs)
    obs, _ = env.reset()
    total = 0.0
    while True:
        shade_mid  = (shade_lo + shade_hi) / 2.0
        d_min, d_max = BASE_ENV_KWARGS["shade_range"]
        shade_norm = np.clip((shade_mid - d_min) / (d_max - d_min), 0.0, 1.0)
        action     = np.array([shade_norm, eta], dtype=np.float32)
        obs, r, terminated, truncated, _ = env.step(action)
        total += r
        if terminated or truncated:
            break
    return total


def evaluate_zi_baseline(ne_strategy: dict, n_episodes: int = 20) -> dict:
    """Evaluate several ZI strategies in the NE environment."""
    s = ne_strategy
    strategies = {
        f"ZI-NE     (shade={s['shade']}, η={s['eta']})":
            (s['shade'][0], s['shade'][1], s['eta']),
        "ZI-tight  (shade=[90,110],   η=0.5)": (90,  110,  0.5),
        "ZI-medium (shade=[250,350],  η=0.5)": (250, 350,  0.5),
        "ZI-wide   (shade=[380,420],  η=1.0)": (380, 420,  1.0),
    }
    results = {}
    for name, (lo, hi, eta) in strategies.items():
        rewards = [_run_zi_episode(lo, hi, eta, ne_strategy) for _ in range(n_episodes)]
        results[name] = {"mean": np.mean(rewards), "std": np.std(rewards)}
        print(f"  {name}  mean={np.mean(rewards):+.3f}  std={np.std(rewards):.3f}")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_learning_curve(run_dir: str, zi_baseline: dict, total_timesteps: int):
    monitor_path = os.path.join(run_dir, "monitor.csv")
    eval_path    = os.path.join(run_dir, "eval_rewards.csv")

    fig, ax = plt.subplots(figsize=(10, 5))

    if os.path.exists(monitor_path):
        with open(monitor_path) as f:
            lines = [l for l in f if not l.startswith("#")]
        reader = csv.DictReader(lines)
        rows   = list(reader)
        if rows:
            ts       = np.cumsum([float(r["l"]) for r in rows])
            rewards  = [float(r["r"]) for r in rows]
            window   = max(1, len(rewards) // 20)
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(ts[window - 1:], smoothed, alpha=0.6, color="steelblue",
                    label="Training reward (smoothed)")
            ax.plot(ts, rewards, alpha=0.15, color="steelblue", linewidth=0.7)

    if os.path.exists(eval_path):
        eval_ts, eval_means = [], []
        with open(eval_path) as f:
            for row in csv.reader(f):
                if row:
                    eval_ts.append(int(row[0]))
                    eval_means.append(float(row[1]))
        if eval_ts:
            ax.plot(eval_ts, eval_means, "o-", color="darkorange",
                    linewidth=2, markersize=4, label="Eval mean reward", zorder=5)

    colors = ["red", "green", "olive", "teal"]
    for (name, stats), color in zip(zi_baseline.items(), colors):
        ax.axhline(stats["mean"], linestyle="--", color=color,
                   linewidth=1.2, alpha=0.8, label=name)
        ax.axhspan(stats["mean"] - stats["std"], stats["mean"] + stats["std"],
                   alpha=0.08, color=color)

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode reward")
    ax.set_title("SAC MLP Agent — ZIEnv Training vs NE Strategy (24 BG, 3M steps)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    out = os.path.join(run_dir, "learning_curve.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Eval helper
# ---------------------------------------------------------------------------

def _eval_policy(model, env, n_episodes: int = 20) -> List[float]:
    rewards   = []
    obs, _    = env.reset()
    ep_reward = 0.0
    while len(rewards) < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, _ = env.step(action)
        ep_reward += r
        if terminated or truncated:
            rewards.append(ep_reward)
            ep_reward = 0.0
            if len(rewards) < n_episodes:
                obs, _ = env.reset()
    return rewards


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ne-strategy", type=int, required=True,
                   help="Index (0–9) of the NE background strategy to use")
    p.add_argument("--timesteps", type=int, default=3_000_000,
                   help="Total SAC training steps (default 3,000,000)")
    p.add_argument("--eval-episodes", type=int, default=20,
                   help="Episodes per eval round (default 20)")
    p.add_argument("--baseline-episodes", type=int, default=20,
                   help="Episodes for ZI baseline stats (default 20)")
    p.add_argument("--n-envs", type=int, default=4,
                   help="Number of parallel training environments (default 4)")
    p.add_argument("--tag", type=str, default=None,
                   help="Run name tag; defaults to timestamp")
    p.add_argument("--seed", type=int, default=BASE_SEED,
                   help=f"Global random seed (default: {BASE_SEED})")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; only evaluate a saved model")
    p.add_argument("--load", type=str, default=None,
                   help="Path to model zip for --eval-only or to resume training")
    return p.parse_args()


def main():
    # Limit PyTorch threads — small MLPs run faster with 2 threads than 36
    # (36-thread default causes overhead that dominates for 256-dim matmuls)
    torch.set_num_threads(2)

    args = parse_args()

    seed = args.seed
    # Global seeding for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.ne_strategy not in ALL_STRATEGIES:
        raise ValueError(f"--ne-strategy must be 0–9; got {args.ne_strategy}")

    ne_strategy = ALL_STRATEGIES[args.ne_strategy]
    print(f"\nNE background strategy: S{args.ne_strategy}  {ne_strategy}")
    print(f"Background agents     : {BASE_ENV_KWARGS['num_background_agents']}")
    print(f"Base seed             : {seed}")

    tag     = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"sac_zi_ne_large_s{args.ne_strategy}_{tag}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # ── ZI Baseline ────────────────────────────────────────────────────────
    print("\n[1/4] Evaluating ZI baseline strategies in NE environment...")
    zi_baseline = evaluate_zi_baseline(ne_strategy, n_episodes=args.baseline_episodes)
    zi_path = os.path.join(run_dir, "zi_baseline.txt")
    with open(zi_path, "w") as f:
        f.write(f"NE background strategy: S{args.ne_strategy}  {ne_strategy}\n")
        f.write(f"Background agents: {BASE_ENV_KWARGS['num_background_agents']}\n\n")
        for name, stats in zi_baseline.items():
            f.write(f"{name}  mean={stats['mean']:+.4f}  std={stats['std']:.4f}\n")

    if args.eval_only:
        assert args.load, "--eval-only requires --load <model_path>"
        print(f"\n[--eval-only] Loading {args.load}")
        ne_bg = [ne_strategy]
        env_kwargs = {**BASE_ENV_KWARGS, "bg_strategies": ne_bg}
        env = Monitor(ZIEnv(**env_kwargs))
        model = SAC.load(args.load, env=env)
        rewards = _eval_policy(model, env, n_episodes=args.eval_episodes)
        print(f"Trained agent  mean={np.mean(rewards):+.4f}  std={np.std(rewards):.4f}")
        return

    # ── Training environment ───────────────────────────────────────────────
    print("\n[2/4] Setting up environments...")
    n_envs = args.n_envs

    if n_envs > 1:
        train_env = DummyVecEnv([make_env(ne_strategy, seed=i) for i in range(n_envs)])
        print(f"  Using {n_envs} parallel training environments (DummyVecEnv)")
    else:
        ne_bg = [ne_strategy]
        env_kwargs = {**BASE_ENV_KWARGS, "bg_strategies": ne_bg}
        train_env = Monitor(ZIEnv(**env_kwargs))
        print("  Using single training environment")

    eval_env = DummyVecEnv([make_env(ne_strategy, seed=99)])

    # ── Callbacks — scale frequencies with total timesteps ─────────────────
    # ~20 eval points and ~5 checkpoints regardless of run length
    eval_freq = max(args.timesteps // (20 * n_envs), 1)
    ckpt_freq = max(args.timesteps // (5 * n_envs), 1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=ckpt_freq,
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix="sac_step",
        verbose=0,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    print("\n[3/4] Training SAC agent...")
    print(f"       ne_strategy   : S{args.ne_strategy}  {ne_strategy}")
    print(f"       timesteps     : {args.timesteps:,}")
    print(f"       n_envs        : {n_envs}")
    print(f"       eval_freq     : {eval_freq * n_envs:,} steps  (~20 evals)")
    print(f"       checkpoint    : {ckpt_freq * n_envs:,} steps  (~5 checkpoints)")
    print()

    if args.load:
        print(f"  Resuming from {args.load}")
        model = SAC.load(args.load, env=train_env, **{
            k: v for k, v in SAC_KWARGS.items()
            if k not in ("policy", "verbose")
        })
    else:
        model = SAC(env=train_env, seed=seed, **SAC_KWARGS)

    t0 = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=False,
    )
    elapsed = time.time() - t0
    print(f"\n  Training finished in {elapsed/60:.1f} min")

    final_path = os.path.join(run_dir, "final_model")
    model.save(final_path)
    print(f"  Saved final model → {final_path}.zip")

    # ── Post-training eval ─────────────────────────────────────────────────
    print("\n[4/4] Evaluating trained agent...")
    ne_bg = [ne_strategy]
    env_kwargs = {**BASE_ENV_KWARGS, "bg_strategies": ne_bg}
    best_model_path = os.path.join(run_dir, "best_model")
    # Fall back to final model if eval callback never saved best_model
    load_path = best_model_path if os.path.exists(best_model_path + ".zip") else final_path
    eval_rewards = _eval_policy(
        SAC.load(load_path, env=Monitor(ZIEnv(**env_kwargs))),
        Monitor(ZIEnv(**env_kwargs)),
        n_episodes=max(args.eval_episodes * 2, 20),
    )
    print(f"  Trained agent (best)  mean={np.mean(eval_rewards):+.4f}  "
          f"std={np.std(eval_rewards):.4f}")
    print("\n  ZI Baseline comparison:")
    for name, stats in zi_baseline.items():
        delta = np.mean(eval_rewards) - stats["mean"]
        print(f"    vs {name}  Δ={delta:+.4f}")

    # Write eval CSV
    eval_csv = os.path.join(run_dir, "eval_rewards.csv")
    evaluations_npz = os.path.join(run_dir, "evaluations.npz")
    if os.path.exists(evaluations_npz):
        data = np.load(evaluations_npz)
        with open(eval_csv, "w", newline="") as f:
            writer = csv.writer(f)
            for ts, ep_rs in zip(data["timesteps"], data["results"]):
                writer.writerow([int(ts), f"{np.mean(ep_rs):.4f}", f"{np.std(ep_rs):.4f}"])

    print("\n  Generating learning curve plot...")
    plot_learning_curve(run_dir, zi_baseline, args.timesteps)

    print(f"\nAll outputs in: {run_dir}/")
    print(f"\nNext step — evaluate MLP vs 24 NE background agents:")
    print(f"  python eval_mlp_ne_large.py --ne-strategy {args.ne_strategy} "
          f"--model {best_model_path} --num-runs 2000 "
          f"--output eval_mlp_ne_large_s{args.ne_strategy}_results.csv")


if __name__ == "__main__":
    main()
