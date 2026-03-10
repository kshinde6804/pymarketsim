"""
train_mlp.py — Train a simple MLP RL agent on ZIEnv using SAC (Stable-Baselines3).

Usage
-----
    # Full training run (500k steps, 4 parallel envs, ~25-40 min)
    python train_mlp.py

    # Quick smoke-test (2k steps, <1 min)
    python train_mlp.py --timesteps 2000 --eval-freq 500 --tag smoke

    # v2 run with parallel envs
    python train_mlp.py --n-envs 4 --timesteps 500000 --tag v2

    # Load a checkpoint and evaluate only (no training)
    python train_mlp.py --eval-only --load runs/sac_zi_YYYYMMDD_HHMMSS/best_model

Outputs (all under runs/<tag>/)
-------------------------------
    best_model.zip        SB3 SAC policy (best eval reward)
    final_model.zip       SB3 SAC policy (end of training)
    monitor.csv           Per-episode rewards during training (Monitor wrapper)
    eval_rewards.csv      Mean ± std episode reward at each eval checkpoint
    learning_curve.png    Plot: training reward + eval checkpoints vs. timesteps
    zi_baseline.txt       ZI baseline stats for comparison
"""

import argparse
import csv
import os
import time
from datetime import datetime
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from marketsim.agent.zero_intelligence_agent import ZIAgent
from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from marketsim.market.market import Market
from marketsim.wrappers.zi_env import ZIEnv

# ── Hyper-parameters ──────────────────────────────────────────────────────────

# All 10 equilibrium background strategies — randomized each episode so the
# agent learns to trade well regardless of which strategy surrounds it.
_STRATEGIES = {
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
BG_STRATEGIES = [
    {'shade': s['shade'], 'eta': s['eta']}
    for s in _STRATEGIES.values()
]

ENV_KWARGS = dict(
    num_background_agents=15,  # match equilibrium n_bg
    sim_time=2000,             # match equilibrium sim_time
    lam=0.005,                 # match equilibrium lam (was 0.1 — 20x reduction)
    lam_zi=0.005,              # match equilibrium lam (was 0.1)
    mean=1e5,
    r=0.01,                    # match equilibrium r (was 0.05 — 5x reduction)
    shock_var=1e6,             # match equilibrium shock_var (was 5e6 — 5x reduction)
    q_max=10,
    pv_var=5e6,
    shade=[250, 500],          # fallback (overridden per episode by bg_strategies)
    shade_range=[0, 600],      # RL agent shade action range
    normalizers={"fundamental": 1e5, "invt": 10, "reward": 1e3, "pv": 5e5},
    # reward normalizer 1e3 (was 1e4): sparser market → ~10x smaller PnL/step
    bg_strategies=BG_STRATEGIES,  # randomize background each episode
    warmup_fraction=0.0,           # no warm-up: sparse market (lam=0.005) causes
                                   # repeated reschedules that push RL arrival past sim_time
)

SAC_KWARGS = dict(
    policy="MlpPolicy",
    learning_rate=1e-4,      # lower LR for more stable updates (3e-4 was too aggressive)
    buffer_size=500_000,     # larger buffer to reduce catastrophic forgetting
    learning_starts=1_000,   # reduced from 5_000: short episodes (~10 steps) fill buffer slowly
    batch_size=256,
    tau=0.005,               # soft target update coefficient
    gamma=0.99,
    train_freq=1,
    gradient_steps=-1,       # one gradient step per env transition collected
    ent_coef="auto",         # automatic entropy tuning
    verbose=0,
)


# ── Environment factory ───────────────────────────────────────────────────────

def make_env(seed: int = 0):
    """Return a Monitor-wrapped ZIEnv (for SB3 episode-reward logging)."""
    def _init():
        env = ZIEnv(**ENV_KWARGS)
        env = Monitor(env)
        return env
    return _init


# ── ZI Baseline evaluation ────────────────────────────────────────────────────

def _run_zi_episode(shade_lo: float, shade_hi: float, eta: float) -> float:
    """Run one full ZIEnv episode with a fixed ZI strategy; return total reward."""
    env = ZIEnv(**ENV_KWARGS)
    obs, _ = env.reset()
    total = 0.0
    while True:
        # Map ZI fixed shade to the action space used by ZIEnv:
        # action[0] = (shade_mean - shade_range[0]) / (shade_range[1] - shade_range[0])
        shade_mid = (shade_lo + shade_hi) / 2.0
        d_min, d_max = ENV_KWARGS["shade_range"]
        shade_norm = np.clip((shade_mid - d_min) / (d_max - d_min), 0.0, 1.0)
        action = np.array([shade_norm, eta], dtype=np.float32)
        obs, r, terminated, truncated, _ = env.step(action)
        total += r
        if terminated or truncated:
            break
    return total


def evaluate_zi_baseline(n_episodes: int = 20) -> dict:
    """Evaluate several canonical ZI strategies and return summary stats."""
    strategies = {
        "ZI-tight  (shade=[90,110],   η=0.5)": (90,  110,  0.5),
        "ZI-medium (shade=[250,350],  η=0.5)": (250, 350,  0.5),
        "ZI-wide   (shade=[380,420],  η=1.0)": (380, 420,  1.0),
    }
    results = {}
    for name, (lo, hi, eta) in strategies.items():
        rewards = [_run_zi_episode(lo, hi, eta) for _ in range(n_episodes)]
        results[name] = {"mean": np.mean(rewards), "std": np.std(rewards)}
        print(f"  {name}  mean={np.mean(rewards):+.3f}  std={np.std(rewards):.3f}")
    return results


# ── Custom callback: log eval rewards to CSV ─────────────────────────────────

class EvalLoggerCallback(BaseCallback):
    """Appends (timestep, mean_reward, std_reward) to a CSV after each eval."""

    def __init__(self, eval_callback: EvalCallback, csv_path: str):
        super().__init__()
        self.eval_callback = eval_callback
        self.csv_path = csv_path
        self._last_logged = 0

    def _on_step(self) -> bool:
        n = self.eval_callback.n_calls
        if n > self._last_logged and self.eval_callback.last_mean_reward is not None:
            self._last_logged = n
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.num_timesteps,
                    f"{self.eval_callback.last_mean_reward:.4f}",
                    f"{self.eval_callback.last_mean_reward:.4f}",  # placeholder std
                ])
        return True


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_learning_curve(
    run_dir: str,
    zi_baseline: dict,
    total_timesteps: int,
):
    """Save a learning-curve PNG with training rewards and eval checkpoints."""
    monitor_path = os.path.join(run_dir, "monitor.csv")
    eval_path = os.path.join(run_dir, "eval_rewards.csv")

    fig, ax = plt.subplots(figsize=(10, 5))

    # ── Training reward (per episode from Monitor) ──
    if os.path.exists(monitor_path):
        with open(monitor_path) as f:
            lines = [l for l in f if not l.startswith("#")]
        reader = csv.DictReader(lines)
        rows = list(reader)
        if rows:
            ts = np.cumsum([float(r["l"]) for r in rows])  # cumulative steps
            rewards = [float(r["r"]) for r in rows]
            # Smooth with a rolling window
            window = max(1, len(rewards) // 20)
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(
                ts[window - 1:], smoothed,
                alpha=0.6, color="steelblue", label="Training reward (smoothed)"
            )
            ax.plot(
                ts, rewards,
                alpha=0.15, color="steelblue", linewidth=0.7
            )

    # ── Eval checkpoints ──
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

    # ── ZI baselines ──
    colors = ["green", "olive", "teal"]
    for (name, stats), color in zip(zi_baseline.items(), colors):
        ax.axhline(stats["mean"], linestyle="--", color=color,
                   linewidth=1.2, alpha=0.8, label=name)
        ax.axhspan(
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            alpha=0.08, color=color
        )

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode reward")
    ax.set_title("SAC MLP Agent — ZIEnv Training")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    out = os.path.join(run_dir, "learning_curve.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=1_000_000,
                   help="Total SAC training steps (default 1 000 000)")
    p.add_argument("--eval-freq", type=int, default=10_000,
                   help="Eval every N training steps (default 10 000)")
    p.add_argument("--eval-episodes", type=int, default=20,
                   help="Episodes per eval round (default 20)")
    p.add_argument("--baseline-episodes", type=int, default=20,
                   help="Episodes for ZI baseline stats (default 20)")
    p.add_argument("--n-envs", type=int, default=4,
                   help="Number of parallel training environments (default 4)")
    p.add_argument("--tag", type=str, default=None,
                   help="Run name tag; defaults to timestamp")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; only evaluate a saved model")
    p.add_argument("--load", type=str, default=None,
                   help="Path to model zip for --eval-only or to resume")
    return p.parse_args()


def main():
    args = parse_args()

    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"sac_zi_{tag}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nRun directory: {run_dir}")

    # ── ZI Baseline ──────────────────────────────────────────────────────
    print("\n[1/4] Evaluating ZI baseline strategies...")
    zi_baseline = evaluate_zi_baseline(n_episodes=args.baseline_episodes)
    zi_path = os.path.join(run_dir, "zi_baseline.txt")
    with open(zi_path, "w") as f:
        for name, stats in zi_baseline.items():
            f.write(f"{name}  mean={stats['mean']:+.4f}  std={stats['std']:.4f}\n")

    if args.eval_only:
        assert args.load, "--eval-only requires --load <model_path>"
        print(f"\n[--eval-only] Loading {args.load}")
        env = Monitor(ZIEnv(**ENV_KWARGS), filename=os.path.join(run_dir, "monitor"))
        model = SAC.load(args.load, env=env)
        rewards = _eval_policy(model, env, n_episodes=args.eval_episodes)
        print(f"Trained agent  mean={np.mean(rewards):+.4f}  std={np.std(rewards):.4f}")
        return

    # ── Training environment ─────────────────────────────────────────────
    print("\n[2/4] Setting up environments...")
    monitor_path = os.path.join(run_dir, "monitor")
    n_envs = args.n_envs

    if n_envs > 1:
        # Parallel training envs: each wrapped with Monitor for episode logging
        train_env = DummyVecEnv([make_env(seed=i) for i in range(n_envs)])
        print(f"  Using {n_envs} parallel training environments (DummyVecEnv)")
    else:
        train_env = Monitor(ZIEnv(**ENV_KWARGS), filename=monitor_path)
        print("  Using single training environment")

    eval_env = DummyVecEnv([make_env(seed=99)])

    # ── Callbacks ────────────────────────────────────────────────────────
    best_model_path = os.path.join(run_dir, "best_model")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.eval_freq * 2, 10_000),
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix="sac_zi",
        verbose=0,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n[3/4] Training SAC agent...")
    print(f"       timesteps     : {args.timesteps:,}")
    print(f"       n_envs        : {n_envs}")
    print(f"       eval_freq     : {args.eval_freq:,}")
    print(f"       eval_episodes : {args.eval_episodes}")
    print(f"       env sim_time  : {ENV_KWARGS['sim_time']:,}")
    print(f"       shade_range   : {ENV_KWARGS['shade_range']}")
    print()

    if args.load:
        print(f"  Resuming from {args.load}")
        model = SAC.load(args.load, env=train_env, **{
            k: v for k, v in SAC_KWARGS.items()
            if k not in ("policy", "verbose")
        })
    else:
        model = SAC(env=train_env, **SAC_KWARGS)

    t0 = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )
    elapsed = time.time() - t0
    print(f"\n  Training finished in {elapsed/60:.1f} min")

    # ── Save final model ──────────────────────────────────────────────────
    final_path = os.path.join(run_dir, "final_model")
    model.save(final_path)
    print(f"  Saved final model → {final_path}.zip")

    # ── Post-training eval ────────────────────────────────────────────────
    print("\n[4/4] Evaluating trained agent...")
    eval_rewards = _eval_policy(
        SAC.load(best_model_path, env=Monitor(ZIEnv(**ENV_KWARGS))),
        Monitor(ZIEnv(**ENV_KWARGS)),
        n_episodes=max(args.eval_episodes * 2, 20),
    )
    print(f"  Trained agent (best)  mean={np.mean(eval_rewards):+.4f}  "
          f"std={np.std(eval_rewards):.4f}")
    print("\n  ZI Baseline comparison:")
    for name, stats in zi_baseline.items():
        delta = np.mean(eval_rewards) - stats["mean"]
        print(f"    vs {name}  Δ={delta:+.4f}")

    # Write eval CSV for plotting
    eval_csv = os.path.join(run_dir, "eval_rewards.csv")
    if os.path.exists(os.path.join(run_dir, "evaluations.npz")):
        data = np.load(os.path.join(run_dir, "evaluations.npz"))
        with open(eval_csv, "w", newline="") as f:
            writer = csv.writer(f)
            for ts, ep_rs in zip(data["timesteps"], data["results"]):
                writer.writerow([int(ts), f"{np.mean(ep_rs):.4f}", f"{np.std(ep_rs):.4f}"])

    # ── Plot ──────────────────────────────────────────────────────────────
    print("\n  Generating learning curve plot...")
    plot_learning_curve(run_dir, zi_baseline, args.timesteps)

    print(f"\nAll outputs in: {run_dir}/")


def _eval_policy(model, env, n_episodes: int = 20) -> List[float]:
    """Roll out the model deterministically for n_episodes; return reward list."""
    rewards = []
    obs, _ = env.reset()
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


if __name__ == "__main__":
    main()
