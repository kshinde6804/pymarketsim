"""
train_tronformer.py — Train the TRONformer agent on TRONEnv using R2D2.

R2D2 (Recurrent Replay Distributed DQN) adapted for a transformer backbone:
  - Dueling DQN architecture (TRONformerPolicy — Pre-LN transformer)
  - Rolling obs buffer stored in replay sequences (no LSTM h/c state)
  - Fixed-length sequence replay (seq_len=32, burnin=4)
  - n-step returns (n=5)
  - Double DQN target computation
  - Epsilon-greedy exploration with linear decay
  - Huber loss on both shade and eta streams
  - Learned reward model: binary classifier for order execution probability
    with shaped reward γ decay (0.8 → 0) over training

Usage
-----
    # Default 2M-step run
    caffeinate -disu python -u train_tronformer.py --tag v1

    # Quick smoke test (200 steps)
    python train_tronformer.py --timesteps 200 --tag smoke

    # Evaluate a saved model
    python train_tronformer.py --eval-only --load runs/tronformer_v1/best_model.pt

Outputs (all under runs/tronformer_<tag>/)
-------------------------------------------
    best_model.pt       TRONformerPolicy state_dict (best eval reward)
    final_model.pt      TRONformerPolicy state_dict (end of training)
    eval_rewards.csv    (timestep, mean_reward, std_reward) at each eval
    learning_curve.png  Eval reward vs. timesteps
"""

import argparse
import collections
import csv
import os
import random
import time
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from marketsim.agent.tronformer_agent import TRONformerPolicy, SEQ_LEN
from marketsim.wrappers.tron_env import TRONEnv

# ── Shade bin options ──────────────────────────────────────────────────────────

UNIFORM_SHADE_BINS = np.linspace(0, 600, 42)

SKEWED_SHADE_BINS = np.concatenate([
    np.linspace(0, 300, 11)[:-1],
    np.linspace(300, 600, 32),
])

# ── ENV hyper-parameters ──────────────────────────────────────────────────────

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
    num_background_agents=24,
    sim_time=2000,
    lam=0.005,
    lam_zi=0.02,
    mean=1e5,
    r=0.01,
    shock_var=1e6,
    q_max=10,
    pv_var=5e6,
    shade=[250, 500],
    normalizers={"fundamental": 1e5, "invt": 10, "reward": 1e4, "pv": 5e5},
    bg_strategies=BG_STRATEGIES,
    warmup_fraction=0.0,
    shade_bins=UNIFORM_SHADE_BINS,
)

# ── R2D2 hyper-parameters ─────────────────────────────────────────────────────

BURNIN = 8
N_STEP = 5
GAMMA = 0.99
BATCH_SIZE = 128
BUFFER_CAPACITY = 100_000
LR = 1e-4
TARGET_UPDATE_FREQ = 1000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 1_000_000
GRAD_CLIP = 10.0
LEARNING_STARTS = 100
TRAIN_FREQ = 4
EVAL_FREQ = 50_000
EVAL_EPISODES = 100

# ── Reward model hyper-parameters ─────────────────────────────────────────────
# GAMMA_RM_START=0.0 disables shaped rewards — the reward model never converged
# (rm_loss ≈ 0.52-0.57 throughout training, i.e. random chance), and the
# unconverged classifier systematically biases the policy toward high-shade actions.

RM_INPUT_DIM = 15        # 14 obs + 1 shade_norm
RM_BUFFER_SIZE = 10_000
RM_TRAIN_FREQ = 500      # Train reward model every N env steps
RM_BATCH_SIZE = 256
RM_LR = 1e-3
GAMMA_RM_START = 0.0     # Shaped reward weight at step 0 (0.0 = disabled)
GAMMA_RM_END = 0.0       # Shaped reward weight at total_steps


# ── Reward Model ───────────────────────────────────────────────────────────────


class RewardModel(nn.Module):
    """Binary classifier predicting P(order executes before agent next acts).

    Features: 14-dim observation concatenated with 1 shade feature (shade_idx
    normalised to [0, 1]) = 15-dim input.

    Args:
        input_dim: Feature dimension (default 15 = 14 obs + 1 shade_norm).
    """

    def __init__(self, input_dim: int = RM_INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)

        Returns:
            exec_prob: (batch,) in [0, 1]
        """
        return self.net(x).squeeze(-1)


# ── Replay Buffer ──────────────────────────────────────────────────────────────


class TronformerReplayBuffer:
    """Fixed-length sequence replay buffer for TRONformer (no LSTM state).

    Each stored entry is a complete sequence of length seq_len:
        obs_seq:  (seq_len, obs_dim)   float32
        act_seq:  (seq_len, 2)         int64   [shade_idx, eta_idx]
        rew_seq:  (seq_len,)           float32
        done_seq: (seq_len,)           bool

    Args:
        capacity: Maximum number of sequences stored.
        seq_len:  Length of each stored sequence.
        obs_dim:  Observation dimensionality (14 for TRONformer).
    """

    def __init__(
        self,
        capacity: int = BUFFER_CAPACITY,
        seq_len: int = SEQ_LEN,
        obs_dim: int = 14,
    ):
        self.capacity = capacity
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self._buf: list = []
        self._pos: int = 0

    def __len__(self) -> int:
        return len(self._buf)

    def add(
        self,
        obs_seq: np.ndarray,
        act_seq: np.ndarray,
        rew_seq: np.ndarray,
        done_seq: np.ndarray,
    ):
        entry = (
            obs_seq.astype(np.float32),
            act_seq.astype(np.int64),
            rew_seq.astype(np.float32),
            done_seq.astype(bool),
        )
        if len(self._buf) < self.capacity:
            self._buf.append(entry)
        else:
            self._buf[self._pos] = entry
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int):
        """Sample a batch of sequences uniformly at random.

        Returns:
            obs:   (batch, seq_len, obs_dim)  float32 tensor
            acts:  (batch, seq_len, 2)        int64 tensor
            rews:  (batch, seq_len)           float32 tensor
            dones: (batch, seq_len)           bool tensor
        """
        indices = random.sample(range(len(self._buf)), batch_size)
        batch = [self._buf[i] for i in indices]

        obs   = torch.tensor(np.stack([b[0] for b in batch]))
        acts  = torch.tensor(np.stack([b[1] for b in batch]))
        rews  = torch.tensor(np.stack([b[2] for b in batch]))
        dones = torch.tensor(np.stack([b[3] for b in batch]))

        return obs, acts, rews, dones


# ── Sequence Collector ────────────────────────────────────────────────────────


class TronformerCollector:
    """Collects transitions and packages them into fixed-length sequences.

    No LSTM state is stored — the transformer reprocesses the full sequence
    from scratch during training.

    Args:
        seq_len: Target sequence length.
        obs_dim: Observation dimension.
    """

    def __init__(self, seq_len: int = SEQ_LEN, obs_dim: int = 14):
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self._obs:  List[np.ndarray] = []
        self._acts: List[np.ndarray] = []
        self._rews: List[float] = []
        self._done: List[bool] = []

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, done: bool):
        self._obs.append(obs.astype(np.float32))
        self._acts.append(np.array(action, dtype=np.int64))
        self._rews.append(float(reward))
        self._done.append(bool(done))

    def ready(self) -> bool:
        return len(self._obs) >= self.seq_len

    def flush(self, pad_if_short: bool = False) -> Optional[tuple]:
        """Return a complete sequence if ready, or None.

        Args:
            pad_if_short: If True, pad short sequences with zeros to seq_len.
        """
        n = len(self._obs)
        if n == 0:
            return None
        if n < self.seq_len and not pad_if_short:
            return None

        while len(self._obs) < self.seq_len:
            self._obs.append(np.zeros(self.obs_dim, dtype=np.float32))
            self._acts.append(np.zeros(2, dtype=np.int64))
            self._rews.append(0.0)
            self._done.append(True)

        seq = (
            np.stack(self._obs[:self.seq_len]),
            np.stack(self._acts[:self.seq_len]),
            np.array(self._rews[:self.seq_len]),
            np.array(self._done[:self.seq_len]),
        )
        self._obs.clear()
        self._acts.clear()
        self._rews.clear()
        self._done.clear()
        return seq


# ── n-step return helper ──────────────────────────────────────────────────────


def compute_nstep_returns(
    rews: torch.Tensor, dones: torch.Tensor, gamma: float, n: int
) -> torch.Tensor:
    """Compute n-step discounted returns for each time step in a sequence.

    Args:
        rews:  (batch, seq_len)  float32
        dones: (batch, seq_len)  bool
        gamma: Discount factor.
        n:     n-step horizon.

    Returns:
        returns: (batch, seq_len)  float32
    """
    B, T = rews.shape
    returns = torch.zeros_like(rews)
    not_done = (~dones).float()

    for t in range(T):
        g = torch.zeros(B)
        discount = torch.ones(B)
        for k in range(n):
            idx = t + k
            if idx >= T:
                break
            g = g + discount * rews[:, idx]
            discount = discount * not_done[:, idx] * gamma
        returns[:, t] = g

    return returns


# ── R2D2 Trainer (TRONformer) ─────────────────────────────────────────────────


class R2D2FormerTrainer:
    """Single actor-learner R2D2 trainer for TRONformerPolicy.

    Maintains an internal obs_deque for action selection (no LSTM state).
    Includes a learned RewardModel for shaped rewards.

    Args:
        obs_dim:            Observation dimension (14).
        d_model:            Transformer hidden size (128).
        seq_len:            Sequence length for rolling buffer.
        burnin:             Steps to skip when computing loss (causal context).
        lr:                 Adam learning rate for the policy.
        gamma:              Discount factor.
        n_step:             n-step return horizon.
        batch_size:         Batch size for gradient updates.
        target_update_freq: Steps between hard target updates.
        eps_start/end/decay_steps: Epsilon-greedy schedule.
        grad_clip:          Max gradient norm.
        shade_bins:         Optional custom shade bins.
    """

    def __init__(
        self,
        obs_dim: int = 14,
        d_model: int = 128,
        seq_len: int = SEQ_LEN,
        burnin: int = BURNIN,
        lr: float = LR,
        gamma: float = GAMMA,
        n_step: int = N_STEP,
        batch_size: int = BATCH_SIZE,
        target_update_freq: int = TARGET_UPDATE_FREQ,
        eps_start: float = EPS_START,
        eps_end: float = EPS_END,
        eps_decay_steps: int = EPS_DECAY_STEPS,
        grad_clip: float = GRAD_CLIP,
        shade_bins: Optional[np.ndarray] = None,
    ):
        self.gamma = gamma
        self.n_step = n_step
        self.seq_len = seq_len
        self.burnin = burnin
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.grad_clip = grad_clip

        self.online = TRONformerPolicy(input_dim=obs_dim, d_model=d_model, shade_bins=shade_bins)
        self.target = TRONformerPolicy(input_dim=obs_dim, d_model=d_model, shade_bins=shade_bins)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        # Reward model
        self.reward_model = RewardModel(input_dim=RM_INPUT_DIM)
        self.rm_optimizer = optim.Adam(self.reward_model.parameters(), lr=RM_LR)
        self.rm_loss_fn = nn.BCELoss()

        self._grad_steps = 0
        self._env_steps = 0

        # Rolling obs buffer for action selection
        self._obs_deque: collections.deque = collections.deque(maxlen=seq_len)

    @property
    def epsilon(self) -> float:
        frac = min(1.0, self._env_steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def reset_obs_buffer(self):
        """Clear the rolling obs buffer (call at episode start)."""
        self._obs_deque.clear()

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Epsilon-greedy action selection using the rolling obs buffer.

        Appends obs to the internal deque, stacks to a sequence, and queries
        the online policy at the last position.

        Args:
            obs: (obs_dim,) numpy array — current observation.

        Returns:
            action: [shade_idx, eta_idx] int64 array.
        """
        self._obs_deque.append(obs.astype(np.float32))
        obs_seq = np.stack(list(self._obs_deque))  # (cur_len, obs_dim)
        obs_t = torch.tensor(obs_seq, dtype=torch.float32).unsqueeze(0)  # (1, cur_len, D)

        with torch.no_grad():
            Q_shade, Q_eta = self.online(obs_t)  # (1, cur_len, ...)

        if random.random() < self.epsilon:
            shade_idx = random.randrange(len(self.online.SHADE_BINS))
            eta_idx = random.randrange(len(self.online.ETA_BINS))
        else:
            shade_idx = int(Q_shade[0, -1, :].argmax().item())
            eta_idx = int(Q_eta[0, -1, :].argmax().item())

        self._env_steps += 1
        return np.array([shade_idx, eta_idx], dtype=np.int64)

    def update(self, buffer: TronformerReplayBuffer) -> Optional[float]:
        """Sample a batch and perform one gradient step on the policy.

        For the transformer, the full sequence context is used (no separate
        LSTM burnin forward pass).  Loss is computed on positions burnin..T-1.

        Returns:
            loss (float) or None if not enough data.
        """
        if len(buffer) < self.batch_size:
            return None

        obs, acts, rews, dones = buffer.sample(self.batch_size)
        # obs:   (B, T, D)
        # acts:  (B, T, 2)
        # rews:  (B, T)
        # dones: (B, T)

        B, T, D = obs.shape
        learn_len = T - self.burnin

        # ── Online Q-values (full sequence, causal masking handles context) ──
        self.online.train()
        Q_shade_online, Q_eta_online = self.online(obs)  # (B, T, n_*)

        # Gather Q for taken actions in the learning window
        shade_taken = acts[:, self.burnin:, 0].long()  # (B, learn_len)
        eta_taken   = acts[:, self.burnin:, 1].long()

        Q_shade_taken = Q_shade_online[:, self.burnin:, :].gather(
            2, shade_taken.unsqueeze(-1)
        ).squeeze(-1)
        Q_eta_taken = Q_eta_online[:, self.burnin:, :].gather(
            2, eta_taken.unsqueeze(-1)
        ).squeeze(-1)

        # ── n-step returns ─────────────────────────────────────────────────
        rews_learn  = rews[:, self.burnin:]    # (B, learn_len)
        dones_learn = dones[:, self.burnin:]
        n_returns = compute_nstep_returns(rews_learn, dones_learn, self.gamma, self.n_step)

        # ── Double DQN targets via full-context forward ────────────────────
        with torch.no_grad():
            # Target network: full sequence
            Q_shade_tgt_full, Q_eta_tgt_full = self.target(obs)  # (B, T, n_*)

            gamma_n = self.gamma ** self.n_step

            # For each learning position t, the n-step bootstrap uses s_{t+n}
            # which is at full-sequence index burnin + t + n_step.
            # Positions where that index >= T use pure n-step returns (no bootstrap).
            shade_target = n_returns.clone()
            eta_target   = n_returns.clone()

            next_full_indices = (
                self.burnin + torch.arange(learn_len) + self.n_step
            )  # (learn_len,)
            valid = next_full_indices < T  # (learn_len,)

            if valid.any():
                valid_idx = next_full_indices[valid]  # (n_valid,)

                # Double DQN: online selects action, target evaluates value
                Q_shade_next_online_v = Q_shade_online[:, valid_idx, :].detach()
                Q_eta_next_online_v   = Q_eta_online[:, valid_idx, :].detach()

                shade_next_act = Q_shade_next_online_v.argmax(dim=-1)  # (B, n_valid)
                eta_next_act   = Q_eta_next_online_v.argmax(dim=-1)

                Q_shade_next_val = Q_shade_tgt_full[:, valid_idx, :].gather(
                    2, shade_next_act.unsqueeze(-1)
                ).squeeze(-1)  # (B, n_valid)
                Q_eta_next_val = Q_eta_tgt_full[:, valid_idx, :].gather(
                    2, eta_next_act.unsqueeze(-1)
                ).squeeze(-1)

                not_done = (~dones_learn[:, valid]).float()  # (B, n_valid)
                shade_target[:, valid] = (
                    n_returns[:, valid] + gamma_n * Q_shade_next_val * not_done
                )
                eta_target[:, valid] = (
                    n_returns[:, valid] + gamma_n * Q_eta_next_val * not_done
                )

        # ── Loss ──────────────────────────────────────────────────────────
        loss_shade = self.loss_fn(Q_shade_taken, shade_target.detach())
        loss_eta   = self.loss_fn(Q_eta_taken,   eta_target.detach())
        loss = loss_shade + loss_eta

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optimizer.step()

        self._grad_steps += 1
        if self._grad_steps % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

        self.online.eval()
        return float(loss.item())

    def update_reward_model(self, rm_data: list) -> Optional[float]:
        """Train the reward model on a batch sampled from rm_data.

        Args:
            rm_data: List of (feat_array, label) tuples. label in {0, 1}.

        Returns:
            BCE loss or None if not enough data.
        """
        if len(rm_data) < RM_BATCH_SIZE:
            return None

        indices = random.sample(range(len(rm_data)), RM_BATCH_SIZE)
        feats  = torch.tensor(
            np.stack([rm_data[i][0] for i in indices]), dtype=torch.float32
        )
        labels = torch.tensor(
            [float(rm_data[i][1]) for i in indices], dtype=torch.float32
        )

        self.reward_model.train()
        preds = self.reward_model(feats)
        rm_loss = self.rm_loss_fn(preds, labels)

        self.rm_optimizer.zero_grad()
        rm_loss.backward()
        self.rm_optimizer.step()
        self.reward_model.eval()

        return float(rm_loss.item())

    def predict_exec_prob(self, obs: np.ndarray, shade_norm: float) -> float:
        """Predict execution probability for the current obs + action.

        Args:
            obs:        14-dim observation array.
            shade_norm: Shade index normalised to [0, 1].

        Returns:
            Scalar execution probability in [0, 1].
        """
        feat = np.concatenate([obs.astype(np.float32), [shade_norm]])
        feat_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return float(self.reward_model(feat_t).item())

    def save(self, path: str):
        torch.save(self.online.state_dict(), path)
        print(f"  Saved model → {path}")

    def load(self, path: str):
        state = torch.load(path, map_location="cpu")
        self.online.load_state_dict(state)
        self.target.load_state_dict(state)
        self.online.train()
        self.target.eval()


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate(
    policy: TRONformerPolicy,
    n_episodes: int = EVAL_EPISODES,
    seq_len: int = SEQ_LEN,
) -> Tuple[float, float]:
    """Roll out the greedy policy deterministically.

    Args:
        seq_len: Rolling context window to use during eval. Must match the
                 seq_len used during training so the obs sequence fed to the
                 policy has the same structure.

    Returns:
        (mean_reward, std_reward) over n_episodes episodes.
    """
    policy.eval()
    env = TRONEnv(**ENV_KWARGS)
    rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        obs_deque: collections.deque = collections.deque(maxlen=seq_len)
        ep_reward = 0.0
        done = False

        while not done:
            obs_deque.append(obs.astype(np.float32))
            obs_seq = np.stack(list(obs_deque))
            obs_t = torch.tensor(obs_seq, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                Q_shade, Q_eta = policy(obs_t)
            shade_idx = int(Q_shade[0, -1, :].argmax().item())
            eta_idx   = int(Q_eta[0, -1, :].argmax().item())
            action = np.array([shade_idx, eta_idx])
            obs, r, terminated, truncated, _ = env.step(action)
            ep_reward += r
            done = terminated or truncated

        rewards.append(ep_reward)

    return float(np.mean(rewards)), float(np.std(rewards))


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_learning_curve(run_dir: str):
    eval_path = os.path.join(run_dir, "eval_rewards.csv")
    if not os.path.exists(eval_path):
        return

    eval_ts, eval_means, eval_stds = [], [], []
    with open(eval_path) as f:
        for row in csv.reader(f):
            if row:
                eval_ts.append(int(row[0]))
                eval_means.append(float(row[1]))
                eval_stds.append(float(row[2]))

    if not eval_ts:
        return

    eval_means = np.array(eval_means)
    eval_stds  = np.array(eval_stds)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eval_ts, eval_means, "o-", color="steelblue",
            linewidth=2, markersize=4, label="Eval mean reward", zorder=5)
    ax.fill_between(eval_ts, eval_means - eval_stds, eval_means + eval_stds,
                    alpha=0.2, color="steelblue")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode reward")
    ax.set_title("TRONformer Agent — R2D2 Training")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    out = os.path.join(run_dir, "learning_curve.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main training loop ────────────────────────────────────────────────────────


def train(args):
    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"tronformer_{tag}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nRun directory: {run_dir}")

    seq_len  = args.seq_len
    burnin   = args.burnin
    batch_sz = args.batch_size
    d_model  = args.d_model
    total_steps = args.timesteps
    reward_norm = ENV_KWARGS["normalizers"]["reward"]

    shade_bins = ENV_KWARGS.get("shade_bins", UNIFORM_SHADE_BINS)

    trainer   = R2D2FormerTrainer(
        obs_dim=14,
        d_model=d_model,
        seq_len=seq_len,
        burnin=burnin,
        batch_size=batch_sz,
        shade_bins=shade_bins,
        eps_decay_steps=args.eps_decay_steps,
    )
    buffer    = TronformerReplayBuffer(seq_len=seq_len)
    collector = TronformerCollector(seq_len=seq_len)

    if args.load:
        print(f"Resuming from checkpoint: {args.load}")
        trainer.load(args.load)

    eval_csv = os.path.join(run_dir, "eval_rewards.csv")
    best_model_path = os.path.join(run_dir, "best_model.pt")
    best_mean = -np.inf

    env = TRONEnv(**ENV_KWARGS)
    obs, _ = env.reset()
    trainer.reset_obs_buffer()

    ep_reward = 0.0
    ep_rewards: list = []
    step = args.start_step
    trainer._env_steps = args.start_step
    next_eval = ((args.start_step // EVAL_FREQ) + 1) * EVAL_FREQ
    losses: list = []

    print(f"\nTraining TRONformer (R2D2) for {total_steps:,} steps")
    print(f"  seq_len={seq_len}, burnin={burnin}, batch={batch_sz}, n_step={N_STEP}")
    print(f"  d_model={d_model}, shade_bins={len(shade_bins)}")
    print(f"  eps_decay_steps={args.eps_decay_steps}, train_freq={TRAIN_FREQ}")
    print(f"  buffer_capacity={BUFFER_CAPACITY}, reward_model=disabled")
    print(f"  env sim_time={ENV_KWARGS['sim_time']}, lam={ENV_KWARGS['lam']}, "
          f"lam_zi={ENV_KWARGS['lam_zi']}\n")

    t0 = time.time()

    while step < total_steps:
        # Action selection (updates internal obs deque)
        action = trainer.select_action(obs)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        shaped_reward = reward  # reward model disabled (GAMMA_RM_START=0.0)

        collector.add(obs, action, shaped_reward, done)
        ep_reward += reward  # track original reward for logging
        step += 1
        obs = next_obs

        if collector.ready():
            seq = collector.flush()
            if seq is not None:
                buffer.add(*seq)

        if done:
            ep_rewards.append(ep_reward)
            ep_reward = 0.0
            seq = collector.flush(pad_if_short=True)
            if seq is not None:
                buffer.add(*seq)
            obs, _ = env.reset()
            trainer.reset_obs_buffer()

        # Train policy
        if step % TRAIN_FREQ == 0 and len(buffer) >= LEARNING_STARTS:
            loss = trainer.update(buffer)
            if loss is not None:
                losses.append(loss)

        # Reward model training is disabled (GAMMA_RM_START=0.0)

        # Evaluate
        if step >= next_eval:
            mean_r, std_r = evaluate(trainer.online, seq_len=seq_len)
            elapsed = (time.time() - t0) / 60
            print(
                f"  step={step:>8,}  eval={mean_r:+.3f}±{std_r:.3f}"
                f"  eps={trainer.epsilon:.3f}"
                f"  loss={np.mean(losses[-100:]) if losses else 0:.4f}"
                f"  {elapsed:.1f}min"
            )
            with open(eval_csv, "a", newline="") as f:
                csv.writer(f).writerow([step, f"{mean_r:.4f}", f"{std_r:.4f}"])

            if mean_r > best_mean:
                best_mean = mean_r
                trainer.save(best_model_path)

            next_eval += EVAL_FREQ

    # ── Final save ────────────────────────────────────────────────────────
    final_path = os.path.join(run_dir, "final_model.pt")
    trainer.save(final_path)

    print(f"\nTraining complete. Best eval: {best_mean:+.4f}")
    print(f"All outputs in: {run_dir}/")

    plot_learning_curve(run_dir)


# ── Eval-only mode ────────────────────────────────────────────────────────────


def eval_only(args):
    assert args.load, "--eval-only requires --load <path>"
    shade_bins = ENV_KWARGS.get("shade_bins", UNIFORM_SHADE_BINS)
    print(f"Loading {args.load}")
    policy = TRONformerPolicy(input_dim=14, d_model=args.d_model, shade_bins=shade_bins)
    policy.load_state_dict(torch.load(args.load, map_location="cpu"))
    policy.eval()

    n = args.eval_episodes
    mean_r, std_r = evaluate(policy, n_episodes=n, seq_len=args.seq_len)
    print(f"Eval ({n} episodes): mean={mean_r:+.4f}  std={std_r:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=2_000_000,
                   help="Total environment steps (default 2 000 000)")
    p.add_argument("--seq-len", type=int, default=SEQ_LEN,
                   help=f"Sequence length / rolling context (default {SEQ_LEN})")
    p.add_argument("--burnin", type=int, default=BURNIN,
                   help=f"Steps to skip for loss (causal context) (default {BURNIN})")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                   help=f"Batch size (default {BATCH_SIZE})")
    p.add_argument("--tag", type=str, default=None,
                   help="Run name tag; defaults to timestamp")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; only evaluate a saved model")
    p.add_argument("--load", type=str, default=None,
                   help="Path to .pt state_dict for --eval-only or to warm-start")
    p.add_argument("--start-step", type=int, default=0,
                   help="Step count to resume from")
    p.add_argument("--n-bg", type=int, default=None,
                   help="Override num_background_agents in ENV_KWARGS")
    p.add_argument("--lam-zi", type=float, default=None,
                   help="Override lam_zi in ENV_KWARGS")
    p.add_argument("--skew-bins", action="store_true",
                   help="Use skewed shade bins")
    p.add_argument("--d-model", type=int, default=128,
                   help="Transformer hidden / embedding size (default 128)")
    p.add_argument("--eps-decay-steps", type=int, default=EPS_DECAY_STEPS,
                   help=f"Epsilon decay steps (default {EPS_DECAY_STEPS})")
    p.add_argument("--bg-strategy", type=int, default=None,
                   help="Fix all BG agents to a single strategy index 0-9")
    p.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES,
                   help=f"Episodes for --eval-only mode (default {EVAL_EPISODES})")
    p.add_argument("--seed", type=int, default=42,
                   help="Global random seed (default 42)")
    p.add_argument("--sim-time", type=int, default=None,
                   help="Override sim_time in ENV_KWARGS (default: use ENV_KWARGS value)")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.n_bg is not None:
        ENV_KWARGS["num_background_agents"] = args.n_bg
    if args.lam_zi is not None:
        ENV_KWARGS["lam_zi"] = args.lam_zi
    if args.sim_time is not None:
        ENV_KWARGS["sim_time"] = args.sim_time
    if args.skew_bins:
        ENV_KWARGS["shade_bins"] = SKEWED_SHADE_BINS
    if args.bg_strategy is not None:
        strat = _STRATEGIES[args.bg_strategy]
        ENV_KWARGS["bg_strategies"] = [{"shade": strat["shade"], "eta": strat["eta"]}]

    if args.eval_only:
        eval_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
