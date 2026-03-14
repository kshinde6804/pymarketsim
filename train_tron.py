"""
train_tron.py — Train the TRON agent on TRONEnv using R2D2.

R2D2 (Recurrent Replay Distributed DQN) with:
  - Dueling DQN architecture (TRONPolicy)
  - LSTM recurrent state stored in replay sequences
  - Fixed-length sequence replay (seq_len=32, burnin=4)
  - n-step returns (n=5)
  - Double DQN target computation
  - Epsilon-greedy exploration with linear decay
  - Huber loss on both shade and eta streams

Usage
-----
    # Default 2M-step run
    python train_tron.py --tag v1

    # Quick smoke test (5k steps)
    python train_tron.py --timesteps 5000 --tag smoke

    # Evaluate a saved model
    python train_tron.py --eval-only --load runs/tron_v1/best_model.pt

Outputs (all under runs/tron_<tag>/)
--------------------------------------
    best_model.pt       TRONPolicy state_dict (best eval reward)
    final_model.pt      TRONPolicy state_dict (end of training)
    eval_rewards.csv    (timestep, mean_reward, std_reward) at each eval
    learning_curve.png  Training + eval reward vs. timesteps
"""

import argparse
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

from marketsim.agent.tron_agent import TRONPolicy
from marketsim.wrappers.tron_env import TRONEnv

# ── Shade bin options ──────────────────────────────────────────────────────────

# Default: 42 uniform bins over [0, 600] (14.3 spacing throughout)
UNIFORM_SHADE_BINS = np.linspace(0, 600, 42)

# Skewed: 10 coarse bins in [0, 300), 32 fine bins in [300, 600] (~9.7 spacing
# in the competitive region near S8). More resolution where it matters.
SKEWED_SHADE_BINS = np.concatenate([
    np.linspace(0, 300, 11)[:-1],  # 0, 30, 60, ..., 270
    np.linspace(300, 600, 32),      # 300, 309.7, ..., 600
])

# ── ENV hyper-parameters (match equilibrium experiment) ───────────────────────

# ZI strategy set from equilibrium_experiment.py (inlined to avoid path issues)
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
    num_background_agents=24,  # paper: N=25 total (24 ZI + 1 TRON)
    sim_time=2000,
    lam=0.005,
    lam_zi=0.005,         # matches background lam → equal participation (~10 RL steps/episode)
    mean=1e5,
    r=0.01,
    shock_var=1e6,
    q_max=10,
    pv_var=5e6,
    shade=[250, 500],
    normalizers={"fundamental": 1e5, "invt": 10, "reward": 1e3, "pv": 5e5},
    bg_strategies=BG_STRATEGIES,
    warmup_fraction=0.0,
    shade_bins=UNIFORM_SHADE_BINS,  # overridden to SKEWED_SHADE_BINS via --skew-bins
)

# ── R2D2 hyper-parameters ─────────────────────────────────────────────────────

SEQ_LEN = 32          # ~10 steps/episode at lam_zi=0.005; 32 contains full episodes (99.9th pctile ~22)
BURNIN = 4            # Small warmup; h0=0 at episode start so no stale state
N_STEP = 5            # n-step return horizon
GAMMA = 0.99          # Discount factor
BATCH_SIZE = 64       # Sequences sampled per gradient update
BUFFER_CAPACITY = 50_000  # Buffer sized for shorter ~10-step episodes
LR = 1e-4
TARGET_UPDATE_FREQ = 1000  # Hard-copy online → target every N gradient steps
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 1_000_000
GRAD_CLIP = 10.0
LEARNING_STARTS = 100    # Sequences collected before first gradient step
TRAIN_FREQ = 4           # Gradient update every N env steps
EVAL_FREQ = 50_000       # Evaluate every N environment steps
EVAL_EPISODES = 100


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class SequenceReplayBuffer:
    """Fixed-length sequence replay buffer for R2D2.

    Each stored entry is a complete sequence of length seq_len:
        obs_seq:  (seq_len, obs_dim)   float32
        act_seq:  (seq_len, 2)         int64   [shade_idx, eta_idx]
        rew_seq:  (seq_len,)           float32
        done_seq: (seq_len,)           bool
        h0:       (1, hidden_dim)      float32  LSTM h at sequence start
        c0:       (1, hidden_dim)      float32  LSTM c at sequence start

    Args:
        capacity:   Maximum number of sequences stored.
        seq_len:    Length of each stored sequence.
        obs_dim:    Observation dimensionality (14 for TRON).
        hidden_dim: LSTM hidden size (128).
    """

    def __init__(
        self,
        capacity: int = BUFFER_CAPACITY,
        seq_len: int = SEQ_LEN,
        obs_dim: int = 14,
        hidden_dim: int = 128,
    ):
        self.capacity = capacity
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        # List-based circular buffer: O(1) indexing, unlike deque which is O(n).
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
        h0: torch.Tensor,
        c0: torch.Tensor,
    ):
        entry = (
            obs_seq.astype(np.float32),
            act_seq.astype(np.int64),
            rew_seq.astype(np.float32),
            done_seq.astype(bool),
            h0.squeeze(0).detach().cpu(),  # (1, hidden_dim)
            c0.squeeze(0).detach().cpu(),
        )
        if len(self._buf) < self.capacity:
            self._buf.append(entry)
        else:
            self._buf[self._pos] = entry
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int):
        """Sample a batch of sequences uniformly at random.

        Returns:
            obs:    (batch, seq_len, obs_dim)    float32 tensor
            acts:   (batch, seq_len, 2)          int64 tensor
            rews:   (batch, seq_len)             float32 tensor
            dones:  (batch, seq_len)             bool tensor
            h0:     (1, batch, hidden_dim)       float32 tensor
            c0:     (1, batch, hidden_dim)       float32 tensor
        """
        indices = random.sample(range(len(self._buf)), batch_size)
        batch = [self._buf[i] for i in indices]  # O(1) per access on list

        obs   = torch.tensor(np.stack([b[0] for b in batch]))
        acts  = torch.tensor(np.stack([b[1] for b in batch]))
        rews  = torch.tensor(np.stack([b[2] for b in batch]))
        dones = torch.tensor(np.stack([b[3] for b in batch]))
        h0    = torch.stack([b[4] for b in batch], dim=1)  # (1, B, H)
        c0    = torch.stack([b[5] for b in batch], dim=1)

        return obs, acts, rews, dones, h0, c0


# ── Sequence collector ────────────────────────────────────────────────────────

class SequenceCollector:
    """Collects transitions and packages them into fixed-length sequences.

    Accumulates transitions from the environment. When seq_len steps have been
    collected (or the episode ends and the partial buffer is padded), yields a
    complete sequence ready for the replay buffer.

    The LSTM state at the START of each sequence (h0, c0) is stored alongside
    the transitions so the learner can properly initialise the LSTM during
    training.

    Args:
        seq_len:    Target sequence length.
        obs_dim:    Observation dimension.
        hidden_dim: LSTM hidden size.
    """

    def __init__(self, seq_len: int = SEQ_LEN, obs_dim: int = 14, hidden_dim: int = 128):
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self._reset_buffers()

    def _reset_buffers(self):
        self._obs  = []
        self._acts = []
        self._rews = []
        self._done = []
        self._h0 = torch.zeros(1, 1, self.hidden_dim)
        self._c0 = torch.zeros(1, 1, self.hidden_dim)
        self._h0_set = False

    def start_sequence(self, h: torch.Tensor, c: torch.Tensor):
        """Record LSTM state at start of a new sequence segment."""
        if not self._h0_set:
            self._h0 = h.detach().clone()
            self._c0 = c.detach().clone()
            self._h0_set = True

    def add(self, obs, action, reward, done):
        self._obs.append(obs.astype(np.float32))
        self._acts.append(np.array(action, dtype=np.int64))
        self._rews.append(float(reward))
        self._done.append(bool(done))

    def ready(self) -> bool:
        return len(self._obs) >= self.seq_len

    def flush(self, pad_if_short: bool = False) -> Optional[tuple]:
        """Return a complete sequence if ready, else None (or pad if requested)."""
        n = len(self._obs)
        if n == 0:
            return None
        if n < self.seq_len and not pad_if_short:
            return None

        # Pad short sequences with zeros (terminal padding)
        while len(self._obs) < self.seq_len:
            self._obs.append(np.zeros(self.obs_dim, dtype=np.float32))
            self._acts.append(np.zeros(2, dtype=np.int64))
            self._rews.append(0.0)
            self._done.append(True)

        seq = (
            np.stack(self._obs[:self.seq_len]),   # (T, D)
            np.stack(self._acts[:self.seq_len]),  # (T, 2)
            np.array(self._rews[:self.seq_len]),  # (T,)
            np.array(self._done[:self.seq_len]),  # (T,)
            self._h0,
            self._c0,
        )
        self._reset_buffers()
        return seq


# ── n-step return helper ──────────────────────────────────────────────────────

def compute_nstep_returns(rews: torch.Tensor, dones: torch.Tensor, gamma: float, n: int) -> torch.Tensor:
    """Compute n-step discounted returns for each time step in a sequence.

    For step t: R_t = r_t + γ·r_{t+1} + ... + γ^{n-1}·r_{t+n-1}
    Accumulation stops at the first done=True (episode boundary).

    Args:
        rews:  (batch, seq_len)  float32
        dones: (batch, seq_len)  bool
        gamma: discount factor
        n:     n-step horizon

    Returns:
        returns: (batch, seq_len)  float32
    """
    B, T = rews.shape
    returns = torch.zeros_like(rews)
    # not_done[b, t] = 1 iff sample b is NOT done at step t
    not_done = (~dones).float()  # (B, T)

    for t in range(T):
        g = torch.zeros(B)
        discount = torch.ones(B)   # per-sample discount (zeroed after done)
        for k in range(n):
            idx = t + k
            if idx >= T:
                break
            g = g + discount * rews[:, idx]
            # After adding reward at idx, zero out discount for any done samples
            discount = discount * not_done[:, idx] * gamma
        returns[:, t] = g

    return returns


# ── R2D2 Trainer ──────────────────────────────────────────────────────────────

class R2D2Trainer:
    """Single actor-learner R2D2 trainer for TRONPolicy.

    Args:
        obs_dim:    Observation dimension (14).
        hidden_dim: LSTM hidden size (128).
        lr:         Adam learning rate.
        gamma:      Discount factor.
        n_step:     n-step return horizon.
        seq_len:    Sequence length.
        burnin:     Steps to run LSTM warmup before computing loss.
        batch_size: Batch size for gradient updates.
        target_update_freq: Steps between hard target updates.
        eps_start, eps_end, eps_decay_steps: Epsilon-greedy schedule.
        grad_clip:  Max gradient norm.
    """

    def __init__(
        self,
        obs_dim: int = 14,
        hidden_dim: int = 128,
        lr: float = LR,
        gamma: float = GAMMA,
        n_step: int = N_STEP,
        seq_len: int = SEQ_LEN,
        burnin: int = BURNIN,
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

        self.online = TRONPolicy(input_dim=obs_dim, hidden_dim=hidden_dim, shade_bins=shade_bins)
        self.target = TRONPolicy(input_dim=obs_dim, hidden_dim=hidden_dim, shade_bins=shade_bins)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self._grad_steps = 0
        self._env_steps = 0

    @property
    def epsilon(self) -> float:
        frac = min(1.0, self._env_steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(
        self,
        obs: np.ndarray,
        h_c: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[np.ndarray, Tuple[torch.Tensor, torch.Tensor]]:
        """Epsilon-greedy action selection with LSTM state update.

        Args:
            obs: (obs_dim,) numpy array
            h_c: current LSTM state

        Returns:
            action: [shade_idx, eta_idx]
            new h_c
        """
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # (1, D)
        with torch.no_grad():
            Q_shade, Q_eta, new_h_c = self.online(obs_t, h_c)

        if random.random() < self.epsilon:
            shade_idx = random.randrange(len(self.online.SHADE_BINS))
            eta_idx = random.randrange(len(self.online.ETA_BINS))
        else:
            shade_idx = int(Q_shade.argmax(dim=-1).item())
            eta_idx = int(Q_eta.argmax(dim=-1).item())

        self._env_steps += 1
        return np.array([shade_idx, eta_idx], dtype=np.int64), new_h_c

    def update(self, buffer: SequenceReplayBuffer) -> Optional[float]:
        """Sample a batch and perform one gradient step.

        Returns:
            loss (float) or None if not enough data.
        """
        if len(buffer) < self.batch_size:
            return None

        obs, acts, rews, dones, h0, c0 = buffer.sample(self.batch_size)
        # obs:   (B, T, D)
        # acts:  (B, T, 2)
        # rews:  (B, T)
        # dones: (B, T)
        # h0/c0: (1, B, H)

        B, T, D = obs.shape
        learn_len = T - self.burnin  # steps for which we compute loss

        # ── Burnin: run online LSTM through first `burnin` steps ──────────
        with torch.no_grad():
            _, _, (h_burn, c_burn) = self.online(obs[:, :self.burnin, :], (h0, c0))

        # ── Online Q-values for learning steps ────────────────────────────
        self.online.train()
        Q_shade_online, Q_eta_online, _ = self.online(
            obs[:, self.burnin:, :], (h_burn, c_burn)
        )
        # Q_shade_online: (B, learn_len, n_shade)
        # Q_eta_online:   (B, learn_len, n_eta)

        # Gather Q for taken actions
        shade_taken = acts[:, self.burnin:, 0].long()   # (B, learn_len)
        eta_taken   = acts[:, self.burnin:, 1].long()   # (B, learn_len)

        Q_shade_taken = Q_shade_online.gather(
            2, shade_taken.unsqueeze(-1)
        ).squeeze(-1)  # (B, learn_len)
        Q_eta_taken = Q_eta_online.gather(
            2, eta_taken.unsqueeze(-1)
        ).squeeze(-1)  # (B, learn_len)

        # ── n-step returns ─────────────────────────────────────────────────
        rews_learn = rews[:, self.burnin:]    # (B, learn_len)
        dones_learn = dones[:, self.burnin:]  # (B, learn_len)
        n_returns = compute_nstep_returns(rews_learn, dones_learn, self.gamma, self.n_step)

        # ── Double DQN targets ─────────────────────────────────────────────
        # Online selects greedy action on s'; target evaluates it.
        obs_next = obs[:, self.burnin + 1:, :]  # (B, learn_len-1, D)
        with torch.no_grad():
            # Run target through burnin too for consistency
            _, _, (h_tgt, c_tgt) = self.target(obs[:, :self.burnin, :], (h0, c0))

            # Online greedy action on s'
            Q_shade_next_online, Q_eta_next_online, _ = self.online(
                obs_next, (h_burn.detach(), c_burn.detach())
            )
            shade_next_act = Q_shade_next_online.argmax(dim=-1)  # (B, learn_len-1)
            eta_next_act   = Q_eta_next_online.argmax(dim=-1)

            # Target value at online's greedy action
            Q_shade_next_tgt, Q_eta_next_tgt, _ = self.target(obs_next, (h_tgt, c_tgt))
            Q_shade_next_val = Q_shade_next_tgt.gather(
                2, shade_next_act.unsqueeze(-1)
            ).squeeze(-1)  # (B, learn_len-1)
            Q_eta_next_val = Q_eta_next_tgt.gather(
                2, eta_next_act.unsqueeze(-1)
            ).squeeze(-1)

            # Bootstrap: not available for the last step in the learning window
            not_done_last = (~dones_learn[:, -1]).float()  # (B,)
            gamma_n = self.gamma ** self.n_step

            # Targets for steps 0..learn_len-2 use bootstrapped value
            # Target for step learn_len-1 uses only n-step return
            shade_target = n_returns.clone()
            eta_target   = n_returns.clone()

            shade_target[:, :-1] = (
                n_returns[:, :-1]
                + gamma_n * Q_shade_next_val * (~dones_learn[:, :-1]).float()
            )
            eta_target[:, :-1] = (
                n_returns[:, :-1]
                + gamma_n * Q_eta_next_val * (~dones_learn[:, :-1]).float()
            )
            # Last step: just n-step return (no bootstrap)
            shade_target[:, -1] = n_returns[:, -1]
            eta_target[:, -1]   = n_returns[:, -1]

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

def evaluate(policy: TRONPolicy, n_episodes: int = EVAL_EPISODES) -> Tuple[float, float]:
    """Roll out the greedy policy deterministically.

    Returns:
        (mean_reward, std_reward) over n_episodes episodes.
    """
    policy.eval()
    env = TRONEnv(**ENV_KWARGS)
    rewards = []

    for _ in range(n_episodes):
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
            action = np.array([shade_idx, eta_idx])
            obs, r, terminated, truncated, _ = env.step(action)
            ep_reward += r
            done = terminated or truncated

        rewards.append(ep_reward)

    return float(np.mean(rewards)), float(np.std(rewards))


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_learning_curve(run_dir: str, total_timesteps: int):
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
    ax.plot(eval_ts, eval_means, "o-", color="darkorange",
            linewidth=2, markersize=4, label="Eval mean reward", zorder=5)
    ax.fill_between(eval_ts, eval_means - eval_stds, eval_means + eval_stds,
                    alpha=0.2, color="darkorange")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode reward")
    ax.set_title("TRON Agent — R2D2 Training")
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
    run_dir = os.path.join("runs", f"tron_{tag}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nRun directory: {run_dir}")

    seq_len   = args.seq_len
    burnin    = args.burnin
    batch_sz  = args.batch_size

    shade_bins = ENV_KWARGS.get('shade_bins', UNIFORM_SHADE_BINS)
    hidden_dim = args.hidden_dim

    trainer = R2D2Trainer(
        obs_dim=14,
        hidden_dim=hidden_dim,
        seq_len=seq_len,
        burnin=burnin,
        batch_size=batch_sz,
        shade_bins=shade_bins,
        eps_decay_steps=args.eps_decay_steps,
    )
    buffer  = SequenceReplayBuffer(seq_len=seq_len, hidden_dim=hidden_dim)
    collector = SequenceCollector(seq_len=seq_len, hidden_dim=hidden_dim)

    if args.load:
        print(f"Resuming from checkpoint: {args.load}")
        trainer.load(args.load)

    eval_csv = os.path.join(run_dir, "eval_rewards.csv")
    best_model_path = os.path.join(run_dir, "best_model.pt")
    best_mean = -np.inf

    env = TRONEnv(**ENV_KWARGS)
    obs, _ = env.reset()
    h = torch.zeros(1, 1, hidden_dim)
    c = torch.zeros(1, 1, hidden_dim)
    collector.start_sequence(h, c)

    ep_reward = 0.0
    ep_rewards = []
    total_steps = args.timesteps
    step = args.start_step
    trainer._env_steps = args.start_step
    next_eval = ((args.start_step // EVAL_FREQ) + 1) * EVAL_FREQ
    losses = []

    bins_desc = "skewed" if np.array_equal(shade_bins, SKEWED_SHADE_BINS) else "uniform"
    print(f"\nTraining TRON (R2D2) for {total_steps:,} steps")
    print(f"  seq_len={seq_len}, burnin={burnin}, batch={batch_sz}, n_step={N_STEP}")
    print(f"  hidden_dim={hidden_dim}, shade_bins={bins_desc} (n={len(shade_bins)})")
    print(f"  eps_decay_steps={args.eps_decay_steps}, train_freq={TRAIN_FREQ}")
    print(f"  env sim_time={ENV_KWARGS['sim_time']}, lam={ENV_KWARGS['lam']}, lam_zi={ENV_KWARGS['lam_zi']}\n")

    t0 = time.time()

    while step < total_steps:
        action, (h, c) = trainer.select_action(obs, (h, c))
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        collector.add(obs, action, reward, done)
        ep_reward += reward
        step += 1

        obs = next_obs

        # When the collector has a full seq_len window, flush it.
        if collector.ready():
            seq = collector.flush()
            if seq is not None:
                buffer.add(*seq)
            # Record LSTM state at start of next sequence segment.
            collector.start_sequence(h, c)

        if done:
            ep_rewards.append(ep_reward)
            ep_reward = 0.0
            # Flush partial episode with zero-padding so each sequence = 1 episode.
            # This matches eval where LSTM is reset to zeros at every episode start.
            seq = collector.flush(pad_if_short=True)
            if seq is not None:
                buffer.add(*seq)
            obs, _ = env.reset()
            h = torch.zeros(1, 1, hidden_dim)
            c = torch.zeros(1, 1, hidden_dim)
            collector.start_sequence(h, c)

        # Learn every TRAIN_FREQ steps once buffer has enough sequences.
        if step % TRAIN_FREQ == 0 and len(buffer) >= LEARNING_STARTS:
            loss = trainer.update(buffer)
            if loss is not None:
                losses.append(loss)

        # Evaluate
        if step >= next_eval:
            mean_r, std_r = evaluate(trainer.online)
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

    plot_learning_curve(run_dir, total_steps)


# ── Eval-only mode ────────────────────────────────────────────────────────────

def eval_only(args):
    assert args.load, "--eval-only requires --load <path>"
    shade_bins = ENV_KWARGS.get('shade_bins', UNIFORM_SHADE_BINS)
    print(f"Loading {args.load}")
    policy = TRONPolicy(input_dim=14, hidden_dim=args.hidden_dim, shade_bins=shade_bins)
    policy.load_state_dict(torch.load(args.load, map_location="cpu"))
    policy.eval()

    n = args.eval_episodes
    mean_r, std_r = evaluate(policy, n_episodes=n)
    print(f"Eval ({n} episodes): mean={mean_r:+.4f}  std={std_r:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=2_000_000,
                   help="Total environment steps (default 2 000 000)")
    p.add_argument("--seq-len", type=int, default=SEQ_LEN,
                   help=f"Sequence length (default {SEQ_LEN})")
    p.add_argument("--burnin", type=int, default=BURNIN,
                   help=f"Burnin steps (default {BURNIN})")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                   help=f"Batch size (default {BATCH_SIZE})")
    p.add_argument("--tag", type=str, default=None,
                   help="Run name tag; defaults to timestamp")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; only evaluate a saved model")
    p.add_argument("--load", type=str, default=None,
                   help="Path to .pt state_dict for --eval-only or to warm-start")
    p.add_argument("--start-step", type=int, default=0,
                   help="Step count to resume from (sets epsilon schedule and eval trigger)")
    p.add_argument("--n-bg", type=int, default=None,
                   help="Override num_background_agents in ENV_KWARGS")
    p.add_argument("--lam-zi", type=float, default=None,
                   help="Override lam_zi in ENV_KWARGS (RL agent arrival rate)")
    p.add_argument("--skew-bins", action="store_true",
                   help="Use skewed shade bins: 10 coarse in [0,300), 32 fine in [300,600]")
    p.add_argument("--hidden-dim", type=int, default=128,
                   help="LSTM hidden dimension (default 128)")
    p.add_argument("--eps-decay-steps", type=int, default=EPS_DECAY_STEPS,
                   help=f"Steps over which epsilon decays from 1.0 to 0.05 (default {EPS_DECAY_STEPS})")
    p.add_argument("--bg-strategy", type=int, default=None,
                   help="Fix all BG agents to a single strategy index 0-9")
    p.add_argument("--eval-episodes", type=int, default=50,
                   help="Episodes for --eval-only mode (default 50)")
    p.add_argument("--seed", type=int, default=42,
                   help="Global random seed for reproducibility (default 42)")
    return p.parse_args()


def main():
    args = parse_args()
    # Apply global seed before any randomness
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.n_bg is not None:
        ENV_KWARGS['num_background_agents'] = args.n_bg
    if args.lam_zi is not None:
        ENV_KWARGS['lam_zi'] = args.lam_zi
    if args.skew_bins:
        ENV_KWARGS['shade_bins'] = SKEWED_SHADE_BINS
    if args.bg_strategy is not None:
        strat = _STRATEGIES[args.bg_strategy]
        ENV_KWARGS['bg_strategies'] = [{'shade': strat['shade'], 'eta': strat['eta']}]
    if args.eval_only:
        eval_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
