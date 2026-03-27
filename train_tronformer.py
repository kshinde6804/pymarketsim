"""
train_tronformer.py — Train the TRONformer agent on TRONEnv using R2D2.

R2D2 (Recurrent Replay Distributed DQN) adapted for a transformer backbone:
  - Dueling DQN architecture (TRONformerPolicy — Pre-LN transformer)
  - Rolling obs buffer stored in replay sequences (no LSTM h/c state)
  - Fixed-length sequence replay (seq_len=32)
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    normalizers={"fundamental": 1e5, "invt": 10, "reward": 1e3, "pv": 5e5},
    bg_strategies=BG_STRATEGIES,
    warmup_fraction=0.0,
    shade_bins=UNIFORM_SHADE_BINS,
)

# ── R2D2 hyper-parameters ─────────────────────────────────────────────────────

N_STEP = 5
GAMMA = 0.99
BATCH_SIZE = 512
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
    """Per-transition replay buffer for TRONformer CLS-based policy.

    Each stored entry is a single transition with n-step return:
        obs_seq:      (seq_len, obs_dim)  float32  — rolling context at time t
        action:       (2,)               int64    — [shade_idx, eta_idx]
        nstep_return: scalar             float32  — n-step discounted return
        next_obs_seq: (seq_len, obs_dim)  float32  — rolling context at time t+n
        done:         bool               — episode ended within n-step horizon

    Args:
        capacity: Maximum number of transitions stored.
        seq_len:  Rolling context window length.
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
        action: np.ndarray,
        nstep_return: float,
        next_obs_seq: np.ndarray,
        done: bool,
    ):
        entry = (
            obs_seq.astype(np.float32),
            action.astype(np.int64),
            float(nstep_return),
            next_obs_seq.astype(np.float32),
            bool(done),
        )
        if len(self._buf) < self.capacity:
            self._buf.append(entry)
        else:
            self._buf[self._pos] = entry
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int):
        """Sample a batch of transitions uniformly at random.

        Returns:
            obs:      (batch, seq_len, obs_dim)  float32 tensor
            acts:     (batch, 2)                 int64 tensor
            rews:     (batch,)                   float32 tensor
            next_obs: (batch, seq_len, obs_dim)  float32 tensor
            dones:    (batch,)                   bool tensor
        """
        indices = random.sample(range(len(self._buf)), batch_size)
        batch = [self._buf[i] for i in indices]

        obs      = torch.tensor(np.stack([b[0] for b in batch]))
        acts     = torch.tensor(np.stack([b[1] for b in batch]))
        rews     = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        next_obs = torch.tensor(np.stack([b[3] for b in batch]))
        dones    = torch.tensor([b[4] for b in batch])

        return obs, acts, rews, next_obs, dones


# ── N-step Transition Buffer ──────────────────────────────────────────────────


class NStepBuffer:
    """Accumulates single-step transitions and emits n-step discounted returns.

    At each step t, the caller adds (obs_seq_t, action_t, reward_t, done_t).
    Once n transitions have accumulated, ``get(next_obs_seq)`` pops the oldest
    and returns a complete (obs_seq_t, action_t, G_t, next_obs_seq, done_flag)
    tuple where G_t = Σ_{k=0}^{n-1} γ^k r_{t+k} (sum stops early if done).

    At episode end ``drain(done_obs_seq)`` flushes all remaining entries with
    truncated returns and done=True (no bootstrapping beyond episode boundary).

    Args:
        n_step: n-step return horizon.
        gamma:  Discount factor.
    """

    def __init__(self, n_step: int, gamma: float):
        self.n_step = n_step
        self.gamma  = gamma
        self._buf: List[tuple] = []  # (obs_seq, action, reward, done)

    def __len__(self) -> int:
        return len(self._buf)

    def add(self, obs_seq: np.ndarray, action: np.ndarray, reward: float, done: bool):
        self._buf.append((obs_seq.astype(np.float32), action.copy(), float(reward), bool(done)))

    def get(self, next_obs_seq: np.ndarray) -> tuple:
        """Pop oldest entry and return its n-step transition.

        Args:
            next_obs_seq: Rolling obs context at time t+n (bootstrap state).
        """
        assert len(self._buf) >= self.n_step, "Buffer not full yet"
        nstep_return = 0.0
        done_flag    = False
        for k in range(self.n_step):
            _, _, r, d = self._buf[k]
            nstep_return += (self.gamma ** k) * r
            if d:
                done_flag = True
                break
        obs_t, act_t, _, _ = self._buf.pop(0)
        return obs_t, act_t, nstep_return, next_obs_seq.astype(np.float32), done_flag

    def drain(self, done_obs_seq: np.ndarray) -> List[tuple]:
        """Flush all remaining transitions at episode end with truncated returns.

        Sets done=True for all: the episode boundary prevents bootstrapping
        regardless of how many rewards remain in the buffer.
        """
        results = []
        while self._buf:
            nstep_return = 0.0
            for k in range(len(self._buf)):
                _, _, r, d = self._buf[k]
                nstep_return += (self.gamma ** k) * r
                if d:
                    break
            obs_t, act_t, _, _ = self._buf.pop(0)
            results.append((obs_t, act_t, nstep_return, done_obs_seq.astype(np.float32), True))
        return results

    def clear(self):
        self._buf.clear()


# ── R2D2 Trainer (TRONformer) ─────────────────────────────────────────────────


class R2D2FormerTrainer:
    """Single actor-learner R2D2 trainer for TRONformerPolicy.

    Maintains an internal obs_deque for action selection (no LSTM state).
    Includes a learned RewardModel for shaped rewards.

    Args:
        obs_dim:            Observation dimension (14).
        d_model:            Transformer hidden size (128).
        n_layers:           Number of Pre-LN transformer blocks (default 2).
        seq_len:            Rolling context window length.
        lr:                 Adam learning rate for the policy.
        gamma:              Discount factor.
        n_step:             n-step return horizon.
        batch_size:         Batch size for gradient updates.
        target_update_freq: Steps between hard target updates.
        eps_start/end/decay_steps: Epsilon-greedy schedule.
        grad_clip:          Max gradient norm.
        shade_bins:         Optional custom shade bins.
        boltzmann:          Use Boltzmann (softmax) exploration instead of epsilon-greedy.
        tau_start:          Initial temperature for Boltzmann exploration.
        tau_end:            Final temperature for Boltzmann exploration.
    """

    def __init__(
        self,
        obs_dim: int = 14,
        d_model: int = 128,
        n_layers: int = 2,
        seq_len: int = SEQ_LEN,
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
        boltzmann: bool = False,
        tau_start: float = 2.0,
        tau_end: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        self.gamma = gamma
        self.n_step = n_step
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.grad_clip = grad_clip
        self.boltzmann = boltzmann
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.device = device or DEVICE

        self.online = TRONformerPolicy(input_dim=obs_dim, d_model=d_model, n_layers=n_layers, shade_bins=shade_bins).to(self.device)
        self.target = TRONformerPolicy(input_dim=obs_dim, d_model=d_model, n_layers=n_layers, shade_bins=shade_bins).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))

        # Reward model
        self.reward_model = RewardModel(input_dim=RM_INPUT_DIM).to(self.device)
        self.rm_optimizer = optim.Adam(self.reward_model.parameters(), lr=RM_LR)
        self.rm_loss_fn = nn.BCELoss()
        self.rm_scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))

        self._grad_steps = 0
        self._env_steps = 0

        # Rolling obs buffer for action selection
        self._obs_deque: collections.deque = collections.deque(maxlen=seq_len)

    @property
    def epsilon(self) -> float:
        frac = min(1.0, self._env_steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    @property
    def tau(self) -> float:
        """Current Boltzmann temperature (linearly decayed from tau_start to tau_end)."""
        frac = min(1.0, self._env_steps / self.eps_decay_steps)
        return self.tau_start + frac * (self.tau_end - self.tau_start)

    def reset_obs_buffer(self):
        """Clear the rolling obs buffer (call at episode start)."""
        self._obs_deque.clear()

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Action selection using the rolling obs buffer.

        Uses Boltzmann (temperature-scaled softmax) if self.boltzmann=True,
        otherwise epsilon-greedy. Appends obs to the internal deque, stacks
        to a sequence, and queries the online policy at the last position.

        Args:
            obs: (obs_dim,) numpy array — current observation.

        Returns:
            action: [shade_idx, eta_idx] int64 array.
        """
        self._obs_deque.append(obs.astype(np.float32))
        obs_seq = np.stack(list(self._obs_deque))  # (cur_len, obs_dim)
        obs_t = torch.tensor(obs_seq, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, cur_len, D)

        with torch.no_grad():
            Q_shade, Q_eta = self.online(obs_t)  # (1, cur_len, ...)

        # Q_shade / Q_eta are (1, n_bins) from CLS token
        if self.boltzmann:
            t = self.tau
            shade_probs = torch.softmax(Q_shade[0, :] / t, dim=-1)
            eta_probs   = torch.softmax(Q_eta[0, :]   / t, dim=-1)
            shade_idx = int(torch.multinomial(shade_probs, 1).item())
            eta_idx   = int(torch.multinomial(eta_probs,   1).item())
        elif random.random() < self.epsilon:
            shade_idx = random.randrange(len(self.online.SHADE_BINS))
            eta_idx   = random.randrange(len(self.online.ETA_BINS))
        else:
            shade_idx = int(Q_shade[0, :].argmax().item())
            eta_idx   = int(Q_eta[0, :].argmax().item())

        self._env_steps += 1
        return np.array([shade_idx, eta_idx], dtype=np.int64)

    def update(self, buffer: TronformerReplayBuffer) -> Optional[float]:
        """Sample a batch of transitions and perform one Double-DQN gradient step.

        Each stored transition carries a pre-computed n-step return G_t and a
        bootstrap state (next_obs_seq) so training is a standard single-step
        DQN update.  Q-values are derived from the CLS token — one value per
        sequence, not one per timestep.

        Returns:
            loss (float) or None if not enough data.
        """
        if len(buffer) < self.batch_size:
            return None

        obs, acts, rews, next_obs, dones = buffer.sample(self.batch_size)
        # obs:      (B, T, obs_dim)
        # acts:     (B, 2)
        # rews:     (B,)   — n-step discounted return G_t
        # next_obs: (B, T, obs_dim)
        # dones:    (B,)   bool
        obs      = obs.to(self.device)
        acts     = acts.to(self.device)
        rews     = rews.to(self.device)
        next_obs = next_obs.to(self.device)
        dones    = dones.to(self.device)

        # ── Online Q-values from CLS ───────────────────────────────────────
        self.online.train()
        with torch.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
            Q_shade_online, Q_eta_online = self.online(obs)   # (B, n_shade), (B, n_eta)

            shade_taken = acts[:, 0].long()   # (B,)
            eta_taken   = acts[:, 1].long()

            Q_shade_taken = Q_shade_online.gather(1, shade_taken.unsqueeze(1)).squeeze(1)  # (B,)
            Q_eta_taken   = Q_eta_online.gather(1, eta_taken.unsqueeze(1)).squeeze(1)

            # ── Double DQN targets ─────────────────────────────────────────────
            with torch.no_grad():
                Q_shade_next_on, Q_eta_next_on = self.online(next_obs)   # (B, n_*)
                Q_shade_next_tg, Q_eta_next_tg = self.target(next_obs)

                # Online selects action; target evaluates value
                shade_next_act = Q_shade_next_on.argmax(dim=-1)  # (B,)
                eta_next_act   = Q_eta_next_on.argmax(dim=-1)

                Q_shade_next = Q_shade_next_tg.gather(
                    1, shade_next_act.unsqueeze(1)
                ).squeeze(1)   # (B,)
                Q_eta_next   = Q_eta_next_tg.gather(
                    1, eta_next_act.unsqueeze(1)
                ).squeeze(1)

                not_done = (~dones).float()
                gamma_n  = self.gamma ** self.n_step

                shade_target = rews + gamma_n * Q_shade_next * not_done   # (B,)
                eta_target   = rews + gamma_n * Q_eta_next   * not_done

            # ── Loss ──────────────────────────────────────────────────────────
            loss_shade = self.loss_fn(Q_shade_taken, shade_target.detach())
            loss_eta   = self.loss_fn(Q_eta_taken,   eta_target.detach())
            loss = loss_shade + loss_eta

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()

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
        ).to(self.device)
        labels = torch.tensor(
            [float(rm_data[i][1]) for i in indices], dtype=torch.float32
        ).to(self.device)

        self.reward_model.train()
        with torch.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
            preds = self.reward_model(feats)
            rm_loss = self.rm_loss_fn(preds, labels)

        self.rm_optimizer.zero_grad()
        self.rm_scaler.scale(rm_loss).backward()
        self.rm_scaler.step(self.rm_optimizer)
        self.rm_scaler.update()
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
        feat_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return float(self.reward_model(feat_t).item())

    def save(self, path: str):
        torch.save(self.online.state_dict(), path)
        print(f"  Saved model → {path}")

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.online.load_state_dict(state)
        self.target.load_state_dict(state)
        self.online.train()
        self.target.eval()


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate(
    policy: TRONformerPolicy,
    n_episodes: int = EVAL_EPISODES,
    seq_len: int = SEQ_LEN,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """Roll out the greedy policy deterministically.

    Returns:
        (mean_reward, std_reward) over n_episodes episodes.
    """
    device = device or DEVICE
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
            obs_t = torch.tensor(obs_seq, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                Q_shade, Q_eta = policy(obs_t)   # (1, n_bins) from CLS
            shade_idx = int(Q_shade[0, :].argmax().item())
            eta_idx   = int(Q_eta[0, :].argmax().item())
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

    device = DEVICE
    print(f"Device: {device}")

    seq_len  = args.seq_len
    batch_sz    = args.batch_size
    d_model     = args.d_model
    total_steps = args.timesteps

    shade_bins = ENV_KWARGS.get("shade_bins", UNIFORM_SHADE_BINS)

    trainer  = R2D2FormerTrainer(
        obs_dim=14,
        d_model=d_model,
        n_layers=args.n_layers,
        seq_len=seq_len,
        lr=args.lr,
        n_step=args.n_step,
        batch_size=batch_sz,
        eps_end=args.eps_end,
        shade_bins=shade_bins,
        eps_decay_steps=args.eps_decay_steps,
        boltzmann=args.boltzmann,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        device=device,
    )
    buffer    = TronformerReplayBuffer(seq_len=seq_len)
    nstep_buf = NStepBuffer(n_step=args.n_step, gamma=GAMMA)

    if args.load:
        print(f"Resuming from checkpoint: {args.load}")
        trainer.load(args.load)

    eval_csv        = os.path.join(run_dir, "eval_rewards.csv")
    best_model_path = os.path.join(run_dir, "best_model.pt")
    best_mean = -np.inf

    env = TRONEnv(**ENV_KWARGS)
    obs, _ = env.reset()
    trainer.reset_obs_buffer()

    ep_reward  = 0.0
    ep_rewards: list = []
    step = args.start_step
    trainer._env_steps = args.start_step
    next_eval = ((args.start_step // EVAL_FREQ) + 1) * EVAL_FREQ
    losses: list = []

    print(f"\nTraining TRONformer (CLS-DQN) for {total_steps:,} steps")
    print(f"  seq_len={seq_len}, n_step={args.n_step}, batch={batch_sz}")
    print(f"  d_model={d_model}, n_layers={args.n_layers}, shade_bins={len(shade_bins)}")
    if args.boltzmann:
        print(f"  exploration=Boltzmann(tau={args.tau_start}→{args.tau_end})")
    else:
        print(f"  exploration=eps-greedy(eps_end={args.eps_end})")
    print(f"  eps_decay_steps={args.eps_decay_steps}, train_freq={TRAIN_FREQ}")
    print(f"  buffer_capacity={BUFFER_CAPACITY}")
    print(f"  env sim_time={ENV_KWARGS['sim_time']}, lam={ENV_KWARGS['lam']}, "
          f"lam_zi={ENV_KWARGS['lam_zi']}\n")

    t0 = time.time()

    obs_dim = obs.shape[0]

    def _pad_obs_seq(seq: np.ndarray) -> np.ndarray:
        """Zero-pad a (T, obs_dim) array to (seq_len, obs_dim)."""
        T = seq.shape[0]
        if T == seq_len:
            return seq
        pad = np.zeros((seq_len - T, obs_dim), dtype=seq.dtype)
        return np.concatenate([pad, seq], axis=0)

    def _peek_next_obs_seq(deque: collections.deque, next_obs: np.ndarray) -> np.ndarray:
        """Compute what the rolling obs_seq will look like after adding next_obs,
        without mutating the deque."""
        window = list(deque) + [next_obs.astype(np.float32)]
        if len(window) > seq_len:
            window = window[-seq_len:]
        return _pad_obs_seq(np.stack(window))

    while step < total_steps:
        # select_action appends obs to trainer._obs_deque and returns action
        action    = trainer.select_action(obs)
        obs_seq_t = _pad_obs_seq(np.stack(list(trainer._obs_deque)))  # (seq_len, obs_dim)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        ep_reward += reward
        step += 1

        # next_obs_seq: what the rolling context will look like at t+1
        if not done:
            next_obs_seq = _peek_next_obs_seq(trainer._obs_deque, next_obs)
        else:
            next_obs_seq = np.zeros((seq_len, obs_dim), dtype=np.float32)  # done — won't be bootstrapped

        nstep_buf.add(obs_seq_t, action, reward, done)

        # Once n transitions accumulated, emit the oldest as a complete n-step transition
        if len(nstep_buf) >= args.n_step:
            buffer.add(*nstep_buf.get(next_obs_seq))

        obs = next_obs

        if done:
            # Drain remaining partial n-step transitions (episode ended early)
            for t_drain in nstep_buf.drain(next_obs_seq):
                buffer.add(*t_drain)
            ep_rewards.append(ep_reward)
            ep_reward = 0.0
            obs, _ = env.reset()
            trainer.reset_obs_buffer()

        # Train policy
        if step % TRAIN_FREQ == 0 and len(buffer) >= LEARNING_STARTS:
            loss = trainer.update(buffer)
            if loss is not None:
                losses.append(loss)

        # Evaluate
        if step >= next_eval:
            mean_r, std_r = evaluate(trainer.online, seq_len=seq_len, device=device)
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
    device = DEVICE
    shade_bins = ENV_KWARGS.get("shade_bins", UNIFORM_SHADE_BINS)
    print(f"Loading {args.load} (device={device})")
    policy = TRONformerPolicy(input_dim=14, d_model=args.d_model, n_layers=args.n_layers, shade_bins=shade_bins).to(device)
    policy.load_state_dict(torch.load(args.load, map_location=device))
    policy.eval()

    n = args.eval_episodes
    mean_r, std_r = evaluate(policy, n_episodes=n, seq_len=args.seq_len, device=device)
    print(f"Eval ({n} episodes): mean={mean_r:+.4f}  std={std_r:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=2_000_000,
                   help="Total environment steps (default 2 000 000)")
    p.add_argument("--seq-len", type=int, default=SEQ_LEN,
                   help=f"Sequence length / rolling context (default {SEQ_LEN})")
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
                   help="Use skewed shade bins (overrides --shade-max and --n-bins)")
    p.add_argument("--shade-max", type=float, default=600.0,
                   help="Max shade value for uniform bins (default 600; ignored when --skew-bins)")
    p.add_argument("--n-bins", type=int, default=42,
                   help="Number of uniform shade bins (default 42; ignored when --skew-bins)")
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
    p.add_argument("--lr", type=float, default=LR,
                   help=f"Adam learning rate (default {LR})")
    p.add_argument("--n-layers", type=int, default=2,
                   help="Number of Pre-LN transformer blocks (default 2, per paper §4.1)")
    p.add_argument("--n-step", type=int, default=N_STEP,
                   help=f"n-step return horizon (default {N_STEP})")
    p.add_argument("--eps-end", type=float, default=EPS_END,
                   help=f"Final epsilon for epsilon-greedy (default {EPS_END})")
    p.add_argument("--boltzmann", action="store_true",
                   help="Use Boltzmann (softmax) exploration instead of epsilon-greedy")
    p.add_argument("--tau-start", type=float, default=2.0,
                   help="Initial temperature for Boltzmann exploration (default 2.0)")
    p.add_argument("--tau-end", type=float, default=0.1,
                   help="Final temperature for Boltzmann exploration (default 0.1)")
    p.add_argument("--lam", type=float, default=None,
                   help="Background market lambda (default: ENV_KWARGS value 0.005)")
    p.add_argument("--shock-var", type=float, default=None,
                   help="Fundamental shock variance (default: ENV_KWARGS value 1e6)")
    p.add_argument("--pv-var", type=float, default=None,
                   help="Private value variance (default: ENV_KWARGS value 5e6)")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    if args.n_bg is not None:
        ENV_KWARGS["num_background_agents"] = args.n_bg
    if args.lam_zi is not None:
        ENV_KWARGS["lam_zi"] = args.lam_zi
    if args.sim_time is not None:
        ENV_KWARGS["sim_time"] = args.sim_time
    if args.skew_bins:
        ENV_KWARGS["shade_bins"] = SKEWED_SHADE_BINS
    elif args.shade_max != 600.0 or args.n_bins != 42:
        ENV_KWARGS["shade_bins"] = np.linspace(0, args.shade_max, args.n_bins)
    if args.bg_strategy is not None:
        strat = _STRATEGIES[args.bg_strategy]
        ENV_KWARGS["bg_strategies"] = [{"shade": strat["shade"], "eta": strat["eta"]}]
    if args.lam is not None:
        ENV_KWARGS["lam"] = args.lam
    if args.shock_var is not None:
        ENV_KWARGS["shock_var"] = args.shock_var
    if args.pv_var is not None:
        ENV_KWARGS["pv_var"] = args.pv_var

    if args.eval_only:
        eval_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
