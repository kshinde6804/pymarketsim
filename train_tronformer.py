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
    lam_zi=0.005,          # matched to background lam → equal participation (~10 RL steps/ep)
    mean=1e5,
    r=0.01,
    shock_var=1e6,
    q_max=10,
    pv_var=5e6,
    shade=[250, 500],
    normalizers={"fundamental": 1e5, "invt": 10, "reward": 1e3, "pv": 5e5},
    bg_strategies=BG_STRATEGIES,
    shade_bins=UNIFORM_SHADE_BINS,
)

# ── R2D2 hyper-parameters ─────────────────────────────────────────────────────

N_STEP = 5
GAMMA = 0.99
BATCH_SIZE = 64              # per-sequence batches (each carries SEQ_LEN steps)
BUFFER_CAPACITY = 50_000     # sequences; each is SEQ_LEN steps (~50 k × 40 = 2 M transitions)
BURNIN = 0                   # positions at seq start with no loss (no LSTM warmup needed)
LR = 1e-4
TARGET_UPDATE_FREQ = 1000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 1_000_000
GRAD_CLIP = 10.0
LEARNING_STARTS = 64         # minimum sequences in buffer before training
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


# ── Per-sequence Replay Buffer ────────────────────────────────────────────────


class TronformerSequenceBuffer:
    """Per-sequence replay buffer for TRONformer.

    Each stored entry is a fixed-length (seq_len) sequence of raw transitions:
        obs_seq:  (seq_len, obs_dim)  float32  — raw observations in time order
        act_seq:  (seq_len, 2)        int64    — [shade_idx, eta_idx] per step
        rew_seq:  (seq_len,)          float32  — single-step rewards
        done_seq: (seq_len,)          bool     — episode-terminal flags

    N-step returns and Double-DQN targets are computed on-the-fly during
    training (inside R2D2FormerTrainer.update), using the target network's
    Q-values at the bootstrap position within the same sequence.

    Sequences span episode boundaries (episodes ≈ 10 steps at lam_zi=0.005
    < seq_len=40), matching TRON's cross-episode sequence design.

    Args:
        capacity: Maximum number of sequences stored (circular buffer).
        seq_len:  Number of timesteps per sequence.
        obs_dim:  Observation dimensionality.
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
        obs_seq: np.ndarray,   # (seq_len, obs_dim)
        act_seq: np.ndarray,   # (seq_len, 2)
        rew_seq: np.ndarray,   # (seq_len,)
        done_seq: np.ndarray,  # (seq_len,) bool
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
        rews  = torch.tensor(np.stack([b[2] for b in batch]), dtype=torch.float32)
        dones = torch.tensor(np.stack([b[3] for b in batch]))

        return obs, acts, rews, dones


# ── Sequence Collector ────────────────────────────────────────────────────────


class SequenceCollector:
    """Accumulates env transitions and emits fixed-length sequences.

    Accumulates (obs, action, reward, done) tuples.  Once seq_len transitions
    have been collected, ``pop_sequence()`` emits a (obs_seq, act_seq,
    rew_seq, done_seq) tuple and clears the buffer for the next sequence.
    Sequences are non-overlapping, maximising diversity in the replay buffer.

    Sequences span episode boundaries by design (episodes ≈ 10 steps at
    lam_zi=0.005, shorter than seq_len=40).  The done flag encodes episode
    boundaries so n-step return computation in the training update can stop
    accumulating at done=True steps.

    Args:
        seq_len: Fixed sequence length.
        obs_dim: Observation dimension.
    """

    def __init__(self, seq_len: int, obs_dim: int = 14):
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self._obs:  list = []
        self._acts: list = []
        self._rews: list = []
        self._done: list = []

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, done: bool):
        self._obs.append(obs.astype(np.float32))
        self._acts.append(action.astype(np.int64).copy())
        self._rews.append(float(reward))
        self._done.append(bool(done))

    def is_full(self) -> bool:
        return len(self._obs) >= self.seq_len

    def pop_sequence(self) -> Optional[tuple]:
        """Emit the oldest seq_len transitions and slide the window forward.

        Returns None if fewer than seq_len transitions have been collected.
        """
        if len(self._obs) < self.seq_len:
            return None
        obs_seq  = np.stack(self._obs[:self.seq_len])           # (T, obs_dim)
        act_seq  = np.stack(self._acts[:self.seq_len])          # (T, 2)
        rew_seq  = np.array(self._rews[:self.seq_len], np.float32)
        done_seq = np.array(self._done[:self.seq_len], bool)
        del self._obs[:self.seq_len]
        del self._acts[:self.seq_len]
        del self._rews[:self.seq_len]
        del self._done[:self.seq_len]
        return obs_seq, act_seq, rew_seq, done_seq


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
        burnin: int = BURNIN,
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
        self.burnin = burnin
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
        self.loss_fn = nn.SmoothL1Loss(reduction="none")  # per-element, for valid masking
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
        to a left-padded sequence of length seq_len (matching the replay buffer
        and TRONformerAgent deployment format), and queries the online policy.

        Args:
            obs: (obs_dim,) numpy array — current observation.

        Returns:
            action: [shade_idx, eta_idx] int64 array.
        """
        self._obs_deque.append(obs.astype(np.float32))
        obs_raw = np.stack(list(self._obs_deque))  # (cur_len, obs_dim)
        # Left-pad to seq_len to match replay buffer and TRONformerAgent format
        obs_seq = np.zeros((self.seq_len, obs_raw.shape[1]), dtype=np.float32)
        obs_seq[self.seq_len - len(obs_raw):] = obs_raw
        obs_t = torch.tensor(obs_seq, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, seq_len, D)

        with torch.no_grad():
            Q_shade, Q_eta = self.online(obs_t)  # (1, 1+seq_len, n_shade), (1, 1+seq_len, n_eta)

        # Read Q from the [CLS] token at position 0 (§4.1 "Policy and value heads")
        if self.boltzmann:
            t = self.tau
            shade_probs = torch.softmax(Q_shade[0, 0, :] / t, dim=-1)
            eta_probs   = torch.softmax(Q_eta[0, 0, :]   / t, dim=-1)
            shade_idx = int(torch.multinomial(shade_probs, 1).item())
            eta_idx   = int(torch.multinomial(eta_probs,   1).item())
        elif random.random() < self.epsilon:
            shade_idx = random.randrange(len(self.online.SHADE_BINS))
            eta_idx   = random.randrange(len(self.online.ETA_BINS))
        else:
            shade_idx = int(Q_shade[0, 0, :].argmax().item())
            eta_idx   = int(Q_eta[0, 0, :].argmax().item())

        self._env_steps += 1
        return np.array([shade_idx, eta_idx], dtype=np.int64)

    def update(self, buffer: "TronformerSequenceBuffer") -> Optional[float]:
        """Sample a batch of sequences and perform one per-sequence Double-DQN step.

        Each stored sequence contains seq_len consecutive (obs, action, reward,
        done) transitions spanning episode boundaries.  A single forward pass
        over the full (B, T, obs_dim) batch produces Q-values at every position.
        N-step returns are computed on-the-fly using per-step rewards and
        bootstrap Q-values from the target network at position t+n_step, all
        within the same sequence.

        Learning window: [burnin, T - n_step) — avoids positions with
        insufficient context (burnin) and those lacking a bootstrap state
        within the sequence (last n_step positions).

        Returns:
            loss (float) or None if buffer too small.
        """
        if len(buffer) < self.batch_size:
            return None

        obs_seqs, act_seqs, rew_seqs, done_seqs = buffer.sample(self.batch_size)
        # obs_seqs:  (B, T, obs_dim)
        # act_seqs:  (B, T, 2)       int64
        # rew_seqs:  (B, T)          float32
        # done_seqs: (B, T)          bool
        B, T, _ = obs_seqs.shape
        n   = self.n_step
        b   = self.burnin
        L   = T - b - n          # learning window length
        assert L > 0, f"seq_len={T} too short for burnin={b} + n_step={n}"

        obs_seqs  = obs_seqs.to(self.device)
        act_seqs  = act_seqs.to(self.device)
        rew_seqs  = rew_seqs.to(self.device)
        done_seqs = done_seqs.to(self.device)

        self.online.train()
        with torch.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
            # ── Online + target forward passes ─────────────────────────────
            # Returns (B, T, n_shade) and (B, T, n_eta)
            Q_shade_on, Q_eta_on = self.online(obs_seqs)

            with torch.no_grad():
                Q_shade_tg, Q_eta_tg = self.target(obs_seqs)

                # ── N-step returns for learning window [b, b+L) ────────────
                # rews_window: (B, T-b) — rewards from position b onward
                rews_w = rew_seqs[:, b:]   # (B, T-b)
                done_w = done_seqs[:, b:]  # (B, T-b)

                # Accumulate discounted rewards; stop on done (but include
                # the done-step reward, matching NStepBuffer.get() behaviour).
                nstep_rets = torch.zeros(B, L, device=self.device)
                done_flags = torch.zeros(B, L, dtype=torch.bool, device=self.device)
                mask = torch.ones(B, L, device=self.device)  # 1 = still accumulating

                for k in range(n):
                    r_k = rews_w[:, k:k + L]          # (B, L)
                    d_k = done_w[:, k:k + L].bool()   # (B, L)
                    nstep_rets += (self.gamma ** k) * r_k * mask
                    done_flags  = done_flags | (d_k & mask.bool())
                    mask        = mask * (~d_k).float()

                # Bootstrap: Double DQN — online selects action, target evaluates.
                # forward() returns (B, 1+T, n_bins): position 0 is CLS, positions
                # 1..T correspond to obs time-steps 0..T-1.  Obs at raw index t maps
                # to output index t+1, so all slices are shifted by +1 vs. the old
                # (no-CLS) code.
                boot_slice = slice(b + n + 1, b + n + L + 1)   # obs positions [b+n, b+n+L)
                Q_shade_on_boot = Q_shade_on[:, boot_slice, :].detach()  # (B, L, n_shade)
                Q_eta_on_boot   = Q_eta_on[:, boot_slice, :].detach()

                shade_boot_act = Q_shade_on_boot.argmax(dim=-1)  # (B, L)
                eta_boot_act   = Q_eta_on_boot.argmax(dim=-1)

                Q_shade_boot = Q_shade_tg[:, boot_slice, :]   # (B, L, n_shade)
                Q_eta_boot   = Q_eta_tg[:, boot_slice, :]

                Q_shade_next = Q_shade_boot.gather(
                    2, shade_boot_act.unsqueeze(-1)
                ).squeeze(-1)   # (B, L)
                Q_eta_next   = Q_eta_boot.gather(
                    2, eta_boot_act.unsqueeze(-1)
                ).squeeze(-1)

                gamma_n = self.gamma ** n
                not_done = (~done_flags).float()
                shade_target = nstep_rets + gamma_n * Q_shade_next * not_done  # (B, L)
                eta_target   = nstep_rets + gamma_n * Q_eta_next   * not_done

            # ── Loss over learning window ───────────────────────────────────
            # Obs at raw index t is at output position t+1 (CLS offset).
            # Learning window covers raw indices [b, b+L) → output positions [b+1, b+L+1).
            learn_slice = slice(b + 1, b + L + 1)
            shade_taken = act_seqs[:, b:b + L, 0]  # (B, L)  — raw action indices unchanged
            eta_taken   = act_seqs[:, b:b + L, 1]

            Q_shade_learn = Q_shade_on[:, learn_slice, :]  # (B, L, n_shade)
            Q_eta_learn   = Q_eta_on[:, learn_slice, :]

            Q_shade_taken = Q_shade_learn.gather(
                2, shade_taken.unsqueeze(-1)
            ).squeeze(-1)   # (B, L)
            Q_eta_taken   = Q_eta_learn.gather(
                2, eta_taken.unsqueeze(-1)
            ).squeeze(-1)

            loss_shade = self.loss_fn(Q_shade_taken, shade_target.detach()).mean()
            loss_eta   = self.loss_fn(Q_eta_taken,   eta_target.detach()).mean()
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
            obs_raw = np.stack(list(obs_deque))
            # Left-pad to seq_len to match replay buffer and TRONformerAgent format
            obs_seq = np.zeros((seq_len, obs_raw.shape[1]), dtype=np.float32)
            obs_seq[seq_len - len(obs_raw):] = obs_raw
            obs_t = torch.tensor(obs_seq, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                Q_shade, Q_eta = policy(obs_t)   # (1, 1+seq_len, n_bins)
            # Read from [CLS] token at position 0 (§4.1 "Policy and value heads")
            shade_idx = int(Q_shade[0, 0, :].argmax().item())
            eta_idx   = int(Q_eta[0, 0, :].argmax().item())
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
        burnin=BURNIN,
        batch_size=batch_sz,
        eps_end=args.eps_end,
        shade_bins=shade_bins,
        eps_decay_steps=args.eps_decay_steps,
        boltzmann=args.boltzmann,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        device=device,
    )
    buffer    = TronformerSequenceBuffer(seq_len=seq_len)
    seq_col   = SequenceCollector(seq_len=seq_len, obs_dim=14)

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

    print(f"\nTraining TRONformer (last-pos causal DQN) for {total_steps:,} steps")
    print(f"  seq_len={seq_len}, burnin={BURNIN}, n_step={args.n_step}, batch={batch_sz}")
    print(f"  d_model={d_model}, n_layers={args.n_layers}, shade_bins={len(shade_bins)}")
    if args.boltzmann:
        print(f"  exploration=Boltzmann(tau={args.tau_start}→{args.tau_end})")
    else:
        print(f"  exploration=eps-greedy(eps_end={args.eps_end})")
    print(f"  eps_decay_steps={args.eps_decay_steps}, train_freq={TRAIN_FREQ}")
    print(f"  buffer_capacity={BUFFER_CAPACITY} sequences ({BUFFER_CAPACITY * seq_len:,} transitions)")
    print(f"  env sim_time={ENV_KWARGS['sim_time']}, lam={ENV_KWARGS['lam']}, "
          f"lam_zi={ENV_KWARGS['lam_zi']}\n")

    t0 = time.time()

    while step < total_steps:
        # select_action appends obs to trainer._obs_deque for inference context
        action   = trainer.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        ep_reward += reward
        step += 1

        # Collector stores raw (obs, action, reward, done) — no pre-padded sequences
        seq_col.add(obs, action, reward, done)
        obs = next_obs

        # Emit a complete sequence to the replay buffer every seq_len steps
        if seq_col.is_full():
            buffer.add(*seq_col.pop_sequence())

        if done:
            ep_rewards.append(ep_reward)
            ep_reward = 0.0
            obs, _ = env.reset()
            trainer.reset_obs_buffer()

        # Train policy every TRAIN_FREQ steps once buffer has enough sequences
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
