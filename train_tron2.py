"""
train_tron2.py — R2D2 training for the TRON2 agent (ICAIF24, §5.3).

Trains a TRONPolicy2 (Dueling DQN + LSTM) via episode-based R2D2:
  - Episode replay buffer (full episodes stored as padded sequences)
  - n-step returns (n=N_STEP)
  - Target network (Polyak soft update, TAU=0.005, every gradient step)
  - ε-greedy exploration (linear decay)
  - Periodic evaluation checkpointing

Usage:
    python train_tron2.py --env-tag B --ne-strategy 8 --tag v1
    python train_tron2.py --env-tag C --ne-strategy 8 --tag v1 --episodes 3000000
    python train_tron2.py --eval-only --load runs/tron2_v1/best_model.pt --env-tag B
"""

import argparse
import copy
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from marketsim.agent.tron2_agent import TRONPolicy2, N_SHADE, N_ETA
from marketsim.wrappers.tron2_env import TRONEnv2, ENV_CONFIGS, STRATEGIES

# ── Hyperparameters ────────────────────────────────────────────────────────────
HIDDEN_DIM      = 128
BATCH_SIZE      = 32        # episodes per gradient step
REPLAY_CAPACITY = 50_000    # max episodes in buffer
MIN_BUFFER_EPS  = 1_000     # warmup: start learning after this many episodes
TRAIN_FREQ      = 1         # gradient steps per episode (after warmup)
TAU             = 0.005     # Polyak soft update coefficient
GAMMA           = 0.99
N_STEP          = 5
LR              = 1e-4
GRAD_CLIP       = 10.0      # matches Dueling DQN §4.2 and train_tron.py (was 40.0)
EPS_START       = 1.0
EPS_END         = 0.05
EPS_DECAY_EPS   = 500_000   # episodes over which ε decays linearly
EVAL_FREQ       = 100_000   # episodes between checkpoint evals
EVAL_RUNS       = 200       # simulations per checkpoint eval
LOG_FREQ        = 10_000    # episodes between log prints


# ── Replay buffer ──────────────────────────────────────────────────────────────

class EpisodeReplayBuffer:
    """Circular buffer storing full episodes as variable-length sequences."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: list = []
        self.pos    = 0

    def add(self, obs, actions, rewards, dones):
        """
        obs:     list/array (T, 14)  float
        actions: list/array (T, 2)   int
        rewards: list/array (T,)     float
        dones:   list/array (T,)     bool  — last element is True
        """
        ep = {
            "obs":     np.array(obs,     dtype=np.float32),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "dones":   np.array(dones,   dtype=bool),
        }
        if len(self.buffer) < self.capacity:
            self.buffer.append(ep)
        else:
            self.buffer[self.pos] = ep
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, device: torch.device):
        """Sample `batch_size` episodes; pad to the longest in the batch."""
        indices  = random.sample(range(len(self.buffer)), min(batch_size, len(self.buffer)))
        episodes = [self.buffer[i] for i in indices]
        B        = len(episodes)
        max_T    = max(len(e["rewards"]) for e in episodes)

        batch_obs     = np.zeros((B, max_T, 14), dtype=np.float32)
        batch_actions = np.zeros((B, max_T, 2),  dtype=np.int64)
        batch_rewards = np.zeros((B, max_T),      dtype=np.float32)
        batch_dones   = np.zeros((B, max_T),      dtype=bool)
        batch_mask    = np.zeros((B, max_T),      dtype=bool)

        for i, ep in enumerate(episodes):
            T = len(ep["rewards"])
            batch_obs[i,     :T] = ep["obs"]
            batch_actions[i, :T] = ep["actions"]
            batch_rewards[i, :T] = ep["rewards"]
            batch_dones[i,   :T] = ep["dones"]
            batch_mask[i,    :T] = True

        return {
            "obs":     torch.tensor(batch_obs,     device=device),
            "actions": torch.tensor(batch_actions, device=device),
            "rewards": torch.tensor(batch_rewards, device=device),
            "dones":   torch.tensor(batch_dones,   device=device),
            "mask":    torch.tensor(batch_mask,    device=device),
        }

    def __len__(self):
        return len(self.buffer)


# ── Loss computation ───────────────────────────────────────────────────────────

def _nstep_returns(rewards, V_bootstrap, dones, mask, gamma, n_step):
    """Compute n-step discounted returns for every valid timestep.

    Args:
        rewards:     (B, T) float tensor
        V_bootstrap: (B, T) float tensor — max Q-values from target net
        dones:       (B, T) bool  tensor
        mask:        (B, T) bool  tensor
        gamma:       float
        n_step:      int

    Returns:
        G: (B, T) float tensor
    """
    B, T     = rewards.shape
    not_done = (~dones).float()
    mask_f   = mask.float()
    G        = torch.zeros_like(rewards)

    # Tracks product of ~dones[t] * ... * ~dones[t+k-1] as k increases.
    # All-ones at k=0 (no survival condition yet).
    not_done_prod = torch.ones(B, T, device=rewards.device)

    for k in range(n_step):
        if k > 0:
            nd_prev = torch.cat([not_done[:, k-1:], not_done.new_zeros(B, k-1)], dim=1)
            not_done_prod = not_done_prod * nd_prev
        r_k = rewards if k == 0 else torch.cat([rewards[:, k:], rewards.new_zeros(B, k)], dim=1)
        m_k = mask_f  if k == 0 else torch.cat([mask_f[:, k:],  mask_f.new_zeros(B, k)],  dim=1)
        G  += (gamma ** k) * r_k * m_k * not_done_prod

    # Bootstrap: gamma^n * V[t+n] * mask[t+n] * ~dones[t:t+n+1].all()
    # not_done_prod has n_step-1 terms after loop; multiply two more to reach n_step+1.
    nd_b1  = torch.cat([not_done[:, n_step-1:], not_done.new_zeros(B, n_step-1)], dim=1)
    nd_b2  = torch.cat([not_done[:, n_step:],   not_done.new_zeros(B, n_step)],   dim=1)
    V_boot = torch.cat([V_bootstrap[:, n_step:], V_bootstrap.new_zeros(B, n_step)], dim=1)
    m_boot = torch.cat([mask_f[:, n_step:],      mask_f.new_zeros(B, n_step)],      dim=1)
    G += (gamma ** n_step) * V_boot * m_boot * (not_done_prod * nd_b1 * nd_b2)

    return G


def compute_loss(batch, online_net, target_net, gamma, n_step, device):
    """R2D2 loss: separate TD loss for shade head and eta head."""
    obs     = batch["obs"]                    # (B, T, 14)
    actions = batch["actions"]                # (B, T, 2)
    rewards = batch["rewards"]                # (B, T)
    dones   = batch["dones"]                  # (B, T)
    mask    = batch["mask"]                   # (B, T)

    # Forward online network (with grad) — used for both action selection and Q_taken
    Q_shade_on, Q_eta_on, _ = online_net(obs)       # (B, T, 21)

    # Forward target network (no grad) — evaluates the actions online selects (Double DQN)
    with torch.no_grad():
        Q_shade_tg, Q_eta_tg, _ = target_net(obs)   # (B, T, 21)

    # Double DQN bootstrap: online selects greedy action, target evaluates it
    shade_act_sel = Q_shade_on.detach().argmax(dim=-1, keepdim=True)  # (B, T, 1)
    eta_act_sel   = Q_eta_on.detach().argmax(  dim=-1, keepdim=True)  # (B, T, 1)
    V_shade_tg    = Q_shade_tg.gather(2, shade_act_sel).squeeze(-1)   # (B, T)
    V_eta_tg      = Q_eta_tg.gather(  2, eta_act_sel  ).squeeze(-1)   # (B, T)

    # Zero bootstrap at terminal states
    V_shade_tg = V_shade_tg * (~dones).float()
    V_eta_tg   = V_eta_tg   * (~dones).float()

    # n-step returns (same reward stream, different bootstrap per head)
    G_shade = _nstep_returns(rewards, V_shade_tg, dones, mask, gamma, n_step)
    G_eta   = _nstep_returns(rewards, V_eta_tg,   dones, mask, gamma, n_step)

    # Gather Q values for the actions taken
    a_shade = actions[:, :, 0].unsqueeze(-1)        # (B, T, 1)
    a_eta   = actions[:, :, 1].unsqueeze(-1)        # (B, T, 1)
    Q_shade_taken = Q_shade_on.gather(2, a_shade).squeeze(-1)  # (B, T)
    Q_eta_taken   = Q_eta_on.gather(  2, a_eta  ).squeeze(-1)  # (B, T)

    # Huber loss (smooth_l1), masked over valid steps
    mask_f  = mask.float()
    n_valid = mask_f.sum().clamp(min=1)

    loss_shade = (F.smooth_l1_loss(Q_shade_taken, G_shade.detach(), reduction='none') * mask_f).sum() / n_valid
    loss_eta   = (F.smooth_l1_loss(Q_eta_taken,   G_eta.detach(),   reduction='none') * mask_f).sum() / n_valid

    return loss_shade + loss_eta


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(policy, env_tag, ne_strategy, n_runs, device):
    """Run n_runs greedy episodes; return mean raw profit."""
    env    = TRONEnv2(env_tag=env_tag, ne_strategy=ne_strategy)
    policy.eval()
    profits = []

    for _ in range(n_runs):
        obs, _ = env.reset()
        h_c    = None
        done   = False

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                Q_shade, Q_eta, h_c = policy(obs_t, h_c)
            shade_idx = int(Q_shade.squeeze().argmax().item())
            eta_idx   = int(Q_eta.squeeze().argmax().item())
            obs, _, terminated, truncated, _ = env.step([shade_idx, eta_idx])
            done = terminated or truncated

        profits.append(env.get_episode_profit())

    policy.train()
    return float(np.mean(profits))


# ── Main training loop ─────────────────────────────────────────────────────────

def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Output directory
    save_dir = os.path.join("runs", f"tron2_{args.tag}")
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "train_log.csv")

    # Environment
    env = TRONEnv2(env_tag=args.env_tag, ne_strategy=args.ne_strategy)
    print(
        f"Env {args.env_tag} | strategy S{args.ne_strategy} "
        f"| shade={env.bg_shade} eta={env.bg_eta}"
    )

    # Networks
    online_net = TRONPolicy2(input_dim=14, hidden_dim=HIDDEN_DIM).to(device)
    target_net = copy.deepcopy(online_net).to(device)
    target_net.eval()

    optimizer = torch.optim.Adam(online_net.parameters(), lr=LR)
    buffer    = EpisodeReplayBuffer(capacity=REPLAY_CAPACITY)

    # ε schedule
    eps_step    = (EPS_START - EPS_END) / EPS_DECAY_EPS
    start_ep    = args.start_episode
    epsilon     = max(EPS_END, EPS_START - eps_step * (start_ep - 1))
    grad_step   = 0
    best_profit = -float("inf")

    if args.load is not None:
        online_net.load_state_dict(torch.load(args.load, map_location=device, weights_only=True))
        target_net.load_state_dict(online_net.state_dict())
        print(f"Loaded weights from {args.load}")

    log_mode = "a" if start_ep > 1 else "w"
    with open(log_path, log_mode, newline="") as f:
        writer = csv.writer(f)
        if log_mode == "w":
            writer.writerow(["episode", "epsilon", "loss", "eval_profit"])

    t0 = time.time()
    print(f"Training for {args.episodes:,} episodes (starting at {start_ep:,}) …")

    for ep in range(start_ep, args.episodes + 1):
        # ── Collect one episode ────────────────────────────────────────────────
        obs, _ = env.reset()
        h_c    = None
        ep_obs, ep_actions, ep_rewards, ep_dones = [], [], [], []
        done   = False

        while not done:
            if random.random() < epsilon:
                shade_idx = random.randint(0, N_SHADE - 1)
                eta_idx   = random.randint(0, N_ETA - 1)
            else:
                obs_t = (
                    torch.tensor(obs, dtype=torch.float32)
                    .unsqueeze(0).unsqueeze(0).to(device)
                )
                with torch.no_grad():
                    Q_shade, Q_eta, h_c = online_net(obs_t, h_c)
                shade_idx = int(Q_shade.squeeze().argmax().item())
                eta_idx   = int(Q_eta.squeeze().argmax().item())

            next_obs, reward, terminated, truncated, _ = env.step([shade_idx, eta_idx])
            done = terminated or truncated

            ep_obs.append(obs)
            ep_actions.append([shade_idx, eta_idx])
            ep_rewards.append(reward)
            ep_dones.append(done)
            obs = next_obs

        buffer.add(ep_obs, ep_actions, ep_rewards, ep_dones)
        epsilon = max(EPS_END, epsilon - eps_step)

        # ── Learn ─────────────────────────────────────────────────────────────
        loss_val = float("nan")
        if len(buffer) >= MIN_BUFFER_EPS:
            batch     = buffer.sample(BATCH_SIZE, device)
            loss      = compute_loss(batch, online_net, target_net, GAMMA, N_STEP, device)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online_net.parameters(), GRAD_CLIP)
            optimizer.step()
            loss_val = loss.item()
            grad_step += 1

            # Polyak soft target update every gradient step
            with torch.no_grad():
                for p_on, p_tgt in zip(online_net.parameters(), target_net.parameters()):
                    p_tgt.data.mul_(1.0 - TAU).add_(TAU * p_on.data)

        # ── Save latest ────────────────────────────────────────────────────────
        if ep % LOG_FREQ == 0:
            torch.save(online_net.state_dict(), os.path.join(save_dir, "latest_model.pt"))

        # ── Periodic eval & checkpoint ─────────────────────────────────────────
        eval_profit = float("nan")
        if ep % EVAL_FREQ == 0:
            eval_profit = evaluate(online_net, args.env_tag, args.ne_strategy, EVAL_RUNS, device)
            if eval_profit > best_profit:
                best_profit = eval_profit
                torch.save(online_net.state_dict(), os.path.join(save_dir, "best_model.pt"))
                print(f"  [ep {ep:>9,}] new best eval profit: {best_profit:.1f}")

        # ── Log ────────────────────────────────────────────────────────────────
        if ep % LOG_FREQ == 0:
            elapsed = time.time() - t0
            eps_per_sec = (ep - start_ep + 1) / max(elapsed, 1e-6)
            print(
                f"ep {ep:>9,} | ε={epsilon:.4f} | buf={len(buffer):>6,} "
                f"| loss={loss_val:.5f} | {eps_per_sec:.0f} ep/s"
            )
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([ep, round(epsilon, 5), round(loss_val, 6),
                                        round(eval_profit, 3) if not np.isnan(eval_profit) else ""])

    # Final eval
    final_profit = evaluate(online_net, args.env_tag, args.ne_strategy, EVAL_RUNS, device)
    print(f"\nFinal eval profit (greedy, {EVAL_RUNS} runs): {final_profit:.1f}")
    if final_profit > best_profit:
        torch.save(online_net.state_dict(), os.path.join(save_dir, "best_model.pt"))
    print(f"Models saved to {save_dir}/")


def eval_only(args):
    """Load a saved model and run evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = TRONPolicy2(input_dim=14, hidden_dim=HIDDEN_DIM).to(device)
    policy.load_state_dict(torch.load(args.load, map_location=device))
    policy.eval()

    profit = evaluate(policy, args.env_tag, args.ne_strategy, EVAL_RUNS, device)
    print(f"Eval profit ({EVAL_RUNS} runs, env {args.env_tag}, S{args.ne_strategy}): {profit:.2f}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train TRON2 agent via R2D2")
    p.add_argument("--env-tag",      default="B",   choices=["A", "B", "C"],
                   help="Market environment (Table 3)")
    p.add_argument("--ne-strategy",  default=8,     type=int,
                   help="Background ZI strategy index 0–9 (default: 8 = S8)")
    p.add_argument("--episodes",     default=3_000_000, type=int,
                   help="Total training episodes (paper: 3M)")
    p.add_argument("--tag",          default="v1",
                   help="Output directory suffix: runs/tron2_<tag>/")
    p.add_argument("--seed",         default=42,    type=int)
    p.add_argument("--eval-only",    action="store_true",
                   help="Skip training; evaluate the model at --load")
    p.add_argument("--load",         default=None,
                   help="Path to .pt model weights to load (eval or resume)")
    p.add_argument("--start-episode", default=1, type=int,
                   help="Episode to resume from (set epsilon and loop start accordingly)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.eval_only:
        if args.load is None:
            raise ValueError("--eval-only requires --load <path>")
        eval_only(args)
    else:
        train(args)
