"""
train_tron2_parallel.py — Parallel R2D2 training for TRON2.

N worker processes collect episodes on CPU; the main learner process runs
gradient updates on GPU. Workers push rollouts via a shared queue; the
learner broadcasts updated weights periodically.

Architecture:
  - Workers: TRONEnv2 + TRONPolicy2 (CPU-only), ε-greedy collection
  - Learner: EpisodeReplayBuffer + GPU gradient updates + periodic eval
  - IPC: rollout_q (workers→learner), weights_q[i] (learner→worker i)

Usage:
    python train_tron2_parallel.py --env-tag C --ne-strategy 8 --tag envc_s8_v1par --workers 8
    python train_tron2_parallel.py --env-tag C --ne-strategy 8 --tag envc_s8_v1par \\
        --load runs/tron2_envc_s8_v1/latest_model.pt --start-episode 1720001 --workers 8
    python train_tron2_parallel.py --eval-only --load runs/tron2_envc_s8_v1par/best_model.pt \\
        --env-tag C
"""

import argparse
import copy
import csv
import os
import queue
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp

from marketsim.agent.tron2_agent import TRONPolicy2, N_SHADE, N_ETA
from marketsim.wrappers.tron2_env import TRONEnv2

# ── Hyperparameters ────────────────────────────────────────────────────────────
HIDDEN_DIM      = 128
BATCH_SIZE      = 32
REPLAY_CAPACITY = 50_000
MIN_BUFFER_EPS  = 1_000
TARGET_UPDATE   = 500
GAMMA           = 0.99
N_STEP          = 5
LR              = 1e-4
GRAD_CLIP       = 40.0
EPS_START       = 1.0
EPS_END         = 0.05
EPS_DECAY_EPS   = 500_000
EVAL_FREQ       = 100_000
EVAL_RUNS       = 500
LOG_FREQ        = 10_000

WEIGHT_BROADCAST_FREQ = 200   # broadcast weights every N episodes received


# ── Replay buffer ──────────────────────────────────────────────────────────────

class EpisodeReplayBuffer:
    """Circular buffer storing full episodes as variable-length sequences."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: list = []
        self.pos    = 0

    def add(self, obs, actions, rewards, dones):
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
    obs     = batch["obs"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    dones   = batch["dones"]
    mask    = batch["mask"]

    with torch.no_grad():
        Q_shade_tg, Q_eta_tg, _ = target_net(obs)

    V_shade_tg = Q_shade_tg.max(dim=-1).values * (~dones).float()
    V_eta_tg   = Q_eta_tg.max(dim=-1).values   * (~dones).float()

    G_shade = _nstep_returns(rewards, V_shade_tg, dones, mask, gamma, n_step)
    G_eta   = _nstep_returns(rewards, V_eta_tg,   dones, mask, gamma, n_step)

    Q_shade_on, Q_eta_on, _ = online_net(obs)

    a_shade       = actions[:, :, 0].unsqueeze(-1)
    a_eta         = actions[:, :, 1].unsqueeze(-1)
    Q_shade_taken = Q_shade_on.gather(2, a_shade).squeeze(-1)
    Q_eta_taken   = Q_eta_on.gather(  2, a_eta  ).squeeze(-1)

    mask_f  = mask.float()
    n_valid = mask_f.sum().clamp(min=1)

    loss_shade = ((Q_shade_taken - G_shade.detach()) ** 2 * mask_f).sum() / n_valid
    loss_eta   = ((Q_eta_taken   - G_eta.detach()  ) ** 2 * mask_f).sum() / n_valid

    return loss_shade + loss_eta


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(policy, env_tag, ne_strategy, n_runs, device):
    """Run n_runs greedy episodes; return mean paired advantage (TRON2 - shadow ZI)."""
    env = TRONEnv2(env_tag=env_tag, ne_strategy=ne_strategy)
    policy.eval()
    advantages = []

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

        advantages.append(env.get_episode_profit() - env.get_shadow_profit())

    policy.train()
    return float(np.mean(advantages))


# ── Worker ─────────────────────────────────────────────────────────────────────

def _worker_fn(worker_id, rollout_q, weights_q, env_tag, ne_strategy, base_seed):
    """Episode collection worker. Always runs on CPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    seed = base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env    = TRONEnv2(env_tag=env_tag, ne_strategy=ne_strategy)
    policy = TRONPolicy2(input_dim=14, hidden_dim=HIDDEN_DIM)
    policy.eval()

    # Block until learner sends initial weights
    msg = weights_q.get()
    if msg is None:
        return
    state_dict, epsilon = msg
    policy.load_state_dict(state_dict)

    while True:
        # Non-blocking check for updated weights or shutdown sentinel
        try:
            msg = weights_q.get_nowait()
            if msg is None:
                break
            state_dict, epsilon = msg
            policy.load_state_dict(state_dict)
        except queue.Empty:
            pass

        # Collect one episode
        obs, _ = env.reset()
        h_c    = None
        ep_obs, ep_actions, ep_rewards, ep_dones = [], [], [], []
        done   = False

        while not done:
            if random.random() < epsilon:
                shade_idx = random.randint(0, N_SHADE - 1)
                eta_idx   = random.randint(0, N_ETA - 1)
            else:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    Q_shade, Q_eta, h_c = policy(obs_t, h_c)
                shade_idx = int(Q_shade.squeeze().argmax().item())
                eta_idx   = int(Q_eta.squeeze().argmax().item())

            next_obs, reward, terminated, truncated, _ = env.step([shade_idx, eta_idx])
            done = terminated or truncated
            ep_obs.append(obs)
            ep_actions.append([shade_idx, eta_idx])
            ep_rewards.append(reward)
            ep_dones.append(done)
            obs = next_obs

        rollout_q.put({
            "obs":     np.array(ep_obs,     dtype=np.float32),
            "actions": np.array(ep_actions, dtype=np.int64),
            "rewards": np.array(ep_rewards, dtype=np.float32),
            "dones":   np.array(ep_dones,   dtype=bool),
        }, block=True)


# ── Learner ────────────────────────────────────────────────────────────────────

def train_parallel(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Learner device: {device} | Workers: {args.workers}")

    save_dir = os.path.join("runs", f"tron2_{args.tag}")
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "train_log.csv")

    # Networks on GPU (learner side)
    online_net = TRONPolicy2(input_dim=14, hidden_dim=HIDDEN_DIM).to(device)
    target_net = copy.deepcopy(online_net).to(device)
    target_net.eval()
    optimizer  = torch.optim.Adam(online_net.parameters(), lr=LR)
    buffer     = EpisodeReplayBuffer(capacity=REPLAY_CAPACITY)

    # ε schedule and episode counter
    eps_step    = (EPS_START - EPS_END) / EPS_DECAY_EPS
    start_ep    = args.start_episode
    global_ep   = start_ep
    epsilon     = max(EPS_END, EPS_START - eps_step * (start_ep - 1))
    grad_step   = 0
    best_profit = -float("inf")

    if args.load is not None:
        online_net.load_state_dict(
            torch.load(args.load, map_location=device, weights_only=True))
        target_net.load_state_dict(online_net.state_dict())
        print(f"Loaded weights from {args.load}")

    # Print env config
    _env = TRONEnv2(env_tag=args.env_tag, ne_strategy=args.ne_strategy)
    print(f"Env {args.env_tag} | strategy S{args.ne_strategy} "
          f"| shade={_env.bg_shade} eta={_env.bg_eta}")
    del _env

    log_mode = "a" if start_ep > 1 else "w"
    with open(log_path, log_mode, newline="") as f:
        if log_mode == "w":
            csv.writer(f).writerow(["episode", "epsilon", "loss", "eval_profit"])

    # Spawn workers
    ctx        = mp.get_context("spawn")
    rollout_q  = ctx.Queue(maxsize=4 * args.workers)
    weights_qs = [ctx.Queue(maxsize=1) for _ in range(args.workers)]

    worker_procs = []
    for wid in range(args.workers):
        p = ctx.Process(
            target=_worker_fn,
            args=(wid, rollout_q, weights_qs[wid],
                  args.env_tag, args.ne_strategy, args.seed),
            daemon=True,
        )
        p.start()
        worker_procs.append(p)
    print(f"Spawned {args.workers} workers (PIDs: {[p.pid for p in worker_procs]})")

    # Send initial weights — workers are blocking on weights_q.get()
    initial_sd = {k: v.detach().cpu().clone() for k, v in online_net.state_dict().items()}
    for wq in weights_qs:
        wq.put((initial_sd, epsilon))

    t0                = time.time()
    last_broadcast_ep = start_ep

    print(f"Training for {args.episodes:,} episodes (starting at {start_ep:,}) …")

    while global_ep <= args.episodes:
        # Pull one rollout (1s timeout so we can detect dead workers)
        try:
            rollout = rollout_q.get(timeout=1.0)
        except queue.Empty:
            alive = sum(1 for p in worker_procs if p.is_alive())
            if alive == 0:
                print("ERROR: all workers died. Aborting.")
                break
            continue

        buffer.add(rollout["obs"], rollout["actions"],
                   rollout["rewards"], rollout["dones"])
        epsilon = max(EPS_END, EPS_START - eps_step * (global_ep - 1))

        # Gradient step (TRAIN_FREQ=1 per episode)
        loss_val = float("nan")
        if len(buffer) >= MIN_BUFFER_EPS:
            batch    = buffer.sample(BATCH_SIZE, device)
            loss     = compute_loss(batch, online_net, target_net, GAMMA, N_STEP, device)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online_net.parameters(), GRAD_CLIP)
            optimizer.step()
            loss_val  = loss.item()
            grad_step += 1
            if grad_step % TARGET_UPDATE == 0:
                target_net.load_state_dict(online_net.state_dict())

        # Broadcast weights to workers (drain-then-put for freshness)
        if (global_ep - last_broadcast_ep) >= args.weight_broadcast_freq:
            cpu_sd = {k: v.detach().cpu().clone() for k, v in online_net.state_dict().items()}
            for wq in weights_qs:
                try:
                    wq.get_nowait()          # evict stale entry
                except queue.Empty:
                    pass
                try:
                    wq.put_nowait((cpu_sd, epsilon))
                except queue.Full:
                    pass                     # worker is loading weights; gets next broadcast
            last_broadcast_ep = global_ep

        # Save latest checkpoint
        if global_ep % LOG_FREQ == 0:
            torch.save(online_net.state_dict(), os.path.join(save_dir, "latest_model.pt"))

        # Periodic eval (workers back-pressure via full rollout_q during eval)
        eval_profit = float("nan")
        if global_ep % EVAL_FREQ == 0:
            eval_profit = evaluate(online_net, args.env_tag, args.ne_strategy,
                                   EVAL_RUNS, device)
            if eval_profit > best_profit:
                best_profit = eval_profit
                torch.save(online_net.state_dict(), os.path.join(save_dir, "best_model.pt"))
                print(f"  [ep {global_ep:>9,}] new best paired advantage: {best_profit:.1f}")

        # Log
        if global_ep % LOG_FREQ == 0:
            elapsed     = time.time() - t0
            eps_per_sec = (global_ep - start_ep + 1) / max(elapsed, 1e-6)
            print(
                f"ep {global_ep:>9,} | ε={epsilon:.4f} | buf={len(buffer):>6,} "
                f"| loss={loss_val:.5f} | {eps_per_sec:.0f} ep/s"
            )
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    global_ep, round(epsilon, 5), round(loss_val, 6),
                    round(eval_profit, 3) if not np.isnan(eval_profit) else ""
                ])

        global_ep += 1

    # Final eval and save
    final_profit = evaluate(online_net, args.env_tag, args.ne_strategy, EVAL_RUNS, device)
    print(f"\nFinal eval paired advantage (greedy, {EVAL_RUNS} runs): {final_profit:.1f}")
    torch.save(online_net.state_dict(), os.path.join(save_dir, "latest_model.pt"))
    if final_profit > best_profit:
        torch.save(online_net.state_dict(), os.path.join(save_dir, "best_model.pt"))
    print(f"Models saved to {save_dir}/")

    # Graceful shutdown: send sentinel to each worker
    for wq in weights_qs:
        try:
            wq.get_nowait()
        except queue.Empty:
            pass
        wq.put(None)
    for p in worker_procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()


# ── Eval only ──────────────────────────────────────────────────────────────────

def eval_only(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = TRONPolicy2(input_dim=14, hidden_dim=HIDDEN_DIM).to(device)
    policy.load_state_dict(torch.load(args.load, map_location=device, weights_only=True))
    policy.eval()
    profit = evaluate(policy, args.env_tag, args.ne_strategy, EVAL_RUNS, device)
    print(f"Eval paired advantage ({EVAL_RUNS} runs, env {args.env_tag}, S{args.ne_strategy}): {profit:.2f}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train TRON2 agent via parallel R2D2")
    p.add_argument("--env-tag",      default="B",       choices=["A", "B", "C"])
    p.add_argument("--ne-strategy",  default=8,         type=int)
    p.add_argument("--episodes",     default=3_000_000, type=int)
    p.add_argument("--tag",          default="v1_par")
    p.add_argument("--seed",         default=42,        type=int)
    p.add_argument("--eval-only",    action="store_true")
    p.add_argument("--load",         default=None,
                   help="Path to .pt weights to load (eval or resume)")
    p.add_argument("--start-episode", default=1, type=int,
                   help="Resume from this episode number")
    p.add_argument("--workers",      default=8, type=int,
                   help="Number of parallel worker processes (default: 8)")
    p.add_argument("--weight-broadcast-freq", default=200, type=int,
                   dest="weight_broadcast_freq",
                   help="Broadcast updated weights every N episodes (default: 200)")
    return p.parse_args()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    if args.eval_only:
        if args.load is None:
            raise ValueError("--eval-only requires --load <path>")
        eval_only(args)
    else:
        train_parallel(args)
