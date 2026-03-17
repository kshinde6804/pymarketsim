# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT: Running Long-Running Processes

**ALL long-running Python commands (training, experiments, equilibrium runs) MUST be wrapped with `caffeinate` to prevent the process from being interrupted if the computer goes to sleep.**

When running any of the following commands from the terminal, ALWAYS prefix them with `caffeinate -disu`:

- `train_mlp.py` — SAC training (can take 25-40+ minutes)
- `train_tron.py` — R2D2 training (can take hours)
- `equilibrium_mlp.py` — Equilibrium experiments (hundreds/thousands of simulations)
- `equilibrium_tron.py` — Equilibrium experiments (hundreds/thousands of simulations)
- `experiments/NE_ZI-ZI/equilibrium_experiment.py` — Full 10×10 experiment (10,000+ simulations)

**Example usage:**
```bash
# Instead of: python train_mlp.py --tag v1
# Always use:
caffeinate -disu python train_mlp.py --tag v1

# Instead of: python equilibrium_tron.py --num-runs 500
# Always use:
caffeinate -disu python equilibrium_tron.py --num-runs 500
```

The `caffeinate` flags:
- `-d` prevents the display from sleeping
- `-i` prevents the system from idle sleeping
- `-s` prevents the system from sleeping (works when on AC power)
- `-u` prevents the system from sleeping when on battery

This ensures that if the user's computer goes to sleep during a long-running task, the process will continue running uninterrupted.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## Running Tests

```bash
# Integration smoke tests (ZIEnv + MLPAgent — checks obs/action/reward shapes, no NaNs)
python test_env.py

# Unit tests for ZI agent behavior
python marketsim/tests/zi_agent_test.py
```

There is no pytest configuration; tests are standalone scripts.

## Project Layout

```
marketsim/
  agent/
    zero_intelligence_agent.py   # ZIAgent base (shade/eta, Bayesian fundamental, update_position)
    mlp_agent.py                 # ZIMlpPolicy + MLPAgent (inherits ZIAgent, 13-dim obs, SAC)
    tron_agent.py                # TRONPolicy + TRONAgent (Dueling DQN + LSTM, 14-dim obs, R2D2)
  wrappers/
    zi_env.py                    # ZIEnv: Gymnasium env for SAC (continuous Box action)
    tron_env.py                  # TRONEnv: Gymnasium env for R2D2 (MultiDiscrete action)
    metrics.py                   # Order-book metrics: RSI, vol_imbalance, realized_vol, etc.
  market/market.py               # Market class (FourHeap + fundamental + event_queue)
  simulator/simulator.py         # Simulator: agent dict, markets list, Poisson arrivals
  event/event_queue.py           # EventQueue: time→orders; set_time(t) must precede add_orders()
  fourheap/fourheap.py           # FourHeap: price-time priority matching

experiments/
  experiment_framework.py        # ExperimentFramework class (3 envs A/B/C, multi-trial runner)
  NE_ZI-ZI/
    equilibrium_experiment.py    # 10×10 ZI NE search (seeded, 500 runs/cell, mp.Pool)
    equilibrium_experiment.ipynb # Notebook version with heatmap display
  NE_ZI-MLP/                     # (MLP equilibrium outputs go here)
  NE_ZI-TRON/                    # (TRON equilibrium outputs go here)

train_mlp.py                     # SAC training script for ZIEnv (stable-baselines3)
train_tron.py                    # R2D2 training script for TRONEnv (custom PyTorch)
equilibrium_mlp.py               # Extend 10×10 ZI table to 10×11 with SAC-MLP deviator
equilibrium_tron.py              # Extend 10×10 ZI table to 10×11 with TRON deviator
```

> **IMPORTANT:** `equilibrium_experiment.py` lives at `experiments/NE_ZI-ZI/`, NOT the project root.
> `equilibrium_mlp.py`, `equilibrium_tron.py`, `train_mlp.py`, and `train_tron.py` **inline** the
> STRATEGIES and ENV dicts rather than importing from `equilibrium_experiment` — this avoids import
> errors since `NE_ZI-ZI` contains hyphens and is not a valid Python package name.

## Architecture

PyMarketSim is an **event-driven, agent-based limit order book simulator** built for market microstructure research and RL experimentation.

### Core Simulation Loop

```
Simulator.run()
  └─ for t in range(sim_time):
       ├─ sample Poisson arrivals → agents[id].take_action() → List[Order]
       ├─ Market.add_orders(orders) → EventQueue.schedule(t, orders)
       └─ Market.step()
            ├─ EventQueue.step() → orders at time t
            ├─ FourHeap.insert(order) for each order
            ├─ FourHeap.market_clear(t) → List[MatchedOrder]
            └─ update_position(agent, qty, price) for each match
```

**Key classes:**
- `Simulator` (`marketsim/simulator/simulator.py`) — manages `sim.agents` dict and `sim.markets` list; drives Poisson arrivals and the per-step loop
- `Market` (`marketsim/market/market.py`) — holds `order_book` (FourHeap), `fundamental`, `event_queue`; exposes `add_orders()`, `step()`, `withdraw_all(agent_id)`
- `EventQueue` (`marketsim/event/event_queue.py`) — maps time → scheduled orders; `set_time(t)` must be called before submitting orders outside the normal arrival flow
- `FourHeap` (`marketsim/fourheap/fourheap.py`) — price-time priority matching; `market_clear(t)` returns matched orders; `get_best_bid()` / `get_best_ask()` return `inf` when book is empty

### Agents

All agents extend `Agent` (ABC) and implement `take_action() → List[Order]`.

- **ZIAgent** (`marketsim/agent/zero_intelligence_agent.py`) — base agent: shade (valuation offset) + eta (liquidity-taking threshold); `estimate_fundamental()` computes Bayesian posterior; `update_position(q, p)` settles executions
- **MLPAgent** (`marketsim/agent/mlp_agent.py`) — inherits ZIAgent; `ZIMlpPolicy` is a 3-layer ReLU MLP (hidden=64) with Sigmoid output producing `[shade_norm, eta]`; `build_obs()` → 13-dim; uses `market.end_time` for correct `time_left` in Simulator context
- **TRONAgent** (`marketsim/agent/tron_agent.py`) — inherits ZIAgent; `TRONPolicy` is Dueling DQN + LSTM (hidden=128); discrete actions: 42 shade bins × 2 eta bins; `build_obs(side)` → 14-dim (13 + side indicator); LSTM state `self.h_c` stored per-episode and zeroed in `reset()`

### Fundamentals

- `GaussianMeanReverting` — pre-generates all values at init (needs `final_time` upfront)
- `LazyGaussianMeanReverting` — generates on-demand; **strictly forward-only**: `_generate_at(t)` requires `t >= latest_t`. Calling `get_final_fundamental()` mid-episode sets `latest_t = sim_time+1`, causing `dt < 0` errors on any subsequent call. **Never call `get_final_fundamental()` during an episode; only in the terminal step's `end_sim()`.**

### RL Environment: ZIEnv

`ZIEnv` (`marketsim/wrappers/zi_env.py`) trains a ZI-style agent via SAC (continuous actions).

**Observation space:** 13-dim `Box`:
```
[time_left, fundamental, best_ask, best_bid, inventory, midprice_delta,
 vol_imbalance, queue_imbalance, realized_vol, rsi, est_fundamental, pv_buy, pv_sell]
```
`best_ask → 1.0` and `best_bid → 0.0` when the book is empty (use `math.isinf()` to detect).

**Action space:** 2-dim `Box([0,1])` — `[shade_norm, eta]`; denormalize shade as:
`shade = shade_norm * (shade_range[1] - shade_range[0]) + shade_range[0]`

**Normalizers** (must match between ZIEnv and MLPAgent):
```python
{"fundamental": 1e5, "invt": 10, "reward": 1e4, "pv": 5e5}
# Use reward=1e3 when lam=0.005 (sparser market → ~10x smaller per-step PnL)
```

**Episode flow:**
1. `reset()` — init `LazyGaussianMeanReverting`, warm up background agents (`run_agents_only()`), advance to first RL arrival (`run_until_next_zi_arrival()`)
2. `step(action)` — submit RL order → background agents step → market clears → reward → advance to next RL arrival

### RL Environment: TRONEnv

`TRONEnv` (`marketsim/wrappers/tron_env.py`) trains the TRON agent via R2D2 (discrete actions).

**Observation space:** 14-dim `Box` — same 13 ZIEnv features + side indicator `[13]` (0.0=BUY, 1.0=SELL).

**Action space:** `MultiDiscrete([42, 2])` — shade_idx and eta_idx.

**Side pre-assignment:** at each RL arrival, `run_until_next_zi_arrival()` samples `self.current_side = random.choice([BUY, SELL])` before calling `update_obs()`, so the obs includes the side before the agent decides.

**Discrete bins** (class-level constants on `TRONPolicy`, referenced by both `TRONEnv` and `TRONAgent`):
```python
SHADE_BINS = np.linspace(0, 600, 42)
ETA_BINS   = [0.0, 1.0]
```

### TRONPolicy Architecture

```
Input: 14-dim
  → Linear(14→128) → ReLU            [shared encoder]
  → LSTM(128, 128, batch_first=True)  [stateful per episode]
  → Linear(128→1)                     [V(s) value head]
  → Adv shade: Linear(128→128)→ReLU→Linear(128→42)
  → Adv eta:   Linear(128→128)→ReLU→Linear(128→2)
Q_shade = V + A_shade − mean(A_shade)
Q_eta   = V + A_eta   − mean(A_eta)
```

### Equilibrium Experiment

**Strategy set** (10 ZI strategies, inlined in all experiment/training files):
```python
STRATEGIES = {
    0: {'shade': [0, 450],    'eta': 0.5},   # S0
    1: {'shade': [0, 600],    'eta': 0.5},   # S1
    2: {'shade': [90, 110],   'eta': 0.5},   # S2
    3: {'shade': [140, 160],  'eta': 0.5},   # S3
    4: {'shade': [190, 210],  'eta': 0.5},   # S4
    5: {'shade': [280, 320],  'eta': 0.5},   # S5
    6: {'shade': [380, 420],  'eta': 0.5},   # S6
    7: {'shade': [380, 420],  'eta': 1.0},   # S7 (market-taker)
    8: {'shade': [460, 540],  'eta': 0.5},   # S8
    9: {'shade': [950, 1050], 'eta': 0.5},   # S9
}
```

**Market ENV** (used by all experiment/training files):
```python
ENV = {'lam': 0.005, 'mean': 1e5, 'r': 0.01, 'shock_var': 1e6,
       'pv_var': 5e6, 'q_max': 10, 'sim_time': 2000, 'n_bg': 15}
```

**ExperimentFramework** (`experiments/experiment_framework.py`) provides 3 pre-defined market environments (A: lam=0.0005, B: lam=0.005, C: lam=0.012) and convenience methods for multi-trial runs and benchmark comparison against paper results.

### Training

**IMPORTANT:** All training commands MUST be run with `caffeinate -disu` to prevent sleep interruptions.

```bash
# SAC MLP agent (stable-baselines3)
caffeinate -disu python train_mlp.py                                        # 1M steps, 4 envs
caffeinate -disu python train_mlp.py --n-envs 4 --timesteps 500000 --tag v2
caffeinate -disu python train_mlp.py --eval-only --load runs/sac_zi_v1/best_model

# TRON agent (custom R2D2)
caffeinate -disu python train_tron.py                                       # 2M steps
caffeinate -disu python train_tron.py --timesteps 5000 --tag smoke
caffeinate -disu python train_tron.py --eval-only --load runs/tron_v1/best_model.pt
```

SAC outputs go to `runs/sac_zi_<tag>/` (SB3 `.zip`). R2D2 outputs go to `runs/tron_<tag>/` (PyTorch `.pt`).

### Equilibrium Evaluation

**IMPORTANT:** All equilibrium commands MUST be run with `caffeinate -disu` to prevent sleep interruptions.

```bash
caffeinate -disu python equilibrium_mlp.py --model runs/sac_zi_v3_eq/best_model --num-runs 500
caffeinate -disu python equilibrium_tron.py --model runs/tron_v1/best_model.pt --num-runs 500
```

Both load ZI×ZI baseline from `equilibrium_results.csv` and add a new deviator column.

### Key Bugs Fixed

1. **`event_queue.py` shuffle**: `random.shuffle()` → `self.rand.shuffle()` — the `rand_seed` param was accepted but ignored; global RNG was used instead, breaking reproducibility.
2. **`LazyGaussianMeanReverting` dt<0**: Always call `market.event_queue.set_time(self.time)` in `market_step()` before `market.step()`, and use `_env_estimate_fundamental()` (env clock) instead of `agent.estimate_fundamental()` (market clock) for reward/obs.
3. **Sparse lam warm-up**: With `lam=0.005`, use `warmup_fraction=0.0` and guarantee first RL arrival < `sim_time` by resampling in a while-loop in `reset_arrivals()`.
4. **`deque` O(n) indexing in replay buffer**: Replaced with a list-based circular buffer (O(1) random access).
5. **Cross-episode sequences**: With ~10 RL steps/episode and `seq_len=80`, sequences accumulate across episode boundaries instead of flushing+padding at each episode end.

### Other Wrappers

`marketsim/wrappers/` also contains `SP_wrapper.py`, `MM_wrapper.py`, and `MMSP_wrapper.py` for strategic play and market-making experiments.
