# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
- **MLPAgent** (`marketsim/agent/mlp_agent.py`) — inherits ZIAgent; `ZIMlpPolicy` is a 3-layer ReLU MLP with Sigmoid output producing `[shade_norm, eta]`; uses identical pricing logic as ZIAgent

### Fundamentals

- `GaussianMeanReverting` — pre-generates all values at init (needs `final_time` upfront)
- `LazyGaussianMeanReverting` — generates on-demand; **strictly forward-only**: `_generate_at(t)` requires `t >= latest_t`. Calling `get_final_fundamental()` mid-episode sets `latest_t = sim_time+1`, causing `dt < 0` errors on any subsequent call. **Never call `get_final_fundamental()` during an episode; only in the terminal step's `end_sim()`.**

### RL Environment: ZIEnv

`ZIEnv` (`marketsim/wrappers/zi_env.py`) is a Gymnasium wrapper that trains a ZI-style agent via RL.

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
```

**Episode flow:**
1. `reset()` — init `LazyGaussianMeanReverting`, warm up background agents (`run_agents_only()`), advance to first RL arrival (`run_until_next_zi_arrival()`)
2. `step(action)` — submit RL order → background agents step → market clears → reward → advance to next RL arrival

**Warm-up arrival fix:** If the RL agent's first scheduled arrival falls during the warm-up window, `run_agents_only()` reschedules it forward so `run_until_next_zi_arrival()` finds a valid future slot.

### Training

```bash
python train_mlp.py                          # default SAC run
python train_mlp.py --n-envs 4 --timesteps 500000 --tag v3
python train_mlp.py --eval-only --load runs/sac_zi_v1/best_model
```

Checkpoints and TensorBoard logs go to `runs/<tag>/`.

### Other Wrappers

`marketsim/wrappers/` also contains `SPEnv`, `MMEnv`, and `MMSPEnv` for strategic play and market-making experiments. `ZIEnv` is the newest and most actively developed wrapper.
