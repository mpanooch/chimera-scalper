            📡 Market Data Ingestion (Sub-ms)
      Tick Data + L2 Order Book + VWAP + Time-of-Day
                           │
──────────────────────────────────────────────────────────────
 LAYER 0: MICROSTRUCTURE ENGINE  (C++/CUDA — ~5ms latency)
──────────────────────────────────────────────────────────────
  ▸ Order book imbalance (bid vs ask pressure)
  ▸ Spread tracking & micro-price
  ▸ Liquidity pockets & hidden size detection
  ▸ Short-term volatility + entropy spikes
  ▸ Volume delta by price level
                           │
                           ▼
──────────────────────────────────────────────────────────────
 LAYER 1: FAST-PATH SCALP FILTER  (Statistical Checks)
──────────────────────────────────────────────────────────────
  Ross triggers: Flat-top breakout, EMA flow, red-to-green
  Bao triggers: VWAP deviation, liquidity void fills
  Nick triggers: Pressure shifts, volume confirms
  Fabio triggers: Chaos metric, market inefficiency spikes
  → Output: "Setup Found" (Pass) or "Ignore"
                           │
                           ▼
──────────────────────────────────────────────────────────────
 LAYER 2: MARKET REGIME CLASSIFICATION (ML — Transformer)
──────────────────────────────────────────────────────────────
  ▸ Trend Up
  ▸ Trend Down
  ▸ Range-bound
  ▸ Liquidity Event
  ▸ Chaotic Spike
  → Output: Regime Label + Confidence
                           │
                           ▼
──────────────────────────────────────────────────────────────
 LAYER 3: MULTI-EXPERT SIGNAL GENERATOR (MoE — RL-Enhanced)
──────────────────────────────────────────────────────────────
  🧠 Experts:
   1. Ross AI → Momentum breakouts, EMA momentum
   2. Bao AI  → VWAP mean reversion, liquidity fills
   3. Nick AI → Volume/pressure scalps, tape confirm
   4. Fabio AI→ Entropy/chaos inefficiency plays

  ▸ Each outputs:
    - Trade Direction
    - Entry Price Zone
    - Stop Loss / Take Profit
    - Confidence Score
  ▸ Gating network picks best match or blends aligned signals
                           │
                           ▼
──────────────────────────────────────────────────────────────
 LAYER 4: PnL-DRIVEN RL FILTER (Profitability Gate)
──────────────────────────────────────────────────────────────
  ▸ Simulates fills + latency + fees
  ▸ Filters out low-expected-profit trades
  ▸ Rewards profitable expert logic
                           │
                           ▼
──────────────────────────────────────────────────────────────
 LAYER 5: EXECUTION & RISK CONTROL (C++ Deterministic)
──────────────────────────────────────────────────────────────
  ▸ Position sizing (Kelly/volatility adjusted)
  ▸ Order slicing to reduce slippage
  ▸ Latency-optimized placement (CUDA-assisted)
  ▸ Hard kill-switch: daily loss / drawdown caps
  ▸ AI cannot override execution safety
                           │
                           ▼
──────────────────────────────────────────────────────────────
📈 LIVE PERFORMANCE MONITORING
──────────────────────────────────────────────────────────────
  ▸ PnL per expert
  ▸ Win rate, profit factor, drawdown
  ▸ Daily "Expert Scorecard"
