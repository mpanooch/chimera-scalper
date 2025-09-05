chimera_scalper/
├── CMakeLists.txt                  # Root build config
├── venv/                           # Python virtual environment (ignored in VCS)
├── dev/                            # Development scripts, notebooks, tests
│   ├── l2_parser.py                # Python helper to parse `.data` L2 files
│   └── visualize_orderbook.ipynb   # Dev notebook for L2 microstructure
├── src/
│   ├── core/
│   │   ├── data_feed.cpp           # Market data + L2 ingestion
│   │   ├── data_feed.h
│   │   ├── features.cu             # CUDA microstructure + L2 features
│   │   └── features.h
│   ├── layers/
│   │   ├── layer1_fastpath.cpp     # Fast scalp filters
│   │   ├── layer1_fastpath.h
│   │   ├── layer2_regime.cpp       # Transformer regime classifier
│   │   ├── layer2_regime.h
│   │   ├── layer3_experts.cpp      # MoE signal logic
│   │   ├── layer3_experts.h
│   │   ├── layer4_pnlfilter.cpp    # PnL-driven RL filter
│   │   ├── layer4_pnlfilter.h
│   │   ├── layer5_execution.cpp    # Final execution + controls
│   │   └── layer5_execution.h
│   ├── main.cpp                    # Entry point
│   └── utils/
│       ├── logger.cpp
│       ├── logger.h
│       ├── config.h                # Global constants
│       └── timer.h
├── data/
│   ├── raw/
│   │   ├── ada_usdt_1m.csv       # Market data ingestion
│   │   ├── ada_usdt_5m.csv
│   │   ├── bnb_usdt_1m.csv
│   │   ├── bnb_usdt_5m.csv
│   │   ├── btc_usdt_1m.csv
│   │   ├── btc_usdt_5m.csv
│   │   ├── doge_usdt_1m.csv
│   │   ├── doge_usdt_5m.csv
│   │   ├── eth_usdt_1m.csv 
│   │   ├── eth_usdt_5m.csv
│   │   ├── meme_usdt_1m.csv
│   │   ├── meme_usdt_5m.csv
│   │   ├── shib_usdt_1m.csv
│   │   ├── shib_usdt_5m.csv
│   │   ├── sol_usdt_1m.csv
│   │   ├── sol_usdt_5m.csv
│   │   ├── shib_usdt_1m.csv
│   │   ├── shib_usdt_5m.csv
│   │   ├── xrp_usdt_1m.csv
│   │   ├── xrp_usdt_5m.csv
│   └── ob/
│       └── 2025-08-01_BTCUSDT_ob200.data   # Raw L2 data (Top 200 levels)
├── models/                         # Saved model weights
├── mathematica/
│   ├── ross_logic.nb               # Ross pattern detection math
│   └── tft_flow.nb                 # TFT flow math
├── README.md                       # Project intro, setup, usage
└── build/                          # CLion or CMake build output
