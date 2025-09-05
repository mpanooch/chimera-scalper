#!/usr/bin/env python3
"""
Simple script to save your Bybit API credentials
"""

import json

# Your API credentials
config = {
    "testnet": {
        "api_key": "Oe8dcE4d3dV2KtJYX8",
        "api_secret": "0SvXiHU2rJ3Ba92t20kF8W3L2u1GNGA2OBV6",
        "use_testnet": True
    },
    "mainnet": {
        "api_key": "",
        "api_secret": "",
        "use_testnet": False
    },
    "trading": {
        "default_leverage": 1,
        "max_position_size_usd": 1000,
        "default_order_size_usd": 100,
        "stop_loss_percent": 2.0,
        "take_profit_percent": 3.0,
        "max_open_positions": 5
    }
}

# Save to file
with open('bybit_config.json', 'w') as f:
    json.dump(config, f, indent=4)

print("✅ Configuration saved to bybit_config.json")
print("\nYour testnet API credentials have been configured.")
print("\n⚠️  IMPORTANT: Add bybit_config.json to .gitignore to keep it secure!")

# Add to .gitignore
try:
    with open('.gitignore', 'a') as f:
        f.write("\nbybit_config.json\n")
    print("✅ Added bybit_config.json to .gitignore")
except:
    pass