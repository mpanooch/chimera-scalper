# config.py
# Bybit API Configuration
BYBIT_API_KEY = "2G7MTtzez9oHZN2KBZ"
BYBIT_API_SECRET = "UIq7KocKSDGRfM2eXfAtF48OTpzK9IxdeQ7O"
BYBIT_TESTNET = True  # Set to False for live trading

# Alternatively, load from environment variables
import os
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY', 'your_api_key_here')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET', 'your_api_secret_here')
BYBIT_TESTNET = os.getenv('BYBIT_TESTNET', 'True').lower() == 'true'