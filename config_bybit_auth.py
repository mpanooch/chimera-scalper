#!/usr/bin/env python3
"""
Bybit API configuration and authenticated trading
"""

import json
import hmac
import hashlib
import time
from pathlib import Path

# ============= CONFIGURATION =============
# Create a config.json file with your API credentials
CONFIG_FILE = "bybit_config.json"

def create_config_template():
    """Create a template configuration file"""
    template = {
        "testnet": {
            "api_key": "YOUR_TESTNET_API_KEY_HERE",
            "api_secret": "YOUR_TESTNET_API_SECRET_HERE",
            "use_testnet": True
        },
        "mainnet": {
            "api_key": "YOUR_MAINNET_API_KEY_HERE",
            "api_secret": "YOUR_MAINNET_API_SECRET_HERE",
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

    with open(CONFIG_FILE, 'w') as f:
        json.dump(template, f, indent=4)

    print(f"✅ Created {CONFIG_FILE} template")
    print("Please edit it with your API credentials")

def load_config():
    """Load configuration from file"""
    if not Path(CONFIG_FILE).exists():
        print(f"❌ {CONFIG_FILE} not found. Creating template...")
        create_config_template()
        return None

    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)

    return config

def get_signature(api_secret, params):
    """Generate signature for authenticated requests"""
    param_str = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
    return hmac.new(
        api_secret.encode('utf-8'),
        param_str.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

class BybitAuthenticatedClient:
    """Authenticated Bybit client for trading"""

    def __init__(self, use_testnet=True):
        self.config = load_config()
        if not self.config:
            raise ValueError("Please configure your API credentials in bybit_config.json")

        # Select testnet or mainnet
        if use_testnet:
            self.api_key = self.config['testnet']['api_key']
            self.api_secret = self.config['testnet']['api_secret']
            self.base_url = "https://api-testnet.bybit.com"
            self.ws_url = "wss://stream-testnet.bybit.com/v5/private"
        else:
            self.api_key = self.config['mainnet']['api_key']
            self.api_secret = self.config['mainnet']['api_secret']
            self.base_url = "https://api.bybit.com"
            self.ws_url = "wss://stream.bybit.com/v5/private"

        # Check if credentials are configured
        if self.api_key == "YOUR_TESTNET_API_KEY_HERE":
            raise ValueError("Please add your actual API credentials to bybit_config.json")

        print(f"✅ Loaded API credentials for {'testnet' if use_testnet else 'mainnet'}")
        self.trading_config = self.config['trading']

    def get_auth_headers(self, params=None):
        """Generate authentication headers"""
        timestamp = str(int(time.time() * 1000))

        if params is None:
            params = {}

        sign_params = {
            'api_key': self.api_key,
            'timestamp': timestamp,
            **params
        }

        signature = get_signature(self.api_secret, sign_params)

        return {
            'X-BAPI-API-KEY': self.api_key,
            'X-BAPI-TIMESTAMP': timestamp,
            'X-BAPI-SIGN': signature,
            'Content-Type': 'application/json'
        }

    async def place_order(self, symbol, side, qty, order_type="Market", price=None):
        """Place an order on Bybit"""
        import aiohttp

        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,  # "Buy" or "Sell"
            "orderType": order_type,
            "qty": str(qty),
            "timeInForce": "GTC"
        }

        if price and order_type == "Limit":
            params["price"] = str(price)

        headers = self.get_auth_headers()

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{self.base_url}/v5/order/create",
                    headers=headers,
                    json=params
            ) as response:
                data = await response.json()

                if data.get('retCode') == 0:
                    order_id = data['result']['orderId']
                    print(f"✅ Order placed: {order_id}")
                    return order_id
                else:
                    print(f"❌ Order failed: {data.get('retMsg')}")
                    return None

    async def get_positions(self):
        """Get current positions"""
        import aiohttp

        params = {"category": "linear"}
        headers = self.get_auth_headers(params)

        async with aiohttp.ClientSession() as session:
            async with session.get(
                    f"{self.base_url}/v5/position/list",
                    headers=headers,
                    params=params
            ) as response:
                data = await response.json()

                if data.get('retCode') == 0:
                    return data['result']['list']
                else:
                    print(f"❌ Failed to get positions: {data.get('retMsg')}")
                    return []

    async def get_wallet_balance(self):
        """Get wallet balance"""
        import aiohttp

        params = {"accountType": "UNIFIED"}
        headers = self.get_auth_headers(params)

        async with aiohttp.ClientSession() as session:
            async with session.get(
                    f"{self.base_url}/v5/account/wallet-balance",
                    headers=headers,
                    params=params
            ) as response:
                data = await response.json()

                if data.get('retCode') == 0:
                    return data['result']['list'][0] if data['result']['list'] else None
                else:
                    print(f"❌ Failed to get balance: {data.get('retMsg')}")
                    return None

# ============= SETUP SCRIPT =============
def setup_api_credentials():
    """Interactive setup for API credentials"""
    print("""
    ╔════════════════════════════════════════╗
    ║   🔑 BYBIT API SETUP 🔑                ║
    ╚════════════════════════════════════════╝
    """)

    print("\nTo get your API keys:")
    print("1. Go to: https://testnet.bybit.com")
    print("2. Login or create account")
    print("3. Go to API Management")
    print("4. Create new API key")
    print("5. Enable 'Contract Trade' permissions")
    print("6. Save your keys securely\n")

    use_testnet = input("Setup for testnet? (y/n): ").lower() == 'y'

    api_key = input("Enter API Key: ").strip()
    api_secret = input("Enter API Secret: ").strip()

    # Load existing config or create new
    if Path(CONFIG_FILE).exists():
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
    else:
        config = {
            "testnet": {},
            "mainnet": {},
            "trading": {
                "default_leverage": 1,
                "max_position_size_usd": 1000,
                "default_order_size_usd": 100,
                "stop_loss_percent": 2.0,
                "take_profit_percent": 3.0,
                "max_open_positions": 5
            }
        }

    # Update config
    if use_testnet:
        config['testnet']['api_key'] = api_key
        config['testnet']['api_secret'] = api_secret
        config['testnet']['use_testnet'] = True
    else:
        config['mainnet']['api_key'] = api_key
        config['mainnet']['api_secret'] = api_secret
        config['mainnet']['use_testnet'] = False

    # Save config
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"\n✅ API credentials saved to {CONFIG_FILE}")
    print("⚠️  Keep this file secure and never commit it to git!")

    # Add to .gitignore
    gitignore_path = Path('.gitignore')
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            content = f.read()
        if CONFIG_FILE not in content:
            with open(gitignore_path, 'a') as f:
                f.write(f"\n{CONFIG_FILE}\n")
            print(f"✅ Added {CONFIG_FILE} to .gitignore")

async def test_authenticated_connection():
    """Test authenticated API connection"""
    import asyncio

    print("\n🔍 Testing authenticated connection...")

    try:
        client = BybitAuthenticatedClient(use_testnet=True)

        # Test getting balance
        balance = await client.get_wallet_balance()
        if balance:
            print(f"✅ Connected successfully!")
            print(f"   Account Type: {balance.get('accountType')}")

            # Show USDT balance
            for coin in balance.get('coin', []):
                if coin['coin'] == 'USDT':
                    print(f"   USDT Balance: {coin['walletBalance']}")
                    print(f"   Available: {coin['availableToWithdraw']}")

        # Test getting positions
        positions = await client.get_positions()
        print(f"   Open Positions: {len(positions)}")

        return True

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup_api_credentials()
    else:
        # Test connection
        import asyncio
        asyncio.run(test_authenticated_connection())