#!/usr/bin/env python3
"""
Bybit Authentication and Configuration Module
Handles demo/live API authentication and configuration management
"""

import json
import hmac
import hashlib
import time
import requests
import asyncio
import websockets
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

class BybitConfig:
    """Configuration management for Bybit API"""

    def __init__(self, config_file: str = "bybit_config.json"):
        self.config_file = Path(config_file)
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if not self.config_file.exists():
            # Create default config
            default_config = {
                "demo": {
                    "api_key": "2G7MTtzez9oHZN2KBZ",
                    "api_secret": "UIq7KocKSDGRfM2eXfAtF48OTpzK9IxdeQ7O"
                },
                "live": {
                    "api_key": "",
                    "api_secret": "",
                    "enabled": False
                },
                "trading": {
                    "default_leverage": 1,
                    "max_position_size_usd": 1000,
                    "default_order_size_usd": 100,
                    "stop_loss_percent": 2.0,
                    "take_profit_percent": 3.0,
                    "max_open_positions": 5,
                    "risk_per_trade": 0.02
                },
                "websocket": {
                    "demo": "wss://stream-demo.bybit.com/v5/public/linear",
                    "live": "wss://stream.bybit.com/v5/public/linear",
                    "ping_interval": 20,
                    "ping_timeout": 20,
                    "close_timeout": 10
                },
                "api_endpoints": {
                    "demo": "https://api-demo.bybit.com",
                    "live": "https://api.bybit.com"
                }
            }
            self.save_config(default_config)
            return default_config

        with open(self.config_file, 'r') as f:
            return json.load(f)

    def save_config(self, config: Dict[str, Any] = None):
        """Save configuration to JSON file"""
        if config:
            self.config = config

        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=4)

    def get_env_config(self, env: str = "demo") -> Dict[str, str]:
        """Get configuration for specific environment"""
        if env not in self.config:
            raise ValueError(f"Environment '{env}' not found in config")
        return self.config[env]

    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading configuration"""
        return self.config.get("trading", {})

    def get_websocket_config(self, env: str = "demo") -> Dict[str, Any]:
        """Get WebSocket configuration for environment"""
        ws_config = self.config.get("websocket", {})
        return {
            "url": ws_config.get(env, ws_config.get("demo")),
            "ping_interval": ws_config.get("ping_interval", 20),
            "ping_timeout": ws_config.get("ping_timeout", 20),
            "close_timeout": ws_config.get("close_timeout", 10)
        }

    def get_api_endpoint(self, env: str = "demo") -> str:
        """Get API endpoint for environment"""
        endpoints = self.config.get("api_endpoints", {})
        return endpoints.get(env, endpoints.get("demo"))

class BybitAuth:
    """Bybit API authentication handler"""

    def __init__(self, config: BybitConfig, env: str = "demo"):
        self.config = config
        self.env = env
        self.env_config = config.get_env_config(env)
        self.api_key = self.env_config.get("api_key")
        self.api_secret = self.env_config.get("api_secret")
        self.base_url = config.get_api_endpoint(env)

        if not self.api_key or not self.api_secret:
            raise ValueError(f"API credentials not configured for {env} environment")

    def generate_signature(self, params: Dict[str, Any]) -> str:
        """Generate HMAC SHA256 signature for Bybit API"""
        # Sort parameters
        sorted_params = sorted(params.items())
        query_string = "&".join([f"{k}={v}" for k, v in sorted_params])

        # Generate signature
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return signature

    def get_signed_params(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get signed parameters for API request"""
        if params is None:
            params = {}

        # Add required parameters
        timestamp = str(int(time.time() * 1000))
        params.update({
            "api_key": self.api_key,
            "timestamp": timestamp,
            "recv_window": "5000"
        })

        # Generate and add signature
        signature = self.generate_signature(params)
        params["sign"] = signature

        return params

    def make_request(self, endpoint: str, method: str = "GET", params: Dict[str, Any] = None) -> requests.Response:
        """Make authenticated API request"""
        if params is None:
            params = {}

        # Get signed parameters
        signed_params = self.get_signed_params(params)

        # Make request
        url = f"{self.base_url}{endpoint}"

        if method.upper() == "GET":
            response = requests.get(url, params=signed_params)
        elif method.upper() == "POST":
            response = requests.post(url, data=signed_params)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        return response

    def get_balance(self) -> Dict[str, Any]:
        """Get account balance"""
        params = {"accountType": "UNIFIED"}
        response = self.make_request("/v5/account/wallet-balance", params=params)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get balance: {response.status_code} - {response.text}")

    def get_positions(self, category: str = "linear") -> Dict[str, Any]:
        """Get current positions"""
        params = {"category": category}
        response = self.make_request("/v5/position/list", params=params)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get positions: {response.status_code} - {response.text}")

    def place_order(self, symbol: str, side: str, order_type: str, qty: str,
                    price: str = None, stop_loss: str = None, take_profit: str = None) -> Dict[str, Any]:
        """Place a trading order"""
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,  # Buy or Sell
            "orderType": order_type,  # Market or Limit
            "qty": qty,
            "timeInForce": "GTC"  # Good Till Cancelled
        }

        if price and order_type == "Limit":
            params["price"] = price

        if stop_loss:
            params["stopLoss"] = stop_loss

        if take_profit:
            params["takeProfit"] = take_profit

        response = self.make_request("/v5/order/create", method="POST", params=params)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to place order: {response.status_code} - {response.text}")

    def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Cancel an existing order"""
        params = {
            "category": "linear",
            "symbol": symbol,
            "orderId": order_id
        }

        response = self.make_request("/v5/order/cancel", method="POST", params=params)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to cancel order: {response.status_code} - {response.text}")

class BybitWebSocketClient:
    """Enhanced WebSocket client with authentication support"""

    def __init__(self, config: BybitConfig, auth: BybitAuth, env: str = "demo"):
        self.config = config
        self.auth = auth
        self.env = env
        self.ws_config = config.get_websocket_config(env)
        self.websocket = None
        self.is_connected = False
        self.subscriptions = set()

    async def connect(self):
        """Connect to Bybit WebSocket"""
        try:
            self.websocket = await websockets.connect(
                self.ws_config["url"],
                ping_interval=self.ws_config["ping_interval"],
                ping_timeout=self.ws_config["ping_timeout"],
                close_timeout=self.ws_config["close_timeout"]
            )
            self.is_connected = True
            print(f"Connected to Bybit WebSocket ({self.env})")
            return True
        except Exception as e:
            print(f"Failed to connect to Bybit WebSocket: {e}")
            self.is_connected = False
            return False

    async def disconnect(self):
        """Disconnect from WebSocket"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            print("Disconnected from Bybit WebSocket")

    async def subscribe(self, topics: list):
        """Subscribe to WebSocket topics"""
        if not self.is_connected:
            raise Exception("Not connected to WebSocket")

        subscribe_msg = {
            "op": "subscribe",
            "args": topics
        }

        await self.websocket.send(json.dumps(subscribe_msg))
        self.subscriptions.update(topics)
        print(f"Subscribed to: {', '.join(topics)}")

    async def unsubscribe(self, topics: list):
        """Unsubscribe from WebSocket topics"""
        if not self.is_connected:
            raise Exception("Not connected to WebSocket")

        unsubscribe_msg = {
            "op": "unsubscribe",
            "args": topics
        }

        await self.websocket.send(json.dumps(unsubscribe_msg))
        self.subscriptions.difference_update(topics)
        print(f"Unsubscribed from: {', '.join(topics)}")

    async def listen(self, message_handler=None):
        """Listen for WebSocket messages"""
        if not self.is_connected:
            raise Exception("Not connected to WebSocket")

        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)

                    # Handle ping/pong
                    if data.get('op') == 'ping':
                        pong_msg = {"op": "pong"}
                        await self.websocket.send(json.dumps(pong_msg))
                        continue

                    # Call message handler if provided
                    if message_handler:
                        await message_handler(data)

                    yield data

                except json.JSONDecodeError as e:
                    print(f"Failed to decode WebSocket message: {e}")
                    continue

        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")
            self.is_connected = False
        except Exception as e:
            print(f"WebSocket error: {e}")
            self.is_connected = False

class BybitTradingClient:
    """High-level trading client combining REST API and WebSocket"""

    def __init__(self, env: str = "demo", config_file: str = "bybit_config.json"):
        self.env = env
        self.config = BybitConfig(config_file)
        self.auth = BybitAuth(self.config, env)
        self.ws_client = BybitWebSocketClient(self.config, self.auth, env)
        self.trading_config = self.config.get_trading_config()

        print(f"Initialized Bybit Trading Client ({env} mode)")

    async def start_websocket(self, symbols: list, message_handler=None):
        """Start WebSocket connection and subscribe to symbols"""
        success = await self.ws_client.connect()
        if not success:
            return False

        # Subscribe to tickers
        topics = [f"tickers.{symbol}" for symbol in symbols]
        await self.ws_client.subscribe(topics)

        return True

    async def listen_market_data(self, message_handler=None):
        """Listen for market data messages"""
        async for message in self.ws_client.listen(message_handler):
            yield message

    def get_account_info(self):
        """Get account balance and positions"""
        try:
            balance = self.auth.get_balance()
            positions = self.auth.get_positions()
            return {
                "balance": balance,
                "positions": positions,
                "status": "success"
            }
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }

    def calculate_position_size(self, price: float, risk_pct: float = None) -> float:
        """Calculate position size based on risk management"""
        if risk_pct is None:
            risk_pct = self.trading_config.get("risk_per_trade", 0.02)

        max_size_usd = self.trading_config.get("max_position_size_usd", 1000)
        default_size_usd = self.trading_config.get("default_order_size_usd", 100)

        # Use default size for now (you can enhance this with portfolio balance)
        position_size = min(default_size_usd, max_size_usd) / price

        return round(position_size, 6)

    def place_market_order(self, symbol: str, side: str, size_usd: float = None,
                           stop_loss_pct: float = None, take_profit_pct: float = None):
        """Place a market order with risk management"""
        try:
            # Get current price (you'd typically get this from market data)
            # For demo, we'll use a placeholder
            current_price = 50000  # This should come from your market data

            if size_usd is None:
                size_usd = self.trading_config.get("default_order_size_usd", 100)

            qty = str(self.calculate_position_size(current_price))

            # Calculate stop loss and take profit
            stop_loss = None
            take_profit = None

            if stop_loss_pct is None:
                stop_loss_pct = self.trading_config.get("stop_loss_percent", 2.0) / 100

            if take_profit_pct is None:
                take_profit_pct = self.trading_config.get("take_profit_percent", 3.0) / 100

            if side.upper() == "BUY":
                stop_loss = str(current_price * (1 - stop_loss_pct))
                take_profit = str(current_price * (1 + take_profit_pct))
            else:
                stop_loss = str(current_price * (1 + stop_loss_pct))
                take_profit = str(current_price * (1 - take_profit_pct))

            # Place order
            result = self.auth.place_order(
                symbol=symbol,
                side=side,
                order_type="Market",
                qty=qty,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

            return {
                "order": result,
                "status": "success",
                "details": {
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit
                }
            }

        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }

    async def close(self):
        """Close WebSocket connection"""
        await self.ws_client.disconnect()

# Utility functions for easy integration
def create_bybit_client(env: str = "demo") -> BybitTradingClient:
    """Factory function to create Bybit trading client"""
    return BybitTradingClient(env)

def test_connection(env: str = "demo") -> bool:
    """Test Bybit API connection"""
    try:
        client = BybitTradingClient(env)
        account_info = client.get_account_info()
        return account_info.get("status") == "success"
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False

# Example usage and testing
if __name__ == "__main__":
    async def test_bybit_integration():
        # Test configuration
        print("Testing Bybit integration...")

        # Test connection
        if test_connection("demo"):
            print("✅ API connection successful")
        else:
            print("❌ API connection failed")
            return

        # Create client
        client = create_bybit_client("demo")

        # Test account info
        account_info = client.get_account_info()
        print(f"Account info: {account_info}")

        # Test WebSocket
        symbols = ["BTCUSDT", "ETHUSDT"]

        async def handle_message(data):
            if 'topic' in data and 'tickers' in data['topic']:
                symbol = data['topic'].split('.')[-1]
                price = data['data'].get('lastPrice', 'N/A')
                print(f"{symbol}: ${price}")

        success = await client.start_websocket(symbols, handle_message)
        if success:
            print("✅ WebSocket connected")

            # Listen for a few messages
            count = 0
            async for message in client.listen_market_data():
                count += 1
                if count >= 10:  # Stop after 10 messages
                    break

        await client.close()
        print("✅ Test completed")

    # Run test
    asyncio.run(test_bybit_integration())