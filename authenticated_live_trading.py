#!/usr/bin/env python3
"""
Authenticated live trading with ML models on Bybit
"""

import asyncio
import websockets
import json
import time
import hmac
import hashlib
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import aiohttp
import warnings
warnings.filterwarnings('ignore')

from config_bybit_auth import BybitAuthenticatedClient, load_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RegimeClassifier(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=64, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        return self.classifier(h_n[-1])

class AuthenticatedTradingSystem:
    """Live trading system with real order execution"""

    def __init__(self, use_testnet=True, paper_trading=True):
        self.use_testnet = use_testnet
        self.paper_trading = paper_trading

        # Initialize Bybit client
        self.client = BybitAuthenticatedClient(use_testnet)

        # Load ML model
        self.model = self.load_model()

        # Trading state
        self.positions = {}
        self.pending_orders = {}
        self.symbol_data = {}
        self.sequence_length = 60

        # Statistics
        self.stats = {
            'signals': 0,
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0
        }

        # Risk management from config
        self.max_positions = self.client.trading_config['max_open_positions']
        self.default_size = self.client.trading_config['default_order_size_usd']
        self.stop_loss_pct = self.client.trading_config['stop_loss_percent']
        self.take_profit_pct = self.client.trading_config['take_profit_percent']

    def load_model(self):
        """Load trained ML model"""
        model_path = 'models/best_regime_classifier.pth'

        if not Path(model_path).exists():
            print("⚠️  No trained model found. Trading without ML signals.")
            return None

        model = RegimeClassifier()
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print("✅ ML model loaded")
        return model

    async def get_auth_ws_connection(self):
        """Create authenticated WebSocket connection"""
        # Generate authentication
        expires = int((time.time() + 10) * 1000)
        signature = hmac.new(
            self.client.api_secret.encode('utf-8'),
            f'GET/realtime{expires}'.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        auth_msg = {
            "op": "auth",
            "args": [self.client.api_key, expires, signature]
        }

        # Connect to private WebSocket
        ws_url = self.client.ws_url
        ws = await websockets.connect(ws_url)

        # Authenticate
        await ws.send(json.dumps(auth_msg))
        response = await ws.recv()
        data = json.loads(response)

        if data.get('success'):
            print("✅ Authenticated WebSocket connected")
            return ws
        else:
            print(f"❌ Authentication failed: {data}")
            return None

    def calculate_features(self, ticker_data):
        """Extract features from ticker data"""
        features = np.zeros(14)

        if 'lastPrice' in ticker_data:
            price = float(ticker_data['lastPrice'])
            bid = float(ticker_data.get('bid1Price', price))
            ask = float(ticker_data.get('ask1Price', price))
            volume = float(ticker_data.get('volume24h', 0))

            features[0] = 0  # returns placeholder
            features[1] = np.log1p(volume / 1e6)
            features[2] = ((ask - bid) / price) * 10000  # spread bps
            features[3] = float(ticker_data.get('price24hPcnt', 0))
            features[11] = 1.0 if ticker_data['symbol'] in ['BTCUSDT', 'ETHUSDT'] else 0.0

        return features

    def predict_signal(self, symbol):
        """Generate ML-based trading signal"""
        if self.model is None or symbol not in self.symbol_data:
            return None

        if len(self.symbol_data[symbol]) < self.sequence_length:
            return None

        # Prepare input
        sequence = np.array(list(self.symbol_data[symbol]))
        input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            regime = output.argmax(1).item()
            confidence = probs[0, regime].item()

        if confidence > 0.7:
            regime_names = ['TREND_UP', 'TREND_DOWN', 'RANGE_BOUND', 'LIQUIDITY_EVENT', 'CHAOTIC']

            # Generate trade decision
            if regime == 0:  # TREND_UP
                return {'action': 'BUY', 'confidence': confidence, 'regime': regime_names[regime]}
            elif regime == 1:  # TREND_DOWN
                return {'action': 'SELL', 'confidence': confidence, 'regime': regime_names[regime]}

        return None

    async def execute_trade(self, symbol, signal, price):
        """Execute real trade on Bybit"""
        self.stats['signals'] += 1

        # Check risk limits
        current_positions = await self.client.get_positions()
        if len(current_positions) >= self.max_positions:
            print(f"⚠️  Max positions reached ({self.max_positions})")
            return

        # Calculate order size
        qty = round(self.default_size / price, 4)

        if self.paper_trading:
            # Paper trading
            print(f"\n📝 PAPER TRADE:")
            print(f"   Symbol: {symbol}")
            print(f"   Side: {signal['action']}")
            print(f"   Qty: {qty}")
            print(f"   Price: ${price:.2f}")
            print(f"   Regime: {signal['regime']}")
            print(f"   Confidence: {signal['confidence']:.2%}")

            self.positions[symbol] = {
                'side': signal['action'],
                'qty': qty,
                'entry_price': price,
                'timestamp': datetime.now()
            }
            self.stats['trades'] += 1

        else:
            # Real trading
            print(f"\n💰 EXECUTING REAL TRADE:")
            print(f"   Symbol: {symbol}")
            print(f"   Side: {signal['action']}")
            print(f"   Qty: {qty}")

            side = "Buy" if signal['action'] == 'BUY' else "Sell"
            order_id = await self.client.place_order(
                symbol=symbol,
                side=side,
                qty=qty,
                order_type="Market"
            )

            if order_id:
                self.stats['trades'] += 1
                print(f"✅ Order executed: {order_id}")

                # Set stop loss and take profit
                if signal['action'] == 'BUY':
                    sl_price = price * (1 - self.stop_loss_pct / 100)
                    tp_price = price * (1 + self.take_profit_pct / 100)
                else:
                    sl_price = price * (1 + self.stop_loss_pct / 100)
                    tp_price = price * (1 - self.take_profit_pct / 100)

                print(f"   Stop Loss: ${sl_price:.2f}")
                print(f"   Take Profit: ${tp_price:.2f}")

    async def run(self):
        """Main trading loop"""
        print(f"""
        ╔════════════════════════════════════════════╗
        ║   🔥 AUTHENTICATED LIVE TRADING 🔥         ║
        ║   Mode: {'TESTNET' if self.use_testnet else 'MAINNET'}                          ║
        ║   Paper Trading: {'ON' if self.paper_trading else 'OFF'}                    ║
        ╚════════════════════════════════════════════╝
        """)

        # Check account balance first
        balance = await self.client.get_wallet_balance()
        if balance:
            for coin in balance.get('coin', []):
                if coin['coin'] == 'USDT':
                    print(f"\n💰 USDT Balance: ${float(coin['walletBalance']):.2f}")
                    print(f"   Available: ${float(coin['availableToWithdraw']):.2f}")

        # Get current positions
        positions = await self.client.get_positions()
        print(f"📊 Open Positions: {len(positions)}")

        # Main trading loop using public WebSocket for market data
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

        ws_url = "wss://stream-testnet.bybit.com/v5/public/linear" if self.use_testnet else "wss://stream.bybit.com/v5/public/linear"

        async with websockets.connect(ws_url) as ws:
            # Subscribe to tickers
            for symbol in symbols:
                subscribe_msg = {
                    "op": "subscribe",
                    "args": [f"tickers.{symbol}"]
                }
                await ws.send(json.dumps(subscribe_msg))

            print(f"\n✅ Monitoring: {', '.join(symbols)}")
            print("=" * 50)

            while True:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=30)
                    data = json.loads(message)

                    if data.get('op') == 'ping':
                        await ws.send(json.dumps({"op": "pong"}))

                    elif 'topic' in data and 'data' in data:
                        topic = data['topic']

                        if 'tickers' in topic:
                            symbol = topic.split('.')[-1]
                            ticker_data = data['data']

                            # Update symbol data
                            if symbol not in self.symbol_data:
                                self.symbol_data[symbol] = deque(maxlen=self.sequence_length)

                            features = self.calculate_features(ticker_data)
                            self.symbol_data[symbol].append(features)

                            # Generate signal
                            signal = self.predict_signal(symbol)

                            if signal and symbol not in self.positions:
                                price = float(ticker_data['lastPrice'])
                                await self.execute_trade(symbol, signal, price)

                    # Print stats periodically
                    if self.stats['trades'] > 0 and self.stats['trades'] % 5 == 0:
                        print(f"\n📊 Stats: Signals={self.stats['signals']}, Trades={self.stats['trades']}")

                except asyncio.TimeoutError:
                    await ws.send(json.dumps({"op": "ping"}))
                except Exception as e:
                    print(f"⚠️  Error: {e}")
                    break

async def main():
    # First run setup if needed
    config = load_config()
    if not config:
        print("Please run: python config_bybit_auth.py setup")
        return

    # Ask user for trading mode
    print("\nSelect trading mode:")
    print("1. Paper trading (recommended)")
    print("2. Real trading (use with caution!)")

    choice = input("Enter choice (1/2): ").strip()
    paper_trading = choice != "2"

    if not paper_trading:
        confirm = input("\n⚠️  WARNING: Real trading mode! Are you sure? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Switching to paper trading mode.")
            paper_trading = True

    # Create and run trading system
    system = AuthenticatedTradingSystem(
        use_testnet=True,
        paper_trading=paper_trading
    )

    try:
        await system.run()
    except KeyboardInterrupt:
        print("\n\n✓ Trading stopped")
        print(f"Final stats: {system.stats}")

if __name__ == "__main__":
    asyncio.run(main())