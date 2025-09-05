#!/usr/bin/env python3
"""
Fixed live trading with better connection handling
"""

import torch
import torch.nn as nn
import numpy as np
import asyncio
import websockets
import json
import time
import ssl
import certifi
from datetime import datetime
from pathlib import Path
from collections import deque
import warnings
warnings.filterwarnings('ignore')

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
        last_hidden = h_n[-1]
        output = self.classifier(last_hidden)
        return output

class SimpleTradingSystem:
    """Simplified trading system with better connection handling"""
    def __init__(self, model_path='models/best_regime_classifier.pth'):
        self.device = device
        self.model = self.load_model(model_path)
        self.symbol_data = {}
        self.sequence_length = 60
        self.signals_generated = 0
        self.last_signal_time = {}

    def load_model(self, model_path):
        print(f"Loading model from {model_path}...")
        model = RegimeClassifier(input_dim=14, hidden_dim=64, num_classes=5)

        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            print("✓ Model loaded successfully")
            return model
        else:
            print(f"❌ Model not found. Using random predictions.")
            return None

    def process_orderbook(self, symbol, bid_price, ask_price):
        """Simple feature extraction from price data"""
        if symbol not in self.symbol_data:
            self.symbol_data[symbol] = deque(maxlen=self.sequence_length)

        # Simple features
        mid_price = (bid_price + ask_price) / 2
        spread = (ask_price - bid_price) / mid_price

        # Create feature vector (14 dimensions)
        features = np.zeros(14)
        features[0] = 0  # returns placeholder
        features[1] = np.log1p(mid_price / 1000)  # normalized price
        features[2] = spread * 100  # spread in basis points
        features[3] = 0  # volume placeholder
        features[11] = 1.0 if symbol in ['BTCUSDT', 'ETHUSDT'] else 0.0
        features[12] = 1.0  # 1m timeframe

        # Add to history
        self.symbol_data[symbol].append(features)

        # Generate signal if enough data
        if len(self.symbol_data[symbol]) >= self.sequence_length:
            return self.generate_signal(symbol, mid_price)
        return None

    def generate_signal(self, symbol, price):
        """Generate trading signal"""
        # Rate limit signals (one per symbol per minute)
        current_time = time.time()
        if symbol in self.last_signal_time:
            if current_time - self.last_signal_time[symbol] < 60:
                return None

        if self.model is None:
            # Random signal for testing without model
            if np.random.random() > 0.95:
                self.signals_generated += 1
                self.last_signal_time[symbol] = current_time
                return {
                    'symbol': symbol,
                    'action': 'BUY' if np.random.random() > 0.5 else 'SELL',
                    'price': price,
                    'confidence': np.random.random(),
                    'regime': 'TEST'
                }
            return None

        # Use ML model
        sequence = np.array(list(self.symbol_data[symbol]))
        input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            regime = output.argmax(1).item()
            confidence = probabilities[0, regime].item()

        if confidence > 0.7:
            regime_names = ['TREND_UP', 'TREND_DOWN', 'RANGE_BOUND', 'LIQUIDITY_EVENT', 'CHAOTIC']

            self.signals_generated += 1
            self.last_signal_time[symbol] = current_time

            return {
                'symbol': symbol,
                'action': 'BUY' if regime == 0 else 'SELL' if regime == 1 else 'HOLD',
                'price': price,
                'confidence': confidence,
                'regime': regime_names[regime]
            }

        return None

async def connect_with_retry(url, max_retries=3):
    """Try to connect with retries"""
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    for attempt in range(max_retries):
        try:
            print(f"Connection attempt {attempt + 1}/{max_retries}...")
            ws = await websockets.connect(
                url,
                ssl=ssl_context,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10,
                max_size=10 * 1024 * 1024  # 10MB max message size
            )
            print("✓ Connected successfully!")
            return ws
        except Exception as e:
            print(f"✗ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)

    return None

async def run_trading():
    """Main trading loop with better error handling"""
    print("""
    ╔════════════════════════════════════════╗
    ║   🔥 CHIMERA ML LIVE TRADING 🔥        ║
    ║   Simplified & Fixed Version            ║
    ╚════════════════════════════════════════╝
    """)

    # Initialize trading system
    system = SimpleTradingSystem()

    # Try different URLs
    urls = [
        "wss://stream-testnet.bybit.com/v5/public/linear",
        "wss://stream.bybit.com/v5/public/linear",  # Mainnet as fallback
    ]

    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

    for url in urls:
        print(f"\nTrying: {url}")
        print("-" * 40)

        ws = await connect_with_retry(url)
        if ws is None:
            continue

        try:
            # Subscribe to tickers (simpler than orderbook)
            for symbol in symbols:
                subscribe_msg = {
                    "op": "subscribe",
                    "args": [f"tickers.{symbol}"]
                }
                await ws.send(json.dumps(subscribe_msg))
                print(f"✓ Subscribed to {symbol}")

            # Main loop
            while True:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=30)
                    data = json.loads(message)

                    # Handle different message types
                    if data.get('op') == 'ping':
                        await ws.send(json.dumps({"op": "pong"}))

                    elif 'topic' in data and 'data' in data:
                        # Extract price data
                        topic = data['topic']
                        if 'tickers' in topic:
                            symbol = topic.split('.')[-1]
                            ticker_data = data['data']

                            if 'bid1Price' in ticker_data and 'ask1Price' in ticker_data:
                                bid_price = float(ticker_data['bid1Price'])
                                ask_price = float(ticker_data['ask1Price'])

                                # Process and maybe generate signal
                                signal = system.process_orderbook(symbol, bid_price, ask_price)

                                if signal:
                                    print(f"\n🎯 SIGNAL: {signal['symbol']} {signal['action']} @ {signal['price']:.2f}")
                                    print(f"   Regime: {signal['regime']} | Confidence: {signal['confidence']:.2%}")

                    # Print stats periodically
                    if system.signals_generated > 0 and system.signals_generated % 5 == 0:
                        print(f"\n📊 Total signals generated: {system.signals_generated}")

                except asyncio.TimeoutError:
                    print(".", end="", flush=True)
                    await ws.send(json.dumps({"op": "ping"}))
                except Exception as e:
                    print(f"\n⚠️ Error: {e}")
                    break

        except Exception as e:
            print(f"\n❌ Connection lost: {e}")
        finally:
            await ws.close()

    print("\n❌ Could not establish connection to any server")

async def main():
    try:
        await run_trading()
    except KeyboardInterrupt:
        print("\n\n✓ Trading stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    # First test the connection
    print("Testing connection first...")
    import subprocess
    result = subprocess.run(['python', 'test_bybit_connection.py'], capture_output=True, text=True)
    print(result.stdout)

    if "successful" in result.stdout:
        print("\n✓ Connection test passed. Starting trading...")
        asyncio.run(main())
    else:
        print("\n❌ Connection test failed. Please check your internet connection.")
        print("\nPossible solutions:")
        print("1. Check if you're behind a firewall")
        print("2. Try using a VPN")
        print("3. Check if Bybit testnet is operational")