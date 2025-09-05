#!/usr/bin/env python3
"""
Final ML Trading System with correct Bybit URLs
"""

import asyncio
import websockets
import json
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import deque
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RegimeClassifier(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=64, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=2, batch_first=True, dropout=0.2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        return self.classifier(h_n[-1])

class MLTradingSystem:
    def __init__(self, testnet=True):
        self.testnet = testnet

        # Correct URLs
        if testnet:
            self.ws_url = "wss://stream-testnet.bybit.com/v5/public/linear"
        else:
            self.ws_url = "wss://stream.bybit.com/v5/public/linear"

        # Load ML model
        self.model = self.load_model()

        # Data storage
        self.symbol_data = {}
        self.sequence_length = 60

        # Trading stats
        self.stats = {
            'ticks': 0,
            'signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'last_prices': {}
        }

        # Positions (paper trading)
        self.positions = {}

    def load_model(self):
        model_path = 'models/best_regime_classifier.pth'
        if not Path(model_path).exists():
            print("⚠️  No model found, will generate random signals for testing")
            return None

        model = RegimeClassifier()
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print(f"✅ ML model loaded from {model_path}")
        return model

    def process_ticker(self, symbol, ticker_data):
        """Process ticker data and generate signals"""
        self.stats['ticks'] += 1

        # Extract data
        price = float(ticker_data.get('lastPrice', 0))
        bid = float(ticker_data.get('bid1Price', price))
        ask = float(ticker_data.get('ask1Price', price))
        volume = float(ticker_data.get('volume24h', 0))
        change_pct = float(ticker_data.get('price24hPcnt', 0))

        self.stats['last_prices'][symbol] = price

        # Create features
        features = np.zeros(14)
        features[0] = change_pct  # 24h change
        features[1] = np.log1p(volume / 1e6)  # Normalized volume
        features[2] = ((ask - bid) / price) * 10000 if price > 0 else 0  # Spread bps
        features[3] = 0  # Placeholder
        features[11] = 1.0 if symbol in ['BTCUSDT', 'ETHUSDT'] else 0.0
        features[12] = 1.0  # Using linear/perpetual

        # Update symbol data
        if symbol not in self.symbol_data:
            self.symbol_data[symbol] = deque(maxlen=self.sequence_length)
        self.symbol_data[symbol].append(features)

        # Generate signal if enough data
        if len(self.symbol_data[symbol]) >= self.sequence_length:
            signal = self.generate_signal(symbol, price)
            if signal:
                self.execute_paper_trade(symbol, signal, price)

    def generate_signal(self, symbol, price):
        """Generate ML-based trading signal"""
        if self.model is None:
            # Random signals for testing
            if np.random.random() > 0.98:
                return {'action': 'BUY' if np.random.random() > 0.5 else 'SELL',
                        'confidence': np.random.random(),
                        'regime': 'TEST'}
            return None

        # ML prediction
        sequence = np.array(list(self.symbol_data[symbol]))
        input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            regime = output.argmax(1).item()
            confidence = probs[0, regime].item()

        if confidence > 0.75:  # High confidence threshold
            regime_names = ['TREND_UP', 'TREND_DOWN', 'RANGE_BOUND', 'LIQUIDITY_EVENT', 'CHAOTIC']

            if regime == 0:  # TREND_UP
                self.stats['signals'] += 1
                self.stats['buy_signals'] += 1
                return {'action': 'BUY', 'confidence': confidence, 'regime': regime_names[regime]}
            elif regime == 1:  # TREND_DOWN
                self.stats['signals'] += 1
                self.stats['sell_signals'] += 1
                return {'action': 'SELL', 'confidence': confidence, 'regime': regime_names[regime]}

        return None

    def execute_paper_trade(self, symbol, signal, price):
        """Execute paper trade"""
        timestamp = datetime.now().strftime('%H:%M:%S')

        print(f"\n🎯 [{timestamp}] SIGNAL GENERATED:")
        print(f"   Symbol: {symbol}")
        print(f"   Action: {signal['action']}")
        print(f"   Price: ${price:,.2f}")
        print(f"   Regime: {signal['regime']}")
        print(f"   Confidence: {signal['confidence']:.1%}")

        # Track position
        if symbol not in self.positions:
            self.positions[symbol] = {
                'side': signal['action'],
                'entry_price': price,
                'entry_time': timestamp,
                'size': 100  # $100 position
            }
            print(f"   ✅ Position opened")
        else:
            # Calculate PnL if closing position
            pos = self.positions[symbol]
            if pos['side'] != signal['action']:
                pnl_pct = ((price - pos['entry_price']) / pos['entry_price']) * 100
                if pos['side'] == 'SELL':
                    pnl_pct = -pnl_pct

                print(f"   📊 Closing {pos['side']} position")
                print(f"   PnL: {pnl_pct:+.2f}%")

                # Open new position
                self.positions[symbol] = {
                    'side': signal['action'],
                    'entry_price': price,
                    'entry_time': timestamp,
                    'size': 100
                }

    async def run(self):
        """Main trading loop"""
        print(f"""
        ╔═══════════════════════════════════════════╗
        ║   🔥 ML TRADING SYSTEM - LIVE 🔥          ║
        ║   {'Testnet' if self.testnet else 'Mainnet'}: {self.ws_url.split('//')[1].split('/')[0][:20]}
        ║   Model: {'Loaded ✓' if self.model else 'Testing Mode'}
        ╚═══════════════════════════════════════════╝
        """)

        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

        print(f"\nConnecting to Bybit...")

        try:
            async with websockets.connect(self.ws_url) as ws:
                print(f"✅ Connected to {self.ws_url}")

                # Subscribe to tickers
                for symbol in symbols:
                    subscribe_msg = {
                        "op": "subscribe",
                        "args": [f"tickers.{symbol}"]
                    }
                    await ws.send(json.dumps(subscribe_msg))

                print(f"✅ Subscribed to: {', '.join(symbols)}")
                print("\n" + "="*50)
                print("Waiting for signals... (high confidence only)")
                print("="*50)

                # Main loop
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
                                self.process_ticker(symbol, ticker_data)

                                # Print status every 100 ticks
                                if self.stats['ticks'] % 100 == 0:
                                    prices_str = ', '.join([f"{s}: ${p:,.0f}"
                                                            for s, p in self.stats['last_prices'].items()])
                                    print(f"\r📊 Ticks: {self.stats['ticks']} | "
                                          f"Signals: {self.stats['signals']} "
                                          f"(B:{self.stats['buy_signals']}/S:{self.stats['sell_signals']}) | "
                                          f"{prices_str}", end='')

                    except asyncio.TimeoutError:
                        await ws.send(json.dumps({"op": "ping"}))

        except KeyboardInterrupt:
            print("\n\n✓ Trading stopped by user")
            self.print_summary()
        except Exception as e:
            print(f"\n❌ Error: {e}")

    def print_summary(self):
        """Print trading summary"""
        print("\n" + "="*50)
        print("📈 TRADING SUMMARY")
        print("="*50)
        print(f"Total ticks processed: {self.stats['ticks']}")
        print(f"Signals generated: {self.stats['signals']}")
        print(f"  Buy signals: {self.stats['buy_signals']}")
        print(f"  Sell signals: {self.stats['sell_signals']}")

        if self.positions:
            print(f"\nOpen positions:")
            for symbol, pos in self.positions.items():
                current_price = self.stats['last_prices'].get(symbol, pos['entry_price'])
                pnl_pct = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
                if pos['side'] == 'SELL':
                    pnl_pct = -pnl_pct
                print(f"  {symbol}: {pos['side']} from ${pos['entry_price']:,.2f} "
                      f"(PnL: {pnl_pct:+.2f}%)")

async def main():
    system = MLTradingSystem(testnet=True)
    await system.run()

if __name__ == "__main__":
    asyncio.run(main())