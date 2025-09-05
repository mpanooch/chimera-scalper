#!/usr/bin/env python3
"""
CHIMERA Scalper Demo Training System
- Collects live market data during demo trading
- Logs all trades and outcomes for retraining
- Runs for 2 weeks then triggers model retrain
"""

import asyncio
import websockets
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
import warnings
import sqlite3
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

class DemoTrainingSystem:
    def __init__(self):
        self.ws_url = "wss://stream-demo.bybit.com/v5/public/linear"

        # Load existing model
        self.model = self.load_model()

        # Data storage for live training
        self.symbol_data = {}
        self.sequence_length = 60

        # Training data collection
        self.training_start = datetime.now()
        self.training_duration = timedelta(days=14)  # 2 weeks
        self.data_buffer = []
        self.trade_log = []

        # Initialize database for data collection
        self.init_database()

        # Trading stats
        self.stats = {
            'ticks': 0,
            'signals': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_pnl': 0.0,
            'last_prices': {}
        }

        # Positions tracking
        self.positions = {}

    def init_database(self):
        """Initialize SQLite database for training data collection"""
        self.db_path = 'demo_training_data.db'
        conn = sqlite3.connect(self.db_path)

        # Market data table
        conn.execute('''
                     CREATE TABLE IF NOT EXISTS market_data (
                                                                id INTEGER PRIMARY KEY,
                                                                timestamp TEXT,
                                                                symbol TEXT,
                                                                price REAL,
                                                                bid REAL,
                                                                ask REAL,
                                                                volume REAL,
                                                                change_pct REAL,
                                                                spread_bps REAL,
                                                                features TEXT  -- JSON encoded features
                     )
                     ''')

        # Trades table
        conn.execute('''
                     CREATE TABLE IF NOT EXISTS trades (
                                                           id INTEGER PRIMARY KEY,
                                                           timestamp TEXT,
                                                           symbol TEXT,
                                                           action TEXT,
                                                           entry_price REAL,
                                                           exit_price REAL,
                                                           regime TEXT,
                                                           confidence REAL,
                                                           pnl_pct REAL,
                                                           success INTEGER  -- 1 for success, 0 for failure
                     )
                     ''')

        conn.commit()
        conn.close()
        print("✅ Database initialized for training data collection")

    def load_model(self):
        """Load existing regime classifier"""
        model_path = 'models/best_regime_classifier.pth'
        if not Path(model_path).exists():
            print("⚠️ No existing model found, starting fresh")
            return None

        model = RegimeClassifier()
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print(f"✅ Existing model loaded from {model_path}")
        return model

    def save_market_data(self, symbol, ticker_data, features):
        """Save market data to database for retraining"""
        conn = sqlite3.connect(self.db_path)

        price = float(ticker_data.get('lastPrice', 0))
        bid = float(ticker_data.get('bid1Price', price))
        ask = float(ticker_data.get('ask1Price', price))
        volume = float(ticker_data.get('volume24h', 0))
        change_pct = float(ticker_data.get('price24hPcnt', 0))
        spread_bps = ((ask - bid) / price) * 10000 if price > 0 else 0

        conn.execute('''
                     INSERT INTO market_data
                     (timestamp, symbol, price, bid, ask, volume, change_pct, spread_bps, features)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                     ''', (
                         datetime.now().isoformat(),
                         symbol,
                         price,
                         bid,
                         ask,
                         volume,
                         change_pct,
                         spread_bps,
                         json.dumps(features.tolist())
                     ))

        conn.commit()
        conn.close()

    def log_trade_outcome(self, symbol, trade_data, outcome):
        """Log trade outcomes for model performance evaluation"""
        conn = sqlite3.connect(self.db_path)

        conn.execute('''
                     INSERT INTO trades
                     (timestamp, symbol, action, entry_price, exit_price, regime, confidence, pnl_pct, success)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                     ''', (
                         datetime.now().isoformat(),
                         symbol,
                         trade_data['action'],
                         trade_data['entry_price'],
                         trade_data.get('exit_price', 0),
                         trade_data['regime'],
                         trade_data['confidence'],
                         outcome['pnl_pct'],
                         1 if outcome['pnl_pct'] > 0 else 0
                     ))

        conn.commit()
        conn.close()

    def check_training_complete(self):
        """Check if 2-week training period is complete"""
        elapsed = datetime.now() - self.training_start
        return elapsed >= self.training_duration

    def trigger_retrain(self):
        """Trigger model retraining after 2 weeks"""
        print("\n🔄 2-week demo period complete! Triggering model retrain...")

        # Export training data
        conn = sqlite3.connect(self.db_path)

        # Get market data
        market_df = pd.read_sql_query(
            "SELECT * FROM market_data ORDER BY timestamp", conn
        )

        # Get trade outcomes
        trades_df = pd.read_sql_query(
            "SELECT * FROM trades ORDER BY timestamp", conn
        )

        conn.close()

        print(f"📊 Collected {len(market_df)} market data points")
        print(f"📊 Executed {len(trades_df)} trades")

        success_rate = trades_df['success'].mean() if len(trades_df) > 0 else 0
        avg_pnl = trades_df['pnl_pct'].mean() if len(trades_df) > 0 else 0

        print(f"📈 Success rate: {success_rate:.1%}")
        print(f"📈 Average PnL: {avg_pnl:+.2f}%")

        # Save for retraining script
        market_df.to_csv('demo_market_data.csv', index=False)
        trades_df.to_csv('demo_trades_log.csv', index=False)

        print("✅ Data exported for retraining. Run retrain_model.py next!")
        return True

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

        # Create features (same as your original)
        features = np.zeros(14)
        features[0] = change_pct
        features[1] = np.log1p(volume / 1e6)
        features[2] = ((ask - bid) / price) * 10000 if price > 0 else 0
        features[11] = 1.0 if symbol in ['BTCUSDT', 'ETHUSDT'] else 0.0
        features[12] = 1.0

        # Save data for retraining
        self.save_market_data(symbol, ticker_data, features)

        # Update symbol data
        if symbol not in self.symbol_data:
            self.symbol_data[symbol] = deque(maxlen=self.sequence_length)
        self.symbol_data[symbol].append(features)

        # Generate signal if enough data
        if len(self.symbol_data[symbol]) >= self.sequence_length:
            signal = self.generate_signal(symbol, price)
            if signal:
                self.execute_demo_trade(symbol, signal, price)

    def generate_signal(self, symbol, price):
        """Generate ML-based trading signal"""
        if self.model is None:
            # Random signals for testing when no model exists
            if np.random.random() > 0.995:  # Very rare for testing
                return {
                    'action': 'BUY' if np.random.random() > 0.5 else 'SELL',
                    'confidence': np.random.random(),
                    'regime': 'RANDOM_TEST'
                }
            return None

        # ML prediction using existing model
        sequence = np.array(list(self.symbol_data[symbol]))
        input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            regime = output.argmax(1).item()
            confidence = probs[0, regime].item()

        # High confidence threshold for demo trading
        if confidence > 0.8:
            regime_names = ['TREND_UP', 'TREND_DOWN', 'RANGE_BOUND', 'LIQUIDITY_EVENT', 'CHAOTIC']

            if regime == 0:  # TREND_UP
                return {
                    'action': 'BUY',
                    'confidence': confidence,
                    'regime': regime_names[regime]
                }
            elif regime == 1:  # TREND_DOWN
                return {
                    'action': 'SELL',
                    'confidence': confidence,
                    'regime': regime_names[regime]
                }

        return None

    def execute_demo_trade(self, symbol, signal, price):
        """Execute demo trade and track outcomes"""
        timestamp = datetime.now().strftime('%H:%M:%S')

        print(f"\n🎯 [{timestamp}] DEMO TRADE:")
        print(f"   Symbol: {symbol}")
        print(f"   Action: {signal['action']}")
        print(f"   Price: ${price:,.2f}")
        print(f"   Regime: {signal['regime']}")
        print(f"   Confidence: {signal['confidence']:.1%}")

        self.stats['signals'] += 1

        # Track position for demo
        trade_data = {
            'action': signal['action'],
            'entry_price': price,
            'entry_time': timestamp,
            'regime': signal['regime'],
            'confidence': signal['confidence'],
            'size': 100  # Demo $100 position
        }

        if symbol not in self.positions:
            self.positions[symbol] = trade_data
            print(f"   ✅ Demo position opened")
        else:
            # Close existing position and calculate outcome
            pos = self.positions[symbol]
            pnl_pct = ((price - pos['entry_price']) / pos['entry_price']) * 100
            if pos['action'] == 'SELL':
                pnl_pct = -pnl_pct

            # Log trade outcome
            outcome = {'pnl_pct': pnl_pct}
            self.log_trade_outcome(symbol, pos, outcome)

            if pnl_pct > 0:
                self.stats['successful_trades'] += 1
                print(f"   ✅ Profitable trade: {pnl_pct:+.2f}%")
            else:
                self.stats['failed_trades'] += 1
                print(f"   ❌ Loss: {pnl_pct:+.2f}%")

            self.stats['total_pnl'] += pnl_pct

            # Open new position
            self.positions[symbol] = trade_data

    async def run(self):
        """Main demo trading loop"""
        print(f"""
        ╔═══════════════════════════════════════════╗
        ║   🔥 CHIMERA DEMO TRAINING - 2 WEEKS 🔥   ║
        ║   Started: {self.training_start.strftime('%Y-%m-%d %H:%M')}              ║
        ║   End: {(self.training_start + self.training_duration).strftime('%Y-%m-%d %H:%M')}                 ║
        ║   Model: {'Loaded ✓' if self.model else 'Training Mode'}                    ║
        ╚═══════════════════════════════════════════╝
        """)

        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'DOGEUSDT']

        try:
            async with websockets.connect(self.ws_url) as ws:
                print(f"✅ Connected to Bybit Demo")

                # Subscribe to tickers
                for symbol in symbols:
                    subscribe_msg = {
                        "op": "subscribe",
                        "args": [f"tickers.{symbol}"]
                    }
                    await ws.send(json.dumps(subscribe_msg))

                print(f"✅ Subscribed to: {', '.join(symbols)}")
                print("\n" + "="*60)
                print("🚀 DEMO TRADING STARTED - COLLECTING DATA FOR RETRAIN")
                print("="*60)

                while True:
                    # Check if training period complete
                    if self.check_training_complete():
                        self.trigger_retrain()
                        break

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

                                # Print progress every 500 ticks
                                if self.stats['ticks'] % 500 == 0:
                                    elapsed = datetime.now() - self.training_start
                                    remaining = self.training_duration - elapsed

                                    success_rate = (self.stats['successful_trades'] /
                                                    max(1, self.stats['successful_trades'] + self.stats['failed_trades']))

                                    print(f"\r📊 Ticks: {self.stats['ticks']} | "
                                          f"Trades: {self.stats['signals']} | "
                                          f"Success: {success_rate:.1%} | "
                                          f"Avg PnL: {self.stats['total_pnl']/max(1,self.stats['signals']):.2f}% | "
                                          f"Time left: {remaining.days}d {remaining.seconds//3600}h", end='')

                    except asyncio.TimeoutError:
                        await ws.send(json.dumps({"op": "ping"}))

        except KeyboardInterrupt:
            print("\n\n✅ Demo training stopped by user")
            if input("Trigger retrain now? (y/n): ").lower() == 'y':
                self.trigger_retrain()

        except Exception as e:
            print(f"\n❌ Error: {e}")

async def main():
    system = DemoTrainingSystem()
    await system.run()

if __name__ == "__main__":
    asyncio.run(main())