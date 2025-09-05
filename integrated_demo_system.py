#!/usr/bin/env python3
"""
CHIMERA Scalper Demo Training System with Dashboard Integration - FIXED
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
import threading
import subprocess
import webbrowser
import time
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

class DashboardBridge:
    def __init__(self, trading_system):
        self.trading_system = trading_system
        self.port = 8765
        self.clients = set()
        self.server = None
        self.loop = None

    def start_server(self):
        """Start the WebSocket server in a separate thread"""
        def run_server():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            async def websocket_handler(websocket, path=None):
                """WebSocket connection handler"""
                await self.register_client(websocket)

            async def start_websocket_server():
                try:
                    self.server = await websockets.serve(
                        websocket_handler,
                        "localhost",
                        self.port
                    )
                    print(f"📊 Dashboard bridge started on port {self.port}")

                    # Start Node.js server and open browser
                    self.start_dashboard()

                    await self.server.wait_closed()
                except Exception as e:
                    print(f"WebSocket server error: {e}")

            self.loop.run_until_complete(start_websocket_server())

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(2)  # Give server more time to start
        return server_thread

    def start_dashboard(self):
        """Start Node.js dashboard server and open browser"""
        try:
            # Start Node.js server in background
            dashboard_path = "chimera-dashboard"  # Adjust if needed
            subprocess.Popen(
                ["node", "server.js"],
                cwd=dashboard_path,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print("🚀 Starting Node.js dashboard server...")

            # Wait for server to start
            time.sleep(3)

            # Open browser
            webbrowser.open("http://localhost:3001")
            print("🌐 Opening dashboard in browser: http://localhost:3001")

        except Exception as e:
            print(f"❌ Failed to start dashboard: {e}")

    async def send_stats_update(self, websocket=None):
        """Send stats update to client(s) - FIXED"""
        try:
            stats = {
                'type': 'stats_update',
                'data': {
                    'ticks': self.trading_system.stats['ticks'],
                    'signals': self.trading_system.stats['signals'],
                    'successful_trades': self.trading_system.stats['successful_trades'],
                    'failed_trades': self.trading_system.stats['failed_trades'],
                    'total_pnl': self.trading_system.stats['total_pnl'],
                    'prices': dict(self.trading_system.stats['last_prices']),
                    'training_start': self.trading_system.training_start.isoformat()
                }
            }

            message = json.dumps(stats)

            if websocket:
                # Use proper WebSocket attribute check
                if hasattr(websocket, 'closed') and not websocket.closed:
                    await websocket.send(message)
            elif self.clients:
                tasks = []
                for client in self.clients.copy():
                    # Use proper WebSocket attribute check
                    if hasattr(client, 'closed') and not client.closed:
                        tasks.append(client.send(message))
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            print(f"Error sending stats update: {e}")

    # Apply the same fix to other send methods:
    async def send_regime_update(self, regime_data):
        """Send market regime update - FIXED"""
        try:
            message = json.dumps({
                'type': 'regime_update',
                'data': regime_data
            })

            if self.clients:
                tasks = []
                for client in self.clients.copy():
                    if hasattr(client, 'closed') and not client.closed:
                        tasks.append(client.send(message))
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            print(f"Error sending regime update: {e}")

    async def send_trade_signal(self, trade_data):
        """Send trade signal update"""
        try:
            message = json.dumps({
                'type': 'trade_signal',
                'data': trade_data
            })

            if self.clients:
                tasks = []
                for client in self.clients.copy():
                    if not client.closed:
                        tasks.append(client.send(message))
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            print(f"Error sending trade signal: {e}")

    async def send_price_update(self, price_data):
        """Send price update"""
        try:
            message = json.dumps({
                'type': 'price_update',
                'data': price_data
            })

            if self.clients:
                tasks = []
                for client in self.clients.copy():
                    if not client.closed:
                        tasks.append(client.send(message))
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            print(f"Error sending price update: {e}")

    async def send_recent_trades(self, websocket):
        """Send recent trades to specific client"""
        try:
            message = json.dumps({
                'type': 'recent_trades',
                'data': []
            })
            if not websocket.closed:
                await websocket.send(message)
        except Exception as e:
            print(f"Error sending recent trades: {e}")

    def notify_stats_update(self):
        """Called by trading system to notify stats update"""
        if self.clients and self.loop and not self.loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(
                    self.send_stats_update(),
                    self.loop
                )
            except Exception:
                pass

    def notify_trade_signal(self, trade_data):
        """Called by trading system to notify new trade"""
        if self.clients and self.loop and not self.loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(
                    self.send_trade_signal(trade_data),
                    self.loop
                )
            except Exception:
                pass

    def notify_regime_update(self, regime_data):
        """Called by trading system to notify regime change"""
        if self.clients and self.loop and not self.loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(
                    self.send_regime_update(regime_data),
                    self.loop
                )
            except Exception:
                pass

    def notify_price_update(self, price_data):
        """Called by trading system to notify price change"""
        if self.clients and self.loop and not self.loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(
                    self.send_price_update(price_data),
                    self.loop
                )
            except Exception:
                pass

class DemoTrainingSystem:
    def __init__(self):
        # FIXED: Updated to correct Bybit WebSocket URL
        self.ws_url = "wss://stream.bybit.com/v5/public/linear"

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

        # Dashboard integration
        self.dashboard = DashboardBridge(self)
        self.dashboard.start_server()

        # Stats broadcasting
        self.stats_broadcast_interval = 5  # seconds
        self.last_stats_broadcast = time.time()
        self.last_price_broadcast = time.time()

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

        # Create features
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

        # Broadcast updates to dashboard
        now = time.time()
        if now - self.last_stats_broadcast > self.stats_broadcast_interval:
            self.dashboard.notify_stats_update()
            self.last_stats_broadcast = now

        if now - self.last_price_broadcast > 2:  # Price updates every 2 seconds
            self.dashboard.notify_price_update({
                'prices': dict(self.stats['last_prices'])
            })
            self.last_price_broadcast = now

    def generate_signal(self, symbol, price):
        """Generate ML-based trading signal"""
        if self.model is None:
            # Random signals for testing when no model exists
            if np.random.random() > 0.995:  # Very rare for testing
                signal = {
                    'action': 'BUY' if np.random.random() > 0.5 else 'SELL',
                    'confidence': np.random.random(),
                    'regime': 'RANDOM_TEST'
                }
                # Notify dashboard of regime
                self.dashboard.notify_regime_update({
                    'regime': signal['regime'],
                    'confidence': signal['confidence'],
                    'symbol': symbol
                })
                return signal
            return None

        # ML prediction using existing model
        sequence = np.array(list(self.symbol_data[symbol]))
        input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            regime = output.argmax(1).item()
            confidence = probs[0, regime].item()

        regime_names = ['TREND_UP', 'TREND_DOWN', 'RANGE_BOUND', 'LIQUIDITY_EVENT', 'CHAOTIC']

        # Always notify dashboard of regime classification
        self.dashboard.notify_regime_update({
            'regime': regime_names[regime],
            'confidence': confidence,
            'symbol': symbol
        })

        # High confidence threshold for demo trading
        if confidence > 0.8:
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
            'symbol': symbol,
            'action': signal['action'],
            'entry_price': price,
            'entry_time': timestamp,
            'regime': signal['regime'],
            'confidence': signal['confidence'],
            'size': 100,  # Demo $100 position
            'timestamp': datetime.now().isoformat()
        }

        if symbol not in self.positions:
            self.positions[symbol] = {
                'action': signal['action'],
                'entry_price': price,
                'entry_time': timestamp,
                'regime': signal['regime'],
                'confidence': signal['confidence'],
                'size': 100
            }
            print(f"   ✅ Demo position opened")

            # Notify dashboard
            self.dashboard.notify_trade_signal(trade_data)
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
            self.positions[symbol] = {
                'action': signal['action'],
                'entry_price': price,
                'entry_time': timestamp,
                'regime': signal['regime'],
                'confidence': signal['confidence'],
                'size': 100
            }

            # Notify dashboard with PnL
            trade_data['pnl_pct'] = pnl_pct
            trade_data['cumulative_pnl'] = self.stats['total_pnl']
            self.dashboard.notify_trade_signal(trade_data)

    async def run(self):
        """Main demo trading loop"""
        print(f"""
        ╔═══════════════════════════════════════════╗
        ║   🔥 CHIMERA DEMO TRAINING - 2 WEEKS 🔥   ║
        ║   Started: {self.training_start.strftime('%Y-%m-%d %H:%M')}              ║
        ║   End: {(self.training_start + self.training_duration).strftime('%Y-%m-%d %H:%M')}                 ║
        ║   Model: {'Loaded ✓' if self.model else 'Training Mode'}                    ║
        ║   Dashboard: http://localhost:3001        ║
        ╚═══════════════════════════════════════════╝
        """)

        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'DOGEUSDT']

        # Try to connect to Bybit with retry logic
        max_retries = 5
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                print(f"🔗 Attempting to connect to Bybit (attempt {attempt + 1}/{max_retries})...")

                # Add timeout and better error handling
                async with websockets.connect(
                        self.ws_url,
                        ping_interval=20,
                        ping_timeout=20,
                        close_timeout=10,
                ) as ws:
                    print(f"✅ Connected to Bybit WebSocket")

                    # Subscribe to tickers
                    for symbol in symbols:
                        subscribe_msg = {
                            "op": "subscribe",
                            "args": [f"tickers.{symbol}"]
                        }
                        await ws.send(json.dumps(subscribe_msg))
                        print(f"✅ Subscribed to {symbol}")
                        await asyncio.sleep(0.1)  # Small delay between subscriptions

                    print(f"✅ Successfully subscribed to: {', '.join(symbols)}")
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
                                # Respond to ping
                                pong_msg = {"op": "pong"}
                                await ws.send(json.dumps(pong_msg))
                                continue

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
                            # Send ping to keep connection alive
                            ping_msg = {"op": "ping"}
                            await ws.send(json.dumps(ping_msg))
                        except json.JSONDecodeError as e:
                            print(f"JSON decode error: {e}")
                            continue
                        except KeyError as e:
                            print(f"Key error in message: {e}, data: {data}")
                            continue

            except asyncio.TimeoutError:
                print(f"❌ Connection timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    print(f"🔄 Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    print("❌ Failed to connect to Bybit after multiple attempts")
                    break
            except websockets.exceptions.InvalidStatusCode as e:
                print(f"❌ Invalid status code: {e.status_code}")
                if attempt < max_retries - 1:
                    print(f"🔄 Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    print("❌ Failed to connect to Bybit after multiple attempts")
                    break
            except ConnectionRefusedError:
                print(f"❌ Connection refused (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    print(f"🔄 Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    print("❌ Failed to connect to Bybit after multiple attempts")
                    break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                if attempt < max_retries - 1:
                    print(f"🔄 Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    print("❌ Failed to connect to Bybit after multiple attempts")
                    break

        # If we get here, we couldn't connect
        print("\n⚠️  Running in offline simulation mode")

        # Simulate market data for demo purposes
        while not self.check_training_complete():
            # Generate simulated ticker data
            for symbol in symbols:
                price = 50000 + np.random.normal(0, 1000) if symbol == 'BTCUSDT' else \
                    3000 + np.random.normal(0, 100) if symbol == 'ETHUSDT' else \
                        100 + np.random.normal(0, 10)

                ticker_data = {
                    'lastPrice': str(price),
                    'bid1Price': str(price - np.random.random() * 10),
                    'ask1Price': str(price + np.random.random() * 10),
                    'volume24h': str(np.random.uniform(1000000, 5000000)),
                    'price24hPcnt': str(np.random.normal(0, 0.02))
                }

                self.process_ticker(symbol, ticker_data)

                # Print progress every 100 ticks
                if self.stats['ticks'] % 100 == 0:
                    elapsed = datetime.now() - self.training_start
                    remaining = self.training_duration - elapsed

                    success_rate = (self.stats['successful_trades'] /
                                    max(1, self.stats['successful_trades'] + self.stats['failed_trades']))

                    print(f"\r📊 Ticks: {self.stats['ticks']} | "
                          f"Trades: {self.stats['signals']} | "
                          f"Success: {success_rate:.1%} | "
                          f"Avg PnL: {self.stats['total_pnl']/max(1,self.stats['signals']):.2f}% | "
                          f"Time left: {remaining.days}d {remaining.seconds//3600}h", end='')

                await asyncio.sleep(0.1)  # Simulate tick rate

        self.trigger_retrain()

async def main():
    system = DemoTrainingSystem()
    await system.run()

if __name__ == "__main__":
    asyncio.run(main())