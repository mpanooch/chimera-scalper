#!/usr/bin/env python3
"""
CHIMERA Scalper Demo System - Cleaned Version
No automatic training, just model loading and periodic saving
"""

import asyncio
import websockets
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
import warnings
import sqlite3
import threading
import time

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = 1e-8

def safe_div(a, b, default=0.0):
    try:
        if b == 0 or b == 0.0:
            return default
        return a / b
    except (ZeroDivisionError, TypeError, ValueError):
        return default

# ==================== MODEL ARCHITECTURE ====================
class RegimeClassifier(nn.Module):
    """Regime classifier for signal generation"""

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
        return self.classifier(last_hidden)


# ==================== LIVE LEARNING COMPONENTS ====================
class TradingDatabase:
    """Thread-safe SQLite database"""

    def __init__(self, db_path="chimera_trades.db"):
        self.db_path = db_path
        self.create_tables()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def create_tables(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS trades (
                                                             id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                             timestamp REAL,
                                                             symbol TEXT,
                                                             expert TEXT,
                                                             direction TEXT,
                                                             entry_price REAL,
                                                             stop_loss REAL,
                                                             take_profit REAL,
                                                             confidence REAL,
                                                             regime TEXT,
                                                             exit_price REAL,
                                                             pnl REAL,
                                                             status TEXT,
                                                             features TEXT
                       )
                       """)
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS wallet (
                                                             id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                             timestamp REAL,
                                                             balance REAL,
                                                             equity REAL,
                                                             available_balance REAL
                       )
                       """)
        conn.commit()
        conn.close()

    def record_trade(self, trade_data):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
                       INSERT INTO trades (timestamp, symbol, expert, direction, entry_price,
                                           stop_loss, take_profit, confidence, regime, status, features)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       """, (
                           time.time(),
                           trade_data['symbol'],
                           trade_data['expert'],
                           trade_data['direction'],
                           trade_data['entry_price'],
                           trade_data['stop_loss'],
                           trade_data['take_profit'],
                           trade_data['confidence'],
                           trade_data.get('regime', 'UNKNOWN'),
                           'OPEN',
                           json.dumps(trade_data.get('features', {}))
                       ))
        conn.commit()
        trade_id = cursor.lastrowid
        conn.close()
        return trade_id

    def update_trade(self, trade_id, exit_price, pnl, status='CLOSED'):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
                       UPDATE trades
                       SET exit_price = ?, pnl = ?, status = ?
                       WHERE id = ?
                       """, (exit_price, pnl, status, trade_id))
        conn.commit()
        conn.close()

    def update_wallet(self, balance, equity, available_balance):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
                       INSERT INTO wallet (timestamp, balance, equity, available_balance)
                       VALUES (?, ?, ?, ?)
                       """, (time.time(), balance, equity, available_balance))
        conn.commit()
        conn.close()


class LiveLearningSystem:
    """Simplified live learning system"""

    def __init__(self):
        self.db = TradingDatabase()
        self.experts = ['ROSS', 'BAO', 'NICK', 'FABIO', 'AI_Regime']
        self.total_trades_recorded = 0
        self.rl_acceptance_rate = 0.8  # Fixed acceptance rate

    def process_signal(self, signal_data):
        # Simple acceptance based on confidence
        if signal_data['confidence'] > 0.7:
            trade_id = self.db.record_trade(signal_data)
            self.total_trades_recorded += 1
            return trade_id, None
        return None, None

    def update_trade_result(self, trade_id, exit_price, pnl, original_state=None):
        self.db.update_trade(trade_id, exit_price, pnl)


# ==================== DASHBOARD BRIDGE ====================
# Web Dashboard Bridge for tradingtoday.com.au
from web_dashboard_bridge import WebDashboardBridge

class DashboardBridge:
    def __init__(self, trading_system, api_url="https://tradingtoday.com.au/api/chimera"):
        self.trading_system = trading_system
        self.web_bridge = WebDashboardBridge(trading_system, api_url)
        print(f"Dashboard bridge initialized - sending to {api_url}")

    def start_server(self):
        """Start the web dashboard bridge"""
        self.web_bridge.start()
        print("Web Dashboard Bridge started")

    def send_stats_update(self):
        """Update stats - handled automatically by web bridge"""
        pass

    def send_regime_update(self, regime, confidence):
        """Send regime update to web dashboard"""
        regime_data = {
            'current': regime,
            'confidence': confidence
        }
        self.web_bridge.update_regime(regime_data)

    def send_trade_update(self, trade_data):
        """Send trade update to web dashboard"""
        self.web_bridge.add_trade(trade_data)

    def send_trade_update(self, trade_data):
        if self.sio and self.connected:
            try:
                # Calculate cumulative PnL for the chart
                cumulative_pnl = (self.trading_system.equity / self.trading_system.initial_balance - 1) * 100

                trade_update = {
                    'symbol': trade_data['symbol'],
                    'action': trade_data['action'],
                    'price': trade_data['price'],
                    'regime': trade_data.get('regime', 'UNKNOWN'),
                    'confidence': trade_data.get('confidence', 0),
                    'expert': trade_data.get('expert', 'AI_Regime'),
                    'cumulative_pnl': cumulative_pnl
                }

                self.sio.emit('trade_update', trade_update)
            except Exception as e:
                print(f"Error sending trade update: {e}")
        elif self.clients and self.loop and not self.loop.is_closed():
            try:
                # Calculate cumulative PnL for the chart
                cumulative_pnl = (self.trading_system.equity / self.trading_system.initial_balance - 1) * 100

                asyncio.run_coroutine_threadsafe(self._send_to_clients(json.dumps({
                    'type': 'trade_update',
                    'data': {
                        'symbol': trade_data['symbol'],
                        'action': trade_data['action'],
                        'price': trade_data['price'],
                        'regime': trade_data.get('regime', 'UNKNOWN'),
                        'confidence': trade_data.get('confidence', 0),
                        'expert': trade_data.get('expert', 'AI_Regime'),
                        'cumulative_pnl': cumulative_pnl
                    }
                })), self.loop)
            except Exception:
                pass

# ==================== MAIN TRADING SYSTEM ====================
class DemoTrainingSystem:
    def __init__(self):
        self.ws_url = "wss://stream.bybit.com/v5/public/spot"
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

        # Wallet and position tracking
        self.initial_balance = 10000.0
        self.wallet_balance = self.initial_balance
        self.available_balance = self.initial_balance
        self.equity = self.initial_balance
        self.unrealized_pnl = 0.0
        self.leverage = 10

        # Load model
        self.model = self.load_model()
        self.current_confidence = 0.0

        # Data storage
        self.symbol_data = {}
        self.sequence_length = 60

        # Live learning system
        self.live_learner = LiveLearningSystem()

        # Timeline
        self.training_start = datetime.now()
        self.start_time = time.time()

        # Initialize database
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
        self._closing_flags = set()

        # Dashboard integration
        self.dashboard = DashboardBridge(self)
        self.dashboard.start_server()

        # Position lock
        self.pos_lock = threading.RLock()

        # Initialize wallet in database
        self.live_learner.db.update_wallet(self.wallet_balance, self.equity, self.available_balance)

        # Periodic model saving
        self.last_save_time = time.time()
        self.save_interval = 3600  # Save every hour
        self.save_thread = threading.Thread(target=self._periodic_save, daemon=True)
        self.save_thread.start()

    def _periodic_save(self):
        """Periodically save the model state"""
        while True:
            time.sleep(60)  # Check every minute
            if time.time() - self.last_save_time > self.save_interval:
                self.save_model_state()
                self.last_save_time = time.time()

    def save_model_state(self):
        """Save current model state and trading stats"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            Path('models').mkdir(exist_ok=True)

            # Save model state
            model_path = f'models/regime_classifier_{timestamp}.pth'
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'timestamp': timestamp,
                'stats': self.stats.copy(),
                'wallet_balance': self.wallet_balance,
                'total_trades': self.live_learner.total_trades_recorded
            }, model_path)

            print(f"Model state saved to {model_path}")
        except Exception as e:
            print(f"Error saving model state: {e}")

    def init_database(self):
        self.db_path = 'demo_training_data.db'
        conn = sqlite3.connect(self.db_path)
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
                                                                features TEXT
                     )
                     ''')
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
                                                           success INTEGER
                     )
                     ''')
        conn.commit()
        conn.close()
        print("Database initialized for data collection")

    def load_model(self):
        model_path = 'models/best_regime_classifier.pth'
        if not Path(model_path).exists():
            print("No existing model found, starting fresh")
            model = RegimeClassifier()
            model.eval()
            return model

        model = RegimeClassifier()
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            print(f"Existing model loaded from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}, starting fresh")
            model = RegimeClassifier()
            model.eval()
        return model

    def process_ticker(self, symbol, ticker_data):
        self.stats['ticks'] += 1

        # Debug output every 50 ticks
        if self.stats['ticks'] % 50 == 0:
            print(f"DEBUG {symbol} raw data: {ticker_data}")

        # Extract price
        price = 0
        if 'lastPrice' in ticker_data:
            try:
                price = float(ticker_data['lastPrice'])
            except (ValueError, TypeError):
                price = 0

        if price <= 0:
            return

        # Sanity checks for price ranges
        if symbol == 'BTCUSDT' and (price > 150000 or price < 30000):
            print(f"WARNING: Suspicious {symbol} price: ${price:,.2f} - skipping")
            return
        if symbol == 'ETHUSDT' and (price > 10000 or price < 1000):
            print(f"WARNING: Suspicious {symbol} price: ${price:,.2f} - skipping")
            return
        if symbol == 'SOLUSDT' and (price > 1000 or price < 50):
            print(f"WARNING: Suspicious {symbol} price: ${price:,.2f} - skipping")
            return

        self.stats['last_prices'][symbol] = price

        # Create features
        features = np.zeros(14)
        features[0] = float(ticker_data.get('price24hPcnt', 0)) * 100
        features[1] = np.log1p(float(ticker_data.get('volume24h', 0)) / 1e6)
        features[2] = 10  # Default spread
        features[11] = 1.0 if symbol in self.symbols else 0.0
        features[12] = 1.0

        # Update symbol data
        if symbol not in self.symbol_data:
            self.symbol_data[symbol] = deque(maxlen=self.sequence_length)
        self.symbol_data[symbol].append(features)

        # Generate signal if enough data
        if len(self.symbol_data[symbol]) >= self.sequence_length:
            signal, regime, confidence = self.generate_signal(symbol, price)
            self.current_confidence = confidence

            if signal:
                print(f"Signal generated: {symbol} {signal['action']} @ ${price:,.2f} ({signal['regime']}, {signal['confidence']:.1%})")
                self.execute_demo_trade(symbol, signal, price)

        # Update dashboard every 5 seconds
        if self.stats['ticks'] % 50 == 0:
            self.dashboard.send_stats_update()

    def generate_signal(self, symbol, price):
        if self.model is None:
            return None, None, 0.0

        # Prepare input sequence
        sequence = np.array(list(self.symbol_data[symbol]))
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)

        # Make prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(sequence_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            confidence = confidence.item()
            regime = predicted.item()

        # Map regime to trading signal
        regime_map = {0: 'BEAR', 1: 'BULL', 2: 'RANGING', 3: 'BREAKOUT', 4: 'REVERSAL'}
        regime_name = regime_map.get(regime, 'UNKNOWN')

        # Send regime update to dashboard
        self.dashboard.send_regime_update(regime_name, confidence)

        # Generate signal based on regime with more relaxed threshold
        signal = None
        if regime in [1, 3] and confidence > 0.65:  # BULL or BREAKOUT - reduced from 0.85 to 0.65
            signal = {
                'action': 'BUY',
                'confidence': confidence,
                'regime': regime_name,
                'expert': 'AI_Regime'
            }
        elif regime in [0, 4] and confidence > 0.65:  # BEAR or REVERSAL - reduced from 0.85 to 0.65
            signal = {
                'action': 'SELL',
                'confidence': confidence,
                'regime': regime_name,
                'expert': 'AI_Regime'
            }
        elif regime == 2 and confidence > 0.75:  # RANGING - generate signals for range trading
            # For ranging markets, alternate between buy and sell based on recent price action
            recent_prices = [self.stats['last_prices'].get(s, 0) for s in self.symbols if s in self.stats['last_prices']]
            if recent_prices:
                avg_price = sum(recent_prices) / len(recent_prices)
                action = 'BUY' if price < avg_price else 'SELL'
                signal = {
                    'action': action,
                    'confidence': confidence,
                    'regime': regime_name,
                    'expert': 'AI_Regime_Range'
                }

        return signal, regime, confidence

    def execute_demo_trade(self, symbol, signal, price):
        self.stats['signals'] += 1

        # Process through live learning filter
        trade_data = {
            'symbol': symbol,
            'expert': signal['expert'],
            'direction': signal['action'],
            'entry_price': price,
            'stop_loss': price * (0.98 if signal['action'] == 'BUY' else 1.02),
            'take_profit': price * (1.02 if signal['action'] == 'BUY' else 0.98),
            'confidence': signal['confidence'],
            'regime': signal['regime'],
            'features': {}
        }

        trade_id, _ = self.live_learner.process_signal(trade_data)

        if trade_id:
            # Calculate position size (1% of equity per trade)
            position_size = (self.equity * 0.01) * self.leverage / price

            # Record position
            with self.pos_lock:
                self.positions[symbol] = {
                    'action': signal['action'],
                    'entry_price': price,
                    'size': position_size,
                    'timestamp': time.time(),
                    'trade_id': trade_id
                }

            # Notify dashboard
            self.dashboard.send_trade_update({
                'symbol': symbol,
                'action': signal['action'],
                'price': price,
                'regime': signal['regime'],
                'confidence': signal['confidence'],
                'expert': signal['expert']
            })

            # Schedule position close
            trade_duration = np.random.uniform(60, 300)  # 1-5 minutes
            threading.Timer(trade_duration, self.close_position, args=[symbol]).start()

    def close_position(self, symbol):
        with self.pos_lock:
            if symbol in self._closing_flags:
                return
            self._closing_flags.add(symbol)

        try:
            with self.pos_lock:
                position = self.positions.get(symbol)
            if not position:
                return

            current_price = self.stats['last_prices'].get(symbol, position['entry_price'])
            if current_price <= 0:
                current_price = position['entry_price']

            # Calculate PnL
            if position['action'] == 'BUY':
                pnl = (current_price - position['entry_price']) * position['size']
            else:  # SELL
                pnl = (position['entry_price'] - current_price) * position['size']

            # Update wallet
            with self.pos_lock:
                self.wallet_balance += pnl
                self.equity = self.wallet_balance
                self.stats['total_pnl'] += pnl

                if pnl >= 0:
                    self.stats['successful_trades'] += 1
                else:
                    self.stats['failed_trades'] += 1

                # Update RL system
                self.live_learner.update_trade_result(position['trade_id'], current_price, pnl)

                # Remove position
                self.positions.pop(symbol, None)

            print(f"[{symbol}] Closed {position['action']} PnL: {pnl:.2f}")

        finally:
            with self.pos_lock:
                self._closing_flags.discard(symbol)

    async def connect_to_bybit(self):
        print("Connecting to Bybit WebSocket...")
        last_heartbeat = time.time()

        try:
            async with websockets.connect(self.ws_url) as ws:
                # Subscribe to ticker streams
                subscribe_message = {
                    "op": "subscribe",
                    "args": [f"tickers.{symbol}" for symbol in self.symbols]
                }
                await ws.send(json.dumps(subscribe_message))

                print(f"Subscribed to {len(self.symbols)} symbols: {', '.join(self.symbols)}")

                while True:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=30)
                        data = json.loads(message)
                        current_time = time.time()

                        # Handle subscription responses
                        if 'success' in data:
                            print(f"Subscription: {data.get('ret_msg', 'OK')}")

                        # Handle ticker data
                        if 'topic' in data and 'data' in data:
                            topic = data['topic']
                            symbol = topic.split('.')[-1]
                            ticker_data = data['data']
                            self.process_ticker(symbol, ticker_data)

                        # Heartbeat
                        if current_time - last_heartbeat > 30:
                            print(f"Heartbeat - Ticks: {self.stats['ticks']}, Signals: {self.stats['signals']}")
                            last_heartbeat = current_time

                    except asyncio.TimeoutError:
                        await ws.send(json.dumps({"op": "ping"}))
                    except Exception as e:
                        print(f"Error processing message: {e}")
                        continue

        except Exception as e:
            print(f"WebSocket connection error: {e}")
            print("Reconnecting in 5 seconds...")
            await asyncio.sleep(5)
            await self.connect_to_bybit()

    def run(self):
        print("Starting CHIMERA Scalper System")
        print("=" * 60)
        print(f"Dashboard: http://localhost:3001")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print("=" * 60)

        # Start WebSocket connection
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.connect_to_bybit())
        except KeyboardInterrupt:
            print("\nShutting down CHIMERA system...")
        finally:
            loop.close()


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CHIMERA Scalper System")
    parser.add_argument("--symbols", nargs="+", default=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
                        help="List of symbols to monitor")
    parser.add_argument("--balance", type=float, default=10000.0,
                        help="Initial demo balance")
    parser.add_argument("--leverage", type=int, default=10,
                        help="Trading leverage")

    args = parser.parse_args()

    system = DemoTrainingSystem()
    system.symbols = args.symbols
    system.initial_balance = args.balance
    system.wallet_balance = args.balance
    system.equity = args.balance
    system.available_balance = args.balance
    system.leverage = args.leverage

    system.run()


