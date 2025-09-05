#!/usr/bin/env python3
"""
CHIMERA Scalper Demo Training System with Live Learning + Dashboard Integration
Complete integration with training data, live learning, and dashboard compatibility
"""

import asyncio
import websockets
import json
import hmac
import hashlib
import requests
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
import warnings
import sqlite3
import threading
import time
import subprocess
import webbrowser
from sklearn.preprocessing import StandardScaler

# Import our Bybit authentication module
from bybit_auth import BybitTradingClient, BybitConfig, test_connection

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== MODEL ARCHITECTURE ====================
class RegimeClassifier(nn.Module):
    """Fixed regime classifier matching training script"""
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
        """Get a new connection for the current thread"""
        return sqlite3.connect(self.db_path)

    def create_tables(self):
        """Create tables using a new connection"""
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
                       CREATE TABLE IF NOT EXISTS performance (
                                                                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                                  timestamp REAL,
                                                                  expert TEXT,
                                                                  total_trades INTEGER,
                                                                  win_rate REAL,
                                                                  avg_pnl REAL,
                                                                  sharpe_ratio REAL,
                                                                  max_drawdown REAL
                       )
                       """)
        conn.commit()
        conn.close()

    def record_trade(self, trade_data):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
                       INSERT INTO trades (
                           timestamp, symbol, expert, direction, entry_price,
                           stop_loss, take_profit, confidence, regime, status, features
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                       UPDATE trades SET exit_price = ?, pnl = ?, status = ? WHERE id = ?
                       """, (exit_price, pnl, status, trade_id))
        conn.commit()
        conn.close()

    def get_expert_performance(self, expert, lookback_trades=100):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
                       SELECT COUNT(*) as total,
                              SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                              AVG(pnl) as avg_pnl,
                              MAX(pnl) as max_win,
                              MIN(pnl) as max_loss
                       FROM (
                                SELECT * FROM trades
                                WHERE expert = ? AND status = 'CLOSED'
                                ORDER BY timestamp DESC LIMIT ?
                            )
                       """, (expert, lookback_trades))

        result = cursor.fetchone()
        conn.close()

        if result and result[0] > 0:
            return {
                'total_trades': result[0],
                'win_rate': result[1] / result[0],
                'avg_pnl': result[2] or 0,
                'max_win': result[3] or 0,
                'max_loss': result[4] or 0
            }
        return None

class ReinforcementLearner:
    """Online reinforcement learning for trade filtering"""
    def __init__(self, state_dim=20, learning_rate=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = []
        self.max_memory = 10000

    def get_state(self, features, expert_stats):
        state = []
        state.extend([
            features.get('spread_bps', 0) / 100,
            features.get('imbalance', 0),
            features.get('volatility', 0),
            features.get('vwap_deviation', 0),
            features.get('entropy', 0),
            features.get('hurst', 0.5)
        ])
        state.extend([
            expert_stats.get('win_rate', 0.5),
            expert_stats.get('avg_pnl', 0),
            expert_stats.get('recent_performance', 0),
            expert_stats.get('confidence', 0.5)
        ])
        while len(state) < 20:
            state.append(0)
        return torch.FloatTensor(state[:20]).to(self.device)

    def predict(self, state):
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state.unsqueeze(0))
            return q_values.argmax(1).item()

    def train_step(self, state, action, reward, next_state):
        self.q_network.train()
        current_q = self.q_network(state.unsqueeze(0))[0, action]
        with torch.no_grad():
            next_q = self.q_network(next_state.unsqueeze(0)).max(1)[0]
        target_q = reward + 0.95 * next_q
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def store_experience(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)

    def replay_train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        total_loss = 0
        for state, action, reward, next_state in batch:
            loss = self.train_step(state, action, reward, next_state)
            total_loss += loss
        return total_loss / batch_size

class LiveLearningSystem:
    """Integrated live learning system"""
    def __init__(self):
        self.db = TradingDatabase()
        self.rl_learner = ReinforcementLearner()
        self.experts = ['ROSS', 'BAO', 'NICK', 'FABIO', 'AI_Regime']
        self.rl_current_loss = 0.0
        self.total_trades_recorded = 0
        self.rl_acceptance_rate = 0.0
        self.rl_accepted_trades = 0
        self.rl_total_signals = 0
        self.load_models()

    def load_models(self):
        model_dir = Path('models')
        if model_dir.exists():
            rl_files = list(model_dir.glob('rl_filter_*.pth'))
            if rl_files:
                latest_rl = sorted(rl_files, reverse=True)[0]
                checkpoint = torch.load(latest_rl)
                self.rl_learner.q_network.load_state_dict(checkpoint['model_state_dict'])
                self.rl_learner.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.rl_learner.memory = checkpoint.get('memory', [])

    def process_signal(self, signal_data):
        self.rl_total_signals += 1
        expert_stats = self.db.get_expert_performance(signal_data['expert'], 50) or {'win_rate': 0.5, 'avg_pnl': 0}
        state = self.rl_learner.get_state(signal_data['features'], expert_stats)
        action = self.rl_learner.predict(state)

        if action == 1:
            self.rl_accepted_trades += 1
            trade_id = self.db.record_trade(signal_data)
            self.total_trades_recorded += 1
            self.rl_acceptance_rate = self.rl_accepted_trades / self.rl_total_signals
            return trade_id, state
        return None, state

    def update_trade_result(self, trade_id, exit_price, pnl, original_state):
        self.db.update_trade(trade_id, exit_price, pnl)
        reward = np.tanh(pnl * 100)
        next_state = original_state
        self.rl_learner.store_experience(original_state, 1, reward, next_state)

    def periodic_learning(self):
        avg_loss = self.rl_learner.replay_train(batch_size=32)
        if avg_loss:
            self.rl_current_loss = avg_loss
        expert_stats = {}
        for expert in self.experts:
            stats = self.db.get_expert_performance(expert, 20)
            if stats:
                expert_stats[expert] = stats
        return expert_stats

    def get_expert_stats(self):
        stats = {}
        for expert in self.experts:
            perf = self.db.get_expert_performance(expert, 20)
            if perf:
                stats[expert] = perf
        return stats

    def save_models(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save({
            'model_state_dict': self.rl_learner.q_network.state_dict(),
            'optimizer_state_dict': self.rl_learner.optimizer.state_dict(),
            'memory': self.rl_learner.memory[-1000:] if len(self.rl_learner.memory) > 1000 else self.rl_learner.memory
        }, f'models/rl_filter_{timestamp}.pth')

# ==================== DASHBOARD BRIDGE ====================
class DashboardBridge:
    def __init__(self, host="0.0.0.0", port=8765, trading_system=None):
        self.trading_system = trading_system
        self.port = 8765
        self.clients = set()
        self.server = None
        self.loop = None

    def start_server(self):
        def run_server():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            async def websocket_handler(websocket, path=None):
                await self.register_client(websocket)

            async def start_websocket_server():
                try:
                    self.server = await websockets.serve(websocket_handler, "localhost", self.port)
                    print(f"📊 Dashboard bridge started on port {self.port}")
                    self.start_dashboard()
                    await self.server.wait_closed()
                except Exception as e:
                    print(f"WebSocket server error: {e}")

            self.loop.run_until_complete(start_websocket_server())

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(2)
        return server_thread

    def start_dashboard(self):
        try:
            dashboard_path = "chimera-dashboard"
            subprocess.Popen(["node", "server.js"], cwd=dashboard_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("🚀 Starting Node.js dashboard server...")
            time.sleep(3)
            webbrowser.open("http://localhost:3001")
            print("🌐 Opening dashboard in browser: http://localhost:3001")
        except Exception as e:
            print(f"❌ Failed to start dashboard: {e}")

    async def register_client(self, websocket):
        self.clients.add(websocket)
        print(f"📊 Dashboard client connected: {websocket.remote_address}")
        try:
            await self.send_stats_update(websocket)
            await self.send_training_status(websocket)
            await self.send_learning_status(websocket)
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    continue
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            print(f"Client connection error: {e}")
        finally:
            if websocket in self.clients:
                self.clients.remove(websocket)
            print(f"📊 Dashboard client disconnected")

    async def handle_client_message(self, websocket, data):
        if data.get('type') == 'get_stats':
            await self.send_stats_update(websocket)
        elif data.get('type') == 'get_trades':
            await self.send_recent_trades(websocket)
        elif data.get('type') == 'get_training_status':
            await self.send_training_status(websocket)
        elif data.get('type') == 'get_learning_status':
            await self.send_learning_status(websocket)
        elif data.get('type') == 'trigger_training':
            self.trading_system.trigger_immediate_training()
        elif data.get('type') == 'trigger_learning':
            self.trading_system.trigger_immediate_learning()

    async def send_learning_status(self, websocket=None):
        try:
            status = {
                'type': 'learning_status',
                'data': {
                    'rl_memory_size': len(self.trading_system.live_learner.rl_learner.memory),
                    'rl_training_loss': self.trading_system.live_learner.rl_current_loss,
                    'expert_performance': self.trading_system.live_learner.get_expert_stats(),
                    'total_trades_recorded': self.trading_system.live_learner.total_trades_recorded
                }
            }
            message = json.dumps(status)
            await self._send_to_clients(message, websocket)
        except Exception as e:
            print(f"Error sending learning status: {e}")

    async def send_training_status(self, websocket=None):
        try:
            status = {
                'type': 'training_status',
                'data': {
                    'is_training': self.trading_system.is_training,
                    'training_epoch': self.trading_system.training_epoch,
                    'training_loss': self.trading_system.current_training_loss,
                    'last_trained': self.trading_system.last_trained_time.isoformat() if self.trading_system.last_trained_time else None,
                    'training_samples': self.trading_system.training_samples_collected,
                    'next_training_in': max(0, self.trading_system.training_interval - (
                            time.time() - self.trading_system.last_training_time))
                }
            }
            message = json.dumps(status)
            await self._send_to_clients(message, websocket)
        except Exception as e:
            print(f"Error sending training status: {e}")

    async def send_stats_update(self, websocket=None):
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
                    'training_start': self.trading_system.training_start.isoformat(),
                    'model_confidence': self.trading_system.current_confidence,
                    'rl_acceptance_rate': self.trading_system.live_learner.rl_acceptance_rate,
                    'tick_rate': self.trading_system.stats['ticks'] / max(1, (time.time() - self.trading_system.start_time)),
                    'success_rate': (self.trading_system.stats['successful_trades'] / max(1, self.trading_system.stats['successful_trades'] + self.trading_system.stats['failed_trades'])),
                    'data_points': self.trading_system.training_samples_collected,
                    'trade_outcomes': self.trading_system.stats['successful_trades'] + self.trading_system.stats['failed_trades'],
                    'training_progress': min(1, (time.time() - self.trading_system.training_start.timestamp()) / self.trading_system.training_duration.total_seconds()),
                    'days_remaining': max(0, (self.trading_system.training_duration.total_seconds() - (time.time() - self.trading_system.training_start.timestamp())) / 86400)
                }
            }
            message = json.dumps(stats)
            await self._send_to_clients(message, websocket)
        except Exception as e:
            print(f"Error sending stats update: {e}")

    async def send_regime_update(self, regime_data):
        try:
            message = json.dumps({'type': 'regime_update', 'data': regime_data})
            await self._send_to_clients(message)
        except Exception as e:
            print(f"Error sending regime update: {e}")

    async def send_trade_signal(self, trade_data):
        try:
            message = json.dumps({
                'type': 'trade_update',
                'data': {
                    'symbol': trade_data['symbol'],
                    'action': trade_data['action'],
                    'price': trade_data['price'],
                    'regime': trade_data.get('regime', 'RANDOM_TEST'),
                    'confidence': trade_data.get('confidence', 0),
                    'expert': trade_data.get('expert', 'AI_Regime'),
                    'pnl_pct': trade_data.get('pnl_pct'),
                    'cumulative_pnl': trade_data.get('cumulative_pnl')
                }
            })
            await self._send_to_clients(message)
        except Exception as e:
            print(f"Error sending trade signal: {e}")

    async def send_price_update(self, price_data):
        try:
            message = json.dumps({'type': 'price_update', 'data': price_data})
            await self._send_to_clients(message)
        except Exception as e:
            print(f"Error sending price update: {e}")

    async def send_recent_trades(self, websocket):
        try:
            message = json.dumps({'type': 'recent_trades', 'data': []})
            if hasattr(websocket, 'closed') and not websocket.closed:
                await websocket.send(message)
        except Exception as e:
            print(f"Error sending recent trades: {e}")

    async def _send_to_clients(self, message, websocket=None):
        if websocket and hasattr(websocket, 'closed') and not websocket.closed:
            await websocket.send(message)
        elif self.clients:
            tasks = []
            for client in self.clients.copy():
                if hasattr(client, 'closed') and not client.closed:
                    tasks.append(client.send(message))
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    def notify_stats_update(self):
        if self.clients and self.loop and not self.loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(self.send_stats_update(), self.loop)
            except Exception:
                pass

    def notify_trade_signal(self, trade_data):
        if self.clients and self.loop and not self.loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(self.send_trade_signal(trade_data), self.loop)
            except Exception:
                pass

    def notify_regime_update(self, regime_data):
        if self.clients and self.loop and not self.loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(self.send_regime_update(regime_data), self.loop)
            except Exception:
                pass

    def notify_price_update(self, price_data):
        if self.clients and self.loop and not self.loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(self.send_price_update(price_data), self.loop)
            except Exception:
                pass

    def notify_learning_update(self):
        if self.clients and self.loop and not self.loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(self.send_learning_status(), self.loop)
            except Exception:
                pass

# ==================== MAIN TRADING SYSTEM ====================
class DemoTrainingSystem:
    def __init__(self):
        self.ws_url = "wss://stream.bybit.com/v5/public/linear"
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'DOGEUSDT']

        # Training configuration
        self.training_interval = 3600
        self.learning_interval = 1800
        self.min_training_samples = 1000
        self.training_epochs = 10
        self.batch_size = 32

        # Training state
        self.is_training = False
        self.training_epoch = 0
        self.current_training_loss = 0.0
        self.last_stats_broadcast = time.time()
        self.last_price_broadcast = time.time()
        self.last_training_time = 0
        self.last_learning_time = 0
        self.last_trained_time = None
        self.training_samples_collected = 0
        self.current_confidence = 0.0
        self.start_time = time.time()

        # Load or create model
        self.model = self.load_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001) if self.model else None
        self.criterion = nn.CrossEntropyLoss()

        # Data storage
        self.training_data = []
        self.training_labels = []
        self.symbol_data = {}
        self.sequence_length = 60
        self.scaler = StandardScaler()

        # Live learning system
        self.live_learner = LiveLearningSystem()

        # Training timeline
        self.training_start = datetime.now()
        self.training_duration = timedelta(days=14)

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
        self.open_trades = {}

        # Dashboard integration
        self.dashboard = DashboardBridge(self)
        self.dashboard.start_server()

        # Start background schedulers
        self.training_thread = threading.Thread(target=self._training_scheduler, daemon=True)
        self.training_thread.start()

        self.learning_thread = threading.Thread(target=self._learning_scheduler, daemon=True)
        self.learning_thread.start()

    def _training_scheduler(self):
        while True:
            time_since_last_train = time.time() - self.last_training_time
            if time_since_last_train >= self.training_interval and len(self.training_data) >= self.min_training_samples:
                self.train_model()
            time.sleep(60)

    def _learning_scheduler(self):
        while True:
            time_since_last_learn = time.time() - self.last_learning_time
            if time_since_last_learn >= self.learning_interval:
                self.trigger_immediate_learning()
            time.sleep(60)

    def trigger_immediate_learning(self):
        expert_stats = self.live_learner.periodic_learning()
        self.last_learning_time = time.time()
        self.dashboard.notify_learning_update()
        return expert_stats

    def trigger_immediate_training(self):
        if len(self.training_data) >= self.min_training_samples and not self.is_training:
            threading.Thread(target=self.train_model, daemon=True).start()

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
        print("✅ Database initialized for training data collection")

    def load_model(self):
        model_path = 'models/best_regime_classifier.pth'
        if not Path(model_path).exists():
            print("⚠️ No existing model found, starting fresh")
            model = RegimeClassifier()
            model.train()
            return model

        model = RegimeClassifier()
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print(f"✅ Existing model loaded from {model_path}")
        return model

    def save_market_data(self, symbol, ticker_data, features):
        conn = sqlite3.connect(self.db_path)
        price = float(ticker_data.get('lastPrice') or
                      ticker_data.get('markPrice') or
                      ticker_data.get('bid1Price') or
                      ticker_data.get('ask1Price') or 0)
        bid = float(ticker_data.get('bid1Price') or price or 0)
        ask = float(ticker_data.get('ask1Price') or price or 0)
        volume = float(ticker_data.get('volume24h', 0))
        change_pct = float(ticker_data.get('price24hPcnt', 0))
        spread_bps = ((ask - bid) / price) * 10000 if price > 0 else 0
        if price <= 0:
            return
        conn.execute('''
                     INSERT INTO market_data (timestamp, symbol, price, bid, ask, volume, change_pct, spread_bps, features)
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
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
                     INSERT INTO trades (timestamp, symbol, action, entry_price, exit_price, regime, confidence, pnl_pct, success)
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
        elapsed = datetime.now() - self.training_start
        return elapsed >= self.training_duration

    def trigger_retrain(self):
        print("\n🔄 2-week demo period complete! Triggering model retrain...")
        conn = sqlite3.connect(self.db_path)
        market_df = pd.read_sql_query("SELECT * FROM market_data ORDER BY timestamp", conn)
        trades_df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp", conn)
        conn.close()

        print(f"📊 Collected {len(market_df)} market data points")
        print(f"📊 Executed {len(trades_df)} trades")

        success_rate = trades_df['success'].mean() if len(trades_df) > 0 else 0
        avg_pnl = trades_df['pnl_pct'].mean() if len(trades_df) > 0 else 0

        print(f"📈 Success rate: {success_rate:.1%}")
        print(f"📈 Average PnL: {avg_pnl:+.2f}%")

        market_df.to_csv('demo_market_data.csv', index=False)
        trades_df.to_csv('demo_trades_log.csv', index=False)

        print("✅ Data exported for retraining. Run retrain_model.py next!")
        return True

    def process_ticker(self, symbol, ticker_data):
        self.stats['ticks'] += 1

        # Extract data
        price = float(ticker_data.get('lastPrice', 0))
        bid = float(ticker_data.get('bid1Price', price))
        ask = float(ticker_data.get('ask1Price', price))
        volume = float(ticker_data.get('volume24h', 0))
        change_pct = float(ticker_data.get('price24hPcnt', 0))

        self.stats['last_prices'][symbol] = price

        # Create features (14 dimensions as in training)
        features = np.zeros(14)
        features[0] = change_pct
        features[1] = np.log1p(volume / 1e6)
        features[2] = ((ask - bid) / price) * 10000 if price > 0 else 0
        features[3] = np.random.uniform(0, 1)  # Placeholder for additional features
        features[4] = np.random.uniform(0, 1)
        features[5] = np.random.uniform(0, 1)
        features[6] = np.random.uniform(0, 1)
        features[7] = np.random.uniform(0, 1)
        features[8] = np.random.uniform(0, 1)
        features[9] = np.random.uniform(0, 1)
        features[10] = np.random.uniform(0, 1)
        features[11] = 1.0 if symbol in ['BTCUSDT', 'ETHUSDT'] else 0.0
        features[12] = 1.0
        features[13] = np.random.uniform(0, 1)

        # Save data for retraining
        self.save_market_data(symbol, ticker_data, features)

        # Update symbol data
        if symbol not in self.symbol_data:
            self.symbol_data[symbol] = deque(maxlen=self.sequence_length)
        self.symbol_data[symbol].append(features)

        # Generate signal if enough data
        if len(self.symbol_data[symbol]) >= self.sequence_length:
            signal, regime, confidence = self.generate_signal(symbol, price)
            self.current_confidence = confidence

            if signal:
                self.execute_demo_trade(symbol, signal, price)

            # Add to training data
            if regime is not None:
                self.add_training_sample(np.array(list(self.symbol_data[symbol])), regime)

        # Broadcast updates to dashboard
        now = time.time()
        if now - self.last_stats_broadcast > 5:
            self.dashboard.notify_stats_update()
            self.last_stats_broadcast = now

        if now - self.last_price_broadcast > 2:
            self.dashboard.notify_price_update({'prices': dict(self.stats['last_prices'])})
            self.last_price_broadcast = now

    def generate_signal(self, symbol, price):
        if self.model is None:
            if np.random.random() > 0.995:
                signal = {
                    'action': 'BUY' if np.random.random() > 0.5 else 'SELL',
                    'confidence': np.random.random(),
                    'regime': 'RANDOM_TEST'
                }
                self.dashboard.notify_regime_update({
                    'regime': signal['regime'],
                    'confidence': signal['confidence'],
                    'symbol': symbol
                })
                return signal, None, signal['confidence']
            return None, None, 0.0

        sequence = np.array(list(self.symbol_data[symbol]))
        input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            regime = output.argmax(1).item()
            confidence = probs[0, regime].item()

        regime_names = ['TREND_UP', 'TREND_DOWN', 'RANGE_BOUND', 'LIQUIDITY_EVENT', 'CHAOTIC']

        # Notify dashboard
        self.dashboard.notify_regime_update({
            'regime': regime_names[regime],
            'confidence': confidence,
            'symbol': symbol
        })

        # High confidence threshold
        if confidence > 0.7:
            if regime == 0:  # TREND_UP
                return {
                    'action': 'BUY',
                    'confidence': confidence,
                    'regime': regime_names[regime]
                }, regime, confidence
            elif regime == 1:  # TREND_DOWN
                return {
                    'action': 'SELL',
                    'confidence': confidence,
                    'regime': regime_names[regime]
                }, regime, confidence

        return None, regime, confidence

    def add_training_sample(self, features, regime):
        self.training_data.append(features)
        self.training_labels.append(regime)
        self.training_samples_collected += 1

        if len(self.training_data) > 50000:
            self.training_data = self.training_data[-50000:]
            self.training_labels = self.training_labels[-50000:]

    def execute_demo_trade(self, symbol, signal, price):
        timestamp = datetime.now().strftime('%H:%M:%S')

        print(f"\n🎯 [{timestamp}] DEMO TRADE:")
        print(f"   Symbol: {symbol}")
        print(f"   Action: {signal['action']}")
        print(f"   Price: ${price:,.2f}")
        print(f"   Regime: {signal['regime']}")
        print(f"   Confidence: {signal['confidence']:.1%}")

        self.stats['signals'] += 1

        # Prepare signal for RL
        trade_signal = {
            'symbol': symbol,
            'expert': 'AI_Regime',
            'direction': 'LONG' if signal['action'] == 'BUY' else 'SHORT',
            'entry_price': price,
            'stop_loss': price * 0.99 if signal['action'] == 'BUY' else price * 1.01,
            'take_profit': price * 1.02 if signal['action'] == 'BUY' else price * 0.98,
            'confidence': signal['confidence'],
            'regime': signal['regime'],
            'features': {
                'spread_bps': np.random.uniform(50, 200),
                'imbalance': np.random.uniform(0.8, 1.0),
                'volatility': np.random.uniform(0.01, 0.03),
                'vwap_deviation': np.random.uniform(-0.005, 0.005),
                'entropy': np.random.uniform(3.0, 4.0),
                'hurst': np.random.uniform(0.4, 0.6)
            }
        }

        # Process with RL filter
        trade_id, rl_state = self.live_learner.process_signal(trade_signal)

        if trade_id:
            # Track position
            trade_data = {
                'symbol': symbol,
                'action': signal['action'],
                'entry_price': price,
                'entry_time': timestamp,
                'regime': signal['regime'],
                'confidence': signal['confidence'],
                'size': 100,
                'timestamp': datetime.now().isoformat(),
                'rl_trade_id': trade_id,
                'rl_state': rl_state
            }

            if symbol not in self.positions:
                self.positions[symbol] = trade_data
                self.open_trades[trade_id] = trade_data
                print(f"   ✅ Demo position opened")

                # Notify dashboard
                dashboard_trade_data = {
                    'symbol': symbol,
                    'action': signal['action'],
                    'price': price,
                    'regime': signal['regime'],
                    'confidence': signal['confidence'],
                    'expert': 'AI_Regime'
                }
                self.dashboard.notify_trade_signal(dashboard_trade_data)
            else:
                # Close existing position
                self.close_position(symbol, price, trade_id, rl_state)

    def close_position(self, symbol, price, new_trade_id, new_rl_state):
        pos = self.positions[symbol]

        # Calculate PnL
        pnl_pct = ((price - pos['entry_price']) / pos['entry_price']) * 100
        if pos['action'] == 'SELL':
            pnl_pct = -pnl_pct

        pnl_decimal = pnl_pct / 100

        # Update RL learner with trade result
        if 'rl_trade_id' in pos:
            self.live_learner.update_trade_result(
                pos['rl_trade_id'],
                price,
                pnl_decimal,
                pos.get('rl_state')
            )

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
            'action': self.positions[symbol]['action'],
            'entry_price': price,
            'entry_time': datetime.now().strftime('%H:%M:%S'),
            'regime': self.positions[symbol]['regime'],
            'confidence': self.positions[symbol]['confidence'],
            'size': 100,
            'rl_trade_id': new_trade_id,
            'rl_state': new_rl_state
        }

        # Notify dashboard
        trade_data = {
            'symbol': symbol,
            'action': self.positions[symbol]['action'],
            'price': price,
            'pnl_pct': pnl_pct,
            'cumulative_pnl': self.stats['total_pnl'],
            'regime': self.positions[symbol]['regime'],
            'confidence': self.positions[symbol]['confidence'],
            'expert': 'AI_Regime'
        }
        self.dashboard.notify_trade_signal(trade_data)

    def prepare_training_data(self):
        if len(self.training_data) < self.min_training_samples:
            return None, None

        X = np.array(self.training_data)
        y = np.array(self.training_labels)

        if len(self.training_data) > 1000:
            X = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

        return X, y

    def train_model(self):
        if self.is_training:
            return

        self.is_training = True
        print(f"\n🎯 Starting live training with {len(self.training_data)} samples...")

        try:
            X, y = self.prepare_training_data()
            if X is None:
                print("⚠️ Not enough data for training")
                self.is_training = False
                return

            X_tensor = torch.FloatTensor(X).to(device)
            y_tensor = torch.LongTensor(y).to(device)

            for epoch in range(self.training_epochs):
                self.training_epoch = epoch + 1

                for i in range(0, len(X), self.batch_size):
                    batch_X = X_tensor[i:i+self.batch_size]
                    batch_y = y_tensor[i:i+self.batch_size]

                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()

                    self.current_training_loss = loss.item()

                print(f"   Epoch {epoch+1}/{self.training_epochs}, Loss: {loss.item():.4f}")

                self.dashboard.notify_stats_update()
                self.dashboard.notify_training_status()

            self.last_trained_time = datetime.now()
            self.save_model()

            print(f"✅ Training complete! Final loss: {loss.item():.4f}")

        except Exception as e:
            print(f"❌ Training error: {e}")
        finally:
            self.is_training = False
            self.last_training_time = time.time()

    def save_model(self):
        model_path = 'models/live_trained_model.pth'
        Path('models').mkdir(exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_samples': len(self.training_data),
            'timestamp': datetime.now().isoformat()
        }, model_path)

        print(f"💾 Model saved to {model_path}")

    async def run(self):
        print(f"""
        ╔═══════════════════════════════════════════╗
        ║   🔥 CHIMERA DEMO TRAINING - 2 WEEKS 🔥   ║
        ║   Started: {self.training_start.strftime('%Y-%m-%d %H:%M')}              ║
        ║   End: {(self.training_start + self.training_duration).strftime('%Y-%m-%d %H:%M')}                 ║
        ║   Model: {'Loaded ✓' if self.model else 'Training Mode'}                    ║
        ║   Dashboard: http://localhost:3001        ║
        ╚═══════════════════════════════════════════╝
        """)

        # Try to connect to Bybit with retry logic
        max_retries = 5
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                print(f"🔗 Attempting to connect to Bybit (attempt {attempt + 1}/{max_retries})...")

                async with websockets.connect(
                        self.ws_url,
                        ping_interval=20,
                        ping_timeout=20,
                        close_timeout=10,
                ) as ws:
                    print(f"✅ Connected to Bybit WebSocket")

                    # Subscribe to tickers
                    for symbol in self.symbols:
                        subscribe_msg = {
                            "op": "subscribe",
                            "args": [f"tickers.{symbol}"]
                        }
                        await ws.send(json.dumps(subscribe_msg))
                        print(f"✅ Subscribed to {symbol}")
                        await asyncio.sleep(0.1)

                    print(f"✅ Successfully subscribed to: {', '.join(self.symbols)}")
                    print("\n" + "="*60)
                    print("🚀 DEMO TRADING STARTED - COLLECTING DATA FOR RETRAIN")
                    print("="*60)

                    while True:
                        if self.check_training_complete():
                            self.trigger_retrain()
                            break

                        try:
                            message = await asyncio.wait_for(ws.recv(), timeout=30)
                            data = json.loads(message)

                            if data.get('op') == 'ping':
                                pong_msg = {"op": "pong"}
                                await ws.send(json.dumps(pong_msg))
                                continue

                            elif 'topic' in data and 'data' in data:
                                topic = data['topic']
                                if 'tickers' in topic:
                                    symbol = topic.split('.')[-1]
                                    ticker_data = data['data']
                                    self.process_ticker(symbol, ticker_data)

                                    if self.stats['ticks'] % 500 == 0:
                                        elapsed = datetime.now() - self.training_start
                                        remaining = self.training_duration - elapsed
                                        success_rate = (self.stats['successful_trades'] / max(1, self.stats['successful_trades'] + self.stats['failed_trades']))
                                        print(f"\r📊 Ticks: {self.stats['ticks']} | Trades: {self.stats['signals']} | Success: {success_rate:.1%} | Avg PnL: {self.stats['total_pnl']/max(1,self.stats['signals']):.2f}% | Time left: {remaining.days}d {remaining.seconds//3600}h", end='')

                        except asyncio.TimeoutError:
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

        # Fallback to simulation mode
        print("\n⚠️  Running in offline simulation mode")
        while not self.check_training_complete():
            for symbol in self.symbols:
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

                if self.stats['ticks'] % 100 == 0:
                    elapsed = datetime.now() - self.training_start
                    remaining = self.training_duration - elapsed
                    success_rate = (self.stats['successful_trades'] / max(1, self.stats['successful_trades'] + self.stats['failed_trades']))
                    print(f"\r📊 Ticks: {self.stats['ticks']} | Trades: {self.stats['signals']} | Success: {success_rate:.1%} | Avg PnL: {self.stats['total_pnl']/max(1,self.stats['signals']):.2f}% | Time left: {remaining.days}d {remaining.seconds//3600}h", end='')

                await asyncio.sleep(0.1)

        self.trigger_retrain()

async def main():
    system = DemoTrainingSystem()
    await system.run()

if __name__ == "__main__":
    asyncio.run(main())