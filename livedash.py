#!/usr/bin/env python3
"""
CHIMERA Scalper Demo Training System with Live Learning + Dashboard Integration
Fixed version with corrected syntax and logic issues
"""

import asyncio
import websockets
import socketio
import eventlet
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
from socketio_dashboard import SocketIODashboardBridge

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

    def get_expert_performance(self, expert, lookback_trades=100):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
                       SELECT COUNT(*) as total,
                              SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                              AVG(pnl) as avg_pnl,
                              MAX(pnl) as max_win,
                              MIN(pnl) as max_loss
                       FROM (SELECT *
                             FROM trades
                             WHERE expert = ? AND status = 'CLOSED'
                             ORDER BY timestamp DESC LIMIT ?)
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

    def update_wallet(self, balance, equity, available_balance):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
                       INSERT INTO wallet (timestamp, balance, equity, available_balance)
                       VALUES (?, ?, ?, ?)
                       """, (time.time(), balance, equity, available_balance))
        conn.commit()
        conn.close()

    def get_wallet_history(self, limit=100):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
                       SELECT timestamp, balance, equity, available_balance
                       FROM wallet
                       ORDER BY timestamp DESC LIMIT ?
                       """, (limit,))
        results = cursor.fetchall()
        conn.close()
        return results


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
                try:
                    checkpoint = torch.load(latest_rl, map_location=device)
                    self.rl_learner.q_network.load_state_dict(checkpoint['model_state_dict'])
                    self.rl_learner.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.rl_learner.memory = checkpoint.get('memory', [])
                    print(f"Loaded RL model from {latest_rl}")
                except Exception as e:
                    print(f"Failed to load RL model: {e}")

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
        Path('models').mkdir(exist_ok=True)
        torch.save({
            'model_state_dict': self.rl_learner.q_network.state_dict(),
            'optimizer_state_dict': self.rl_learner.optimizer.state_dict(),
            'memory': self.rl_learner.memory[-1000:] if len(self.rl_learner.memory) > 1000 else self.rl_learner.memory
        }, f'models/rl_filter_{timestamp}.pth')


# ==================== DASHBOARD BRIDGE ====================
class DashboardBridge:
    def __init__(self, trading_system):
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
                    print(f"Dashboard bridge started on port {self.port}")
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
            subprocess.Popen(["node", "server.js"], cwd=dashboard_path, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            print("Starting Node.js dashboard server...")
            time.sleep(3)
            webbrowser.open("http://localhost:3001")
            print("Opening dashboard in browser: http://localhost:3001")
        except Exception as e:
            print(f"Failed to start dashboard: {e}")

    async def register_client(self, websocket):
        self.clients.add(websocket)
        print(f"Dashboard client connected: {websocket.remote_address}")
        try:
            await self.send_stats_update(websocket)
            await self.send_training_status(websocket)
            await self.send_learning_status(websocket)
            await self.send_wallet_update(websocket)
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
            print(f"Dashboard client disconnected")

    async def handle_client_message(self, websocket, data):
        if data.get('type') == 'get_stats':
            await self.send_stats_update(websocket)
        elif data.get('type') == 'get_trades':
            await self.send_recent_trades(websocket)
        elif data.get('type') == 'get_training_status':
            await self.send_training_status(websocket)
        elif data.get('type') == 'get_learning_status':
            await self.send_learning_status(websocket)
        elif data.get('type') == 'get_wallet':
            await self.send_wallet_update(websocket)
        elif data.get('type') == 'trigger_training':
            self.trading_system.trigger_immediate_training()
        elif data.get('type') == 'trigger_learning':
            self.trading_system.trigger_immediate_learning()

    async def send_wallet_update(self, websocket=None):
        try:
            wallet_data = {
                'balance': self.trading_system.wallet_balance,
                'equity': self.trading_system.equity,
                'available_balance': self.trading_system.available_balance,
                'unrealized_pnl': self.trading_system.unrealized_pnl,
                'total_pnl': self.trading_system.stats['total_pnl']
            }
            message = json.dumps({'type': 'wallet_update', 'data': wallet_data})
            await self._send_to_clients(message, websocket)
        except Exception as e:
            print(f"Error sending wallet update: {e}")

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
                    'next_training_in': max(0, self.trading_system.training_interval - (time.time() - self.trading_system.last_training_time))
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
                    'tick_rate': self.trading_system.stats['ticks'] / max(1, (
                            time.time() - self.trading_system.start_time)),
                    'success_rate': safe_div(self.trading_system.stats['successful_trades'],
                                             (self.trading_system.stats['successful_trades'] + self.trading_system.stats['failed_trades']), 0),
                    'data_points': self.trading_system.training_samples_collected,
                    'trade_outcomes': self.trading_system.stats['successful_trades'] + self.trading_system.stats[
                        'failed_trades'],
                    'training_progress': min(1, safe_div(
                        (time.time() - self.trading_system.training_start.timestamp()),
                        self.trading_system.training_duration.total_seconds(), 0.0)),
                    'days_remaining': max(0, (self.trading_system.training_duration.total_seconds() - (
                            time.time() - self.trading_system.training_start.timestamp())) / 86400)
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
                    'regime': trade_data.get('regime', 'UNKNOWN'),
                    'confidence': trade_data.get('confidence', 0),
                    'expert': trade_data.get('expert', 'AI_Regime'),
                    'pnl_pct': trade_data.get('pnl_pct'),
                    'cumulative_pnl': trade_data.get('cumulative_pnl')
                }
            })
            await self._send_to_clients(message)
        except Exception as e:
            print(f"Error sending trade signal: {e}")

    async def send_position_update(self, position_data):
        try:
            message = json.dumps({'type': 'position_update', 'data': position_data})
            await self._send_to_clients(message)
        except Exception as e:
            print(f"Error sending position update: {e}")

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

    def notify_position_update(self, position_data):
        if self.clients and self.loop and not self.loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(self.send_position_update(position_data), self.loop)
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

    def notify_wallet_update(self):
        if self.clients and self.loop and not self.loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(self.send_wallet_update(), self.loop)
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

        # Wallet and position tracking
        self.initial_balance = 10000.0  # Starting with $10,000 demo balance
        self.wallet_balance = self.initial_balance
        self.available_balance = self.initial_balance
        self.equity = self.initial_balance
        self.unrealized_pnl = 0.0
        self.leverage = 10  # 10x leverage for demo trading

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
        self.start_time = time.time()

        # Initialize Dashboard Bridge
        self.dashboard_bridge = SocketIODashboardBridge(self)
        self.dashboard_thread = self.dashboard_bridge.start()

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
        self._closing_flags = set()

        # Start background schedulers
        self.pos_lock = threading.RLock()
        self.training_thread = threading.Thread(target=self._training_scheduler, daemon=True)
        self.training_thread.start()

        self.learning_thread = threading.Thread(target=self._learning_scheduler, daemon=True)
        self.learning_thread.start()

        # Initialize wallet in database
        self.live_learner.db.update_wallet(self.wallet_balance, self.equity, self.available_balance)

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
        print("Database initialized for training data collection")

    def load_model(self):
        model_path = 'models/best_regime_classifier.pth'
        if not Path(model_path).exists():
            print("No existing model found, starting fresh")
            model = RegimeClassifier()
            model.train()
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
            model.train()
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
        spread_bps = safe_div((ask - bid), price, 0) * 10000

        if price <= 0:
            conn.close()
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

    def update_wallet(self):
        # Calculate total equity including unrealized PnL
        self.equity = self.wallet_balance + self.unrealized_pnl

        # Update available balance (wallet balance minus margin used)
        margin_used = sum([abs(pos['size'] * pos['entry_price']) / self.leverage for pos in self.positions.values()])
        self.available_balance = max(0, self.wallet_balance - margin_used)

        # Save to database
        self.live_learner.db.update_wallet(self.wallet_balance, self.equity, self.available_balance)

        # Notify dashboard
        self.dashboard.notify_wallet_update()

    def process_ticker(self, symbol, ticker_data):
        self.stats['ticks'] += 1

        # Extract data
        price = float(ticker_data.get('lastPrice', 0))
        bid = float(ticker_data.get('bid1Price', price))
        ask = float(ticker_data.get('ask1Price', price))
        volume = float(ticker_data.get('volume24h', 0))
        change_pct = float(ticker_data.get('price24hPcnt', 0))

        if price <= 0:
            return

        self.stats['last_prices'][symbol] = price

        # Update unrealized PnL for open positions
        if symbol in self.positions:
            pos = self.positions[symbol]
            if pos['action'] == 'BUY':
                self.unrealized_pnl = (price - pos['entry_price']) * pos['size']
            else:  # SELL
                self.unrealized_pnl = (pos['entry_price'] - price) * pos['size']
            self.update_wallet()

        # Create features (14 dimensions as in training)
        features = np.zeros(14)
        features[0] = change_pct
        features[1] = np.log1p(volume / 1e6)
        features[2] = safe_div((ask - bid), price, 0) * 10000
        features[3] = np.random.uniform(0, 1)  # Placeholder for additional features
        features[4] = np.random.uniform(0, 1)
        features[5] = np.random.uniform(0, 1)
        features[6] = np.random.uniform(0, 1)
        features[7] = np.random.uniform(0, 1)
        features[8] = np.random.uniform(0, 1)
        features[9] = np.random.uniform(0, 1)
        features[10] = np.random.uniform(0, 1)
        features[11] = 1.0 if symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'DOGEUSDT'] else 0.0
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
                    'confidence': np.random.uniform(0.6, 0.9),
                    'regime': 'UNKNOWN',
                    'expert': 'AI_Regime'
                }
                return signal, None, signal['confidence']
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

        # Generate signal based on regime
        signal = None
        if regime in [1, 3] and confidence > 0.7:  # BULL or BREAKOUT
            signal = {
                'action': 'BUY',
                'confidence': confidence,
                'regime': regime_name,
                'expert': 'AI_Regime'
            }
        elif regime in [0, 4] and confidence > 0.7:  # BEAR or REVERSAL
            signal = {
                'action': 'SELL',
                'confidence': confidence,
                'regime': regime_name,
                'expert': 'AI_Regime'
            }

        return signal, regime, confidence

    def execute_demo_trade(self, symbol, signal, price):
        self.stats['signals'] += 1
        if price is None or price <= 0:
            return

        # Process through reinforcement learning filter
        trade_data = {
            'symbol': symbol,
            'expert': signal['expert'],
            'direction': signal['action'],
            'entry_price': price,
            'stop_loss': price * (0.98 if signal['action'] == 'BUY' else 1.02),
            'take_profit': price * (1.02 if signal['action'] == 'BUY' else 0.98),
            'confidence': signal['confidence'],
            'regime': signal['regime'],
            'features': {
                'spread_bps': safe_div((price * 1.0001 - price * 0.9999), price, 0.0) * 10000,
                'volatility': np.random.uniform(0.5, 2.0),
                'vwap_deviation': np.random.uniform(-0.1, 0.1),
                'entropy': np.random.uniform(0, 1),
                'hurst': np.random.uniform(0.3, 0.7),
                'imbalance': np.random.uniform(-1, 1)
            }
        }

        trade_id, rl_state = self.live_learner.process_signal(trade_data)

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
                    'trade_id': trade_id,
                    'rl_state': rl_state
                }

            # Notify dashboard
            self.dashboard.notify_trade_signal({
                'symbol': symbol,
                'action': signal['action'],
                'price': price,
                'regime': signal['regime'],
                'confidence': signal['confidence'],
                'expert': signal['expert']
            })

            # Schedule position close (simulate trade duration)
            trade_duration = np.random.uniform(30, 300)  # 30 seconds to 5 minutes
            threading.Timer(trade_duration, self.close_position, args=[symbol]).start()

    def close_position(self, symbol):
        # Debounce duplicate/overlapping closes
        with self.pos_lock:
            if symbol in self._closing_flags:
                return
            self._closing_flags.add(symbol)

        try:
            with self.pos_lock:
                position = self.positions.get(symbol)
            if not position:
                # Already closed elsewhere
                return

            current_price = self.stats['last_prices'].get(symbol, position['entry_price'])
            if current_price <= 0:
                current_price = position['entry_price']

            # Calculate PnL
            if position['action'] == 'BUY':
                pnl = (current_price - position['entry_price']) * position['size']
                pnl_pct = safe_div(current_price, position['entry_price'], 1.0) - 1
            else:  # SELL
                pnl = (position['entry_price'] - current_price) * position['size']
                pnl_pct = safe_div(position['entry_price'], current_price, 1.0) - 1

            pnl_pct *= 100  # Convert to percentage

            # Update wallet
            with self.pos_lock:
                self.wallet_balance += pnl
                self.equity = self.wallet_balance + self.unrealized_pnl
                self.stats['total_pnl'] += pnl

                if pnl >= 0:
                    self.stats['successful_trades'] += 1
                else:
                    self.stats['failed_trades'] += 1

                # Update RL system with trade result
                if 'rl_state' in position:
                    self.live_learner.update_trade_result(
                        position['trade_id'], current_price, pnl, position['rl_state']
                    )

                # Remove position
                self.positions.pop(symbol, None)

            # Log trade outcome
            self.log_trade_outcome(symbol, {
                'action': position['action'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'regime': 'UNKNOWN',
                'confidence': 0.0
            }, {
                                       'pnl_pct': pnl_pct
                                   })

            print(f"[{symbol}] Closed {position['action']} PnL: {pnl:.2f} ({pnl_pct:.2f}%)")

            # Notify dashboard
            self.dashboard.notify_position_update({
                'symbol': symbol,
                'action': 'CLOSE',
                'price': current_price,
                'pnl': pnl
            })

        finally:
            with self.pos_lock:
                self._closing_flags.discard(symbol)

    def add_training_sample(self, sequence, regime):
        self.training_data.append(sequence)
        self.training_labels.append(regime)
        self.training_samples_collected += 1

        if len(self.training_data) >= self.min_training_samples and not self.is_training:
            self.trigger_immediate_training()

    def train_model(self):
        if len(self.training_data) < self.min_training_samples:
            print(f"Not enough training data: {len(self.training_data)}/{self.min_training_samples}")
            return

        print(f"Starting model training with {len(self.training_data)} samples...")
        self.is_training = True
        self.training_epoch = 0

        # Convert to tensors
        X = torch.FloatTensor(np.array(self.training_data))
        y = torch.LongTensor(np.array(self.training_labels))

        # Training loop
        for epoch in range(self.training_epochs):
            self.training_epoch = epoch + 1
            self.model.train()
            total_loss = 0

            # Mini-batch training
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i + self.batch_size].to(device)
                batch_y = y[i:i + self.batch_size].to(device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / max(1, len(X) // self.batch_size)
            self.current_training_loss = avg_loss

            print(f"Epoch {epoch + 1}/{self.training_epochs}, Loss: {avg_loss:.4f}")

        # Save model
        model_path = 'models/best_regime_classifier.pth'
        Path('models').mkdir(exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.training_epochs,
            'loss': self.current_training_loss
        }, model_path)

        print(f"Model trained and saved to {model_path}")
        self.is_training = False
        self.last_trained_time = datetime.now()
        self.last_training_time = time.time()

        # Clear training data
        self.training_data.clear()
        self.training_labels.clear()
        self.training_samples_collected = 0

    async def connect_to_bybit(self):
        print("Connecting to Bybit WebSocket...")
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

                        if 'topic' in data and 'data' in data:
                            topic = data['topic']
                            symbol = topic.split('.')[-1]
                            ticker_data = data['data']

                            self.process_ticker(symbol, ticker_data)

                    except asyncio.TimeoutError:
                        # Send ping to keep connection alive
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
        print("Starting CHIMERA Scalper Demo Training System")
        print("=" * 60)
        print(f"Dashboard: http://localhost:3001")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Training Samples: {self.min_training_samples}")
        print("=" * 60)

        # Start WebSocket connection in a separate thread
        ws_thread = threading.Thread(target=self.run_websocket_connection, daemon=True)
        ws_thread.start()

        # Main loop for periodic tasks
        try:
            while True:
                time.sleep(10)  # Main thread can perform other tasks or just sleep
                self.dashboard_bridge.notify_stats_update()

        except KeyboardInterrupt:
            print("\nShutting down CHIMERA system...")
        finally:
            # Cleanup if necessary
            pass

    def run_websocket_connection(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.connect_to_bybit())
        finally:
            loop.close()


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CHIMERA Scalper Demo Training System")
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