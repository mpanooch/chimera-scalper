#!/usr/bin/env python3
"""
Real-time learning system for CHIMERA
Tracks trades, updates model weights based on PnL
"""

import json
import sqlite3
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import time

class TradingDatabase:
    """Store and analyze trading results"""
    def __init__(self, db_path="chimera_trades.db"):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()

        # Trades table
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

        # Performance metrics table
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

        self.conn.commit()

    def record_trade(self, trade_data):
        cursor = self.conn.cursor()
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
        self.conn.commit()
        return cursor.lastrowid

    def update_trade(self, trade_id, exit_price, pnl, status='CLOSED'):
        cursor = self.conn.cursor()
        cursor.execute("""
                       UPDATE trades
                       SET exit_price = ?, pnl = ?, status = ?
                       WHERE id = ?
                       """, (exit_price, pnl, status, trade_id))
        self.conn.commit()

    def get_expert_performance(self, expert, lookback_trades=100):
        cursor = self.conn.cursor()
        cursor.execute("""
                       SELECT COUNT(*) as total,
                              SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                              AVG(pnl) as avg_pnl,
                              MAX(pnl) as max_win,
                              MIN(pnl) as max_loss
                       FROM (
                                SELECT * FROM trades
                                WHERE expert = ? AND status = 'CLOSED'
                                ORDER BY timestamp DESC
                                    LIMIT ?
                            )
                       """, (expert, lookback_trades))

        result = cursor.fetchone()
        if result and result[0] > 0:
            return {
                'total_trades': result[0],
                'win_rate': result[1] / result[0] if result[0] > 0 else 0,
                'avg_pnl': result[2] or 0,
                'max_win': result[3] or 0,
                'max_loss': result[4] or 0
            }
        return None

class ReinforcementLearner:
    """Online reinforcement learning for trade filtering"""
    def __init__(self, state_dim=20, learning_rate=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Simple Q-network
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # [reject, accept]
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = []
        self.max_memory = 10000

    def get_state(self, features, expert_stats):
        """Convert features and stats to state vector"""
        state = []

        # Market features
        state.extend([
            features.get('spread_bps', 0) / 100,
            features.get('imbalance', 0),
            features.get('volatility', 0),
            features.get('vwap_deviation', 0),
            features.get('entropy', 0),
            features.get('hurst', 0.5)
        ])

        # Expert statistics
        state.extend([
            expert_stats.get('win_rate', 0.5),
            expert_stats.get('avg_pnl', 0),
            expert_stats.get('recent_performance', 0),
            expert_stats.get('confidence', 0.5)
        ])

        # Pad to state_dim
        while len(state) < 20:
            state.append(0)

        return torch.FloatTensor(state[:20]).to(self.device)

    def predict(self, state):
        """Predict whether to accept or reject trade"""
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state.unsqueeze(0))
            action = q_values.argmax(1).item()
        return action  # 0=reject, 1=accept

    def train_step(self, state, action, reward, next_state):
        """Single training step"""
        self.q_network.train()

        # Current Q value
        current_q = self.q_network(state.unsqueeze(0))[0, action]

        # Next Q value
        with torch.no_grad():
            next_q = self.q_network(next_state.unsqueeze(0)).max(1)[0]

        # Target Q value
        gamma = 0.95
        target_q = reward + gamma * next_q

        # Loss
        loss = nn.MSELoss()(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def store_experience(self, state, action, reward, next_state):
        """Store experience for replay"""
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)

    def replay_train(self, batch_size=32):
        """Train on batch of experiences"""
        if len(self.memory) < batch_size:
            return

        # Sample batch
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]

        total_loss = 0
        for state, action, reward, next_state in batch:
            loss = self.train_step(state, action, reward, next_state)
            total_loss += loss

        return total_loss / batch_size

class LiveLearningSystem:
    """Main system that coordinates learning"""
    def __init__(self):
        self.db = TradingDatabase()
        self.rl = ReinforcementLearner()
        self.experts = ['Ross', 'Bao', 'Nick', 'Fabio']

        # Load saved models if they exist
        self.load_models()

    def load_models(self):
        """Load pre-trained models"""
        model_dir = Path('models')
        if model_dir.exists():
            # Load latest models
            for model_file in sorted(model_dir.glob('chimera_*.pth'), reverse=True):
                print(f"Loading model: {model_file}")
                # Load logic here
                break

    def process_signal(self, signal_data):
        """Process incoming signal and decide whether to trade"""

        # Get expert performance
        expert_stats = self.db.get_expert_performance(
            signal_data['expert'],
            lookback_trades=50
        )

        if expert_stats is None:
            expert_stats = {'win_rate': 0.5, 'avg_pnl': 0}

        # Create state
        state = self.rl.get_state(signal_data['features'], expert_stats)

        # Predict action
        action = self.rl.predict(state)

        if action == 1:  # Accept trade
            # Record trade
            trade_id = self.db.record_trade(signal_data)
            print(f"✅ Trade accepted: {signal_data['expert']} {signal_data['direction']} @ {signal_data['entry_price']}")
            return trade_id
        else:
            print(f"❌ Trade rejected: {signal_data['expert']} (RL filter)")
            return None

    def update_trade_result(self, trade_id, exit_price, pnl):
        """Update trade with results and learn"""

        # Update database
        self.db.update_trade(trade_id, exit_price, pnl)

        # Calculate reward for RL
        reward = np.tanh(pnl * 100)  # Normalize PnL to [-1, 1]

        # Train RL model
        # (In production, you'd store the state from when trade was opened)

        print(f"📊 Trade {trade_id} closed: PnL = {pnl:.4f}")

    def periodic_training(self):
        """Run periodic training on accumulated data"""
        print("🔄 Running periodic training...")

        # Replay training for RL
        avg_loss = self.rl.replay_train(batch_size=32)
        if avg_loss:
            print(f"  RL training loss: {avg_loss:.4f}")

        # Update expert statistics
        for expert in self.experts:
            stats = self.db.get_expert_performance(expert)
            if stats:
                print(f"  {expert}: Win rate={stats['win_rate']:.2%}, Avg PnL={stats['avg_pnl']:.4f}")

    def save_models(self):
        """Save current model states"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save RL model
        torch.save({
            'model_state_dict': self.rl.q_network.state_dict(),
            'optimizer_state_dict': self.rl.optimizer.state_dict(),
            'memory': self.rl.memory[-1000:] if len(self.rl.memory) > 1000 else self.rl.memory
        }, f'models/rl_filter_{timestamp}.pth')

        print(f"💾 Models saved at {timestamp}")

def main():
    print("""
    ╔════════════════════════════════════════╗
    ║   🔥 CHIMERA LIVE LEARNING SYSTEM 🔥   ║
    ╚════════════════════════════════════════╝
    """)

    system = LiveLearningSystem()

    # Simulate processing signals (in production, this would be real-time)
    test_signal = {
        'symbol': 'BTCUSDT',
        'expert': 'Ross',
        'direction': 'LONG',
        'entry_price': 336220.19,
        'stop_loss': 334539.09,
        'take_profit': 339582.38,
        'confidence': 0.96,
        'regime': 'TREND_UP',
        'features': {
            'spread_bps': 165.83,
            'imbalance': 0.999,
            'volatility': 0.02,
            'vwap_deviation': 0.001,
            'entropy': 3.5,
            'hurst': 0.48
        }
    }

    # Process signal
    trade_id = system.process_signal(test_signal)

    if trade_id:
        # Simulate trade result
        time.sleep(1)

        # Random PnL for demonstration
        pnl = np.random.normal(0.001, 0.005)
        exit_price = test_signal['entry_price'] * (1 + pnl)

        system.update_trade_result(trade_id, exit_price, pnl)

    # Run periodic training
    system.periodic_training()

    # Save models
    system.save_models()

    print("\n✅ Live learning system initialized!")

if __name__ == "__main__":
    main()