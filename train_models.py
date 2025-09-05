"""
CHIMERA Scalper - ML Model Training Pipeline
Trains all neural network models for Layers 2-4
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'data_dir': './data/raw/',
    'model_dir': './models/',
    'sequence_length': 60,
    'hidden_dim': 256,
    'num_heads': 8,
    'num_layers': 4,
    'dropout': 0.1,
    'learning_rate': 0.0001,
    'batch_size': 256,
    'epochs': 100,
    'validation_split': 0.2,
    'test_split': 0.1,
    'pairs': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
}

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

class MarketDataset(Dataset):
    """Custom dataset for market data with microstructure features"""

    def __init__(self, data, sequence_length=60, transform=None):
        self.data = data
        self.sequence_length = sequence_length
        self.transform = transform

    def __len__(self):
        return len(self.data) - self.sequence_length - 1

    def __getitem__(self, idx):
        # Get sequence of features
        sequence = self.data[idx:idx + self.sequence_length]
        target = self.data[idx + self.sequence_length]

        if self.transform:
            sequence = self.transform(sequence)

        return torch.FloatTensor(sequence), torch.FloatTensor(target)

def load_market_data(pair, timeframe='1m'):
    """Load and preprocess market data with feature engineering"""

    filepath = os.path.join(CONFIG['data_dir'], f"{pair.lower()}_{timeframe}.csv")
    df = pd.read_csv(filepath)

    # Calculate technical indicators
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['volume_delta'] = df['Volume'].diff()

    # Price features
    df['hl_spread'] = (df['High'] - df['Low']) / df['Close']
    df['co_spread'] = (df['Close'] - df['Open']) / df['Open']

    # Volume features
    df['volume_ma'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma']

    # Volatility
    df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)

    # VWAP
    df['vwap'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['vwap_deviation'] = (df['Close'] - df['vwap']) / df['vwap']

    # EMAs
    df['ema_9'] = df['Close'].ewm(span=9).mean()
    df['ema_21'] = df['Close'].ewm(span=21).mean()
    df['ema_cross'] = (df['ema_9'] - df['ema_21']) / df['Close']

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(14).mean()

    # Order flow imbalance proxy (using volume and price action)
    df['buy_pressure'] = df.apply(
        lambda x: x['Volume'] if x['Close'] > x['Open'] else 0, axis=1
    )
    df['sell_pressure'] = df.apply(
        lambda x: x['Volume'] if x['Close'] < x['Open'] else 0, axis=1
    )
    df['order_imbalance'] = (df['buy_pressure'] - df['sell_pressure']) / df['Volume']

    # Market microstructure features (synthetic)
    df['spread'] = np.random.uniform(0.0001, 0.001, len(df))  # Synthetic spread
    df['liquidity_score'] = 1 / (1 + df['spread'] * df['volatility'])
    df['entropy'] = -df['volume_ratio'] * np.log(df['volume_ratio'].clip(0.001, None))

    # Labels for supervised learning
    df['future_return'] = df['Close'].shift(-1) / df['Close'] - 1
    df['direction'] = np.where(df['future_return'] > 0.001, 1,
                               np.where(df['future_return'] < -0.001, -1, 0))

    # Market regime (simplified)
    df['trend'] = np.where(df['ema_9'] > df['ema_21'], 1, -1)
    df['regime'] = pd.cut(df['volatility'], bins=3, labels=['low_vol', 'med_vol', 'high_vol'])

    # Drop NaN values
    df = df.dropna()

    return df

def create_feature_matrix(df):
    """Create feature matrix for model training"""

    feature_cols = [
        'returns', 'log_returns', 'volume_ratio', 'hl_spread', 'co_spread',
        'volatility', 'vwap_deviation', 'ema_cross', 'rsi', 'atr',
        'order_imbalance', 'spread', 'liquidity_score', 'entropy'
    ]

    X = df[feature_cols].values
    y = df[['direction', 'future_return']].values

    # Normalize features
    scaler = RobustScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler

# ============================================================================
# LAYER 2: MARKET REGIME CLASSIFIER (TRANSFORMER)
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class RegimeClassifier(nn.Module):
    """Transformer-based market regime classifier"""

    def __init__(self, input_dim=14, hidden_dim=256, num_heads=8,
                 num_layers=4, num_classes=5, dropout=0.1):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, sequence, features)
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)

        # Use last hidden state for classification
        x = x[:, -1, :]
        return self.classifier(x)

# ============================================================================
# LAYER 3: EXPERT MODELS (MoE)
# ============================================================================

class ExpertModel(nn.Module):
    """Individual expert model for specific trading strategy"""

    def __init__(self, input_dim, hidden_dim=128, expert_type='ross'):
        super().__init__()
        self.expert_type = expert_type

        # Shared backbone
        self.backbone = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=2, batch_first=True,
            dropout=0.1, bidirectional=True
        )

        # Expert-specific heads
        if expert_type == 'ross':  # Momentum
            self.head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 4)  # direction, entry, stop, confidence
            )
        elif expert_type == 'bao':  # Mean reversion
            self.head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 4)
            )
        elif expert_type == 'nick':  # Volume/pressure
            self.head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, 64),
                nn.ELU(),
                nn.Linear(64, 4)
            )
        elif expert_type == 'fabio':  # Chaos/entropy
            self.head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, 4)
            )

    def forward(self, x):
        lstm_out, _ = self.backbone(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.head(last_hidden)

        # Split output into components
        direction = torch.tanh(output[:, 0:1])  # -1 to 1
        entry = output[:, 1:2]  # Raw value
        stop = torch.sigmoid(output[:, 2:3]) * 0.02  # 0-2% stop
        confidence = torch.sigmoid(output[:, 3:4])  # 0-1 confidence

        return torch.cat([direction, entry, stop, confidence], dim=1)

class MixtureOfExperts(nn.Module):
    """Gating network for expert selection"""

    def __init__(self, input_dim, hidden_dim=256, num_experts=4):
        super().__init__()

        # Individual experts
        self.experts = nn.ModuleList([
            ExpertModel(input_dim, hidden_dim, 'ross'),
            ExpertModel(input_dim, hidden_dim, 'bao'),
            ExpertModel(input_dim, hidden_dim, 'nick'),
            ExpertModel(input_dim, hidden_dim, 'fabio')
        ])

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Get expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        # Get gating weights
        gate_weights = self.gate(x[:, -1, :])  # Use last timestep
        gate_weights = gate_weights.unsqueeze(2)

        # Weighted combination
        output = torch.sum(expert_outputs * gate_weights, dim=1)

        return output, gate_weights.squeeze()

# ============================================================================
# LAYER 4: PNL FILTER (RL-BASED)
# ============================================================================

class PnLFilter(nn.Module):
    """Reinforcement learning based profitability filter"""

    def __init__(self, state_dim=32, action_dim=2):
        super().__init__()

        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=1)
        )

        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value

    def act(self, state):
        action_probs, _ = self.forward(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action, dist.log_prob(action)

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_regime_classifier(X_train, y_train, X_val, y_val):
    """Train the market regime classifier"""

    print("\n" + "="*60)
    print("Training Market Regime Classifier (Layer 2)")
    print("="*60)

    model = RegimeClassifier(
        input_dim=X_train.shape[2],
        hidden_dim=CONFIG['hidden_dim'],
        num_heads=CONFIG['num_heads'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.CrossEntropyLoss()

    train_dataset = MarketDataset(X_train, CONFIG['sequence_length'])
    val_dataset = MarketDataset(X_val, CONFIG['sequence_length'])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])

    best_val_loss = float('inf')

    for epoch in range(CONFIG['epochs']):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y[:, 0].long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y[:, 0].long())
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y[:, 0].long()).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(CONFIG['model_dir'], 'regime_classifier.pt'))

        scheduler.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{CONFIG['epochs']} - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Accuracy: {accuracy:.2f}%")

    print(f"Best validation loss: {best_val_loss:.4f}")
    return model

def train_expert_models(X_train, y_train, X_val, y_val):
    """Train the mixture of experts"""

    print("\n" + "="*60)
    print("Training Expert Models (Layer 3)")
    print("="*60)

    model = MixtureOfExperts(
        input_dim=X_train.shape[2],
        hidden_dim=CONFIG['hidden_dim'] // 2,
        num_experts=4
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'])

    # Custom loss for trading signals
    def trading_loss(outputs, targets):
        direction_loss = F.mse_loss(outputs[:, 0], targets[:, 0])
        return_loss = F.mse_loss(outputs[:, 1], targets[:, 1])
        confidence_penalty = -torch.mean(outputs[:, 3])  # Encourage confidence

        return direction_loss + 0.5 * return_loss + 0.01 * confidence_penalty

    train_dataset = MarketDataset(X_train, CONFIG['sequence_length'])
    val_dataset = MarketDataset(X_val, CONFIG['sequence_length'])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])

    best_val_loss = float('inf')

    for epoch in range(CONFIG['epochs']):
        # Training
        model.train()
        train_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs, gate_weights = model(batch_x)
            loss = trading_loss(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs, gate_weights = model(batch_x)
                loss = trading_loss(outputs, batch_y)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save individual expert models
            for i, expert_name in enumerate(['ross', 'bao', 'nick', 'fabio']):
                torch.save(model.experts[i].state_dict(),
                           os.path.join(CONFIG['model_dir'], f'{expert_name}_expert.pt'))

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{CONFIG['epochs']} - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}")

    print(f"Best validation loss: {best_val_loss:.4f}")
    return model

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""

    print("\n" + "="*60)
    print("CHIMERA Scalper - Model Training Pipeline")
    print("="*60)

    # Create directories
    os.makedirs(CONFIG['model_dir'], exist_ok=True)

    # Load and combine data from all pairs
    all_data = []

    for pair in CONFIG['pairs']:
        print(f"\nLoading data for {pair}...")
        try:
            df_1m = load_market_data(pair, '1m')
            df_5m = load_market_data(pair, '5m')

            # Create features
            X_1m, y_1m, scaler_1m = create_feature_matrix(df_1m)
            X_5m, y_5m, scaler_5m = create_feature_matrix(df_5m)

            # Combine timeframes
            all_data.append((X_1m, y_1m))
            all_data.append((X_5m, y_5m))

            print(f"  1m data shape: {X_1m.shape}")
            print(f"  5m data shape: {X_5m.shape}")

        except Exception as e:
            print(f"  Error loading {pair}: {e}")
            continue

    # Combine all data
    X_combined = np.vstack([x[0] for x in all_data])
    y_combined = np.vstack([x[1] for x in all_data])

    print(f"\nCombined data shape: {X_combined.shape}")

    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_combined, y_combined, test_size=CONFIG['test_split'], random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=CONFIG['validation_split'], random_state=42
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Train models
    regime_model = train_regime_classifier(X_train, y_train, X_val, y_val)
    expert_model = train_expert_models(X_train, y_train, X_val, y_val)

    # Save configuration
    with open(os.path.join(CONFIG['model_dir'], 'config.json'), 'w') as f:
        json.dump(CONFIG, f, indent=2)

    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Models saved to: {CONFIG['model_dir']}")
    print("="*60)

if __name__ == "__main__":
    main()#!/usr/bin/env python3