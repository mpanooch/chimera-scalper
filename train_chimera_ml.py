#!/usr/bin/env python3
"""
Train CHIMERA's ML models using historical data and live trading results
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import pickle

# ============= Layer 2: Market Regime Classifier =============
class RegimeClassifier(nn.Module):
    """Transformer-based market regime classifier"""
    def __init__(self, input_dim=14, hidden_dim=128, num_heads=4, num_classes=5):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, sequence, features)
        x = self.input_projection(x)
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]

        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)

# ============= Layer 3: Expert Models =============
class RossExpert(nn.Module):
    """Momentum breakout expert"""
    def __init__(self, input_dim=14):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.attention = nn.MultiheadAttention(64, num_heads=4, batch_first=True)

        self.signal_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # [direction, entry_offset, stop_offset, confidence]
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        final_state = attn_out[:, -1, :]
        return self.signal_head(final_state)

class BaoExpert(nn.Module):
    """Mean reversion expert"""
    def __init__(self, input_dim=14):
        super().__init__()
        self.gru = nn.GRU(input_dim, 64, num_layers=2, batch_first=True, dropout=0.2)

        self.vwap_processor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        self.signal_head = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        final_state = gru_out[:, -1, :]
        vwap_features = self.vwap_processor(final_state)
        return self.signal_head(vwap_features)

class MixtureOfExperts(nn.Module):
    """Gating network for expert selection"""
    def __init__(self, input_dim=14, num_experts=4):
        super().__init__()
        self.num_experts = num_experts

        self.gating = nn.Sequential(
            nn.Linear(input_dim + 5, 32),  # +5 for regime one-hot
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_experts),
            nn.Softmax(dim=-1)
        )

        self.ross = RossExpert(input_dim)
        self.bao = BaoExpert(input_dim)
        # Add Nick and Fabio experts later

    def forward(self, features, regime):
        # Concatenate features with regime
        batch_size = features.size(0)
        last_features = features[:, -1, :]  # Use last timestep

        # One-hot encode regime
        regime_onehot = torch.zeros(batch_size, 5)
        regime_onehot.scatter_(1, regime.unsqueeze(1), 1)

        gating_input = torch.cat([last_features, regime_onehot], dim=-1)
        weights = self.gating(gating_input)

        # Get expert outputs
        ross_out = self.ross(features)
        bao_out = self.bao(features)

        # Weighted combination
        combined = weights[:, 0:1] * ross_out + weights[:, 1:2] * bao_out

        return combined, weights

# ============= Layer 4: PnL Filter (RL) =============
class PnLFilter(nn.Module):
    """Reinforcement learning based PnL filter"""
    def __init__(self, state_dim=20, action_dim=2):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.actor(state), self.critic(state)

# ============= Dataset =============
class TradingDataset(Dataset):
    def __init__(self, csv_files, sequence_length=60):
        self.sequence_length = sequence_length
        self.data = []
        self.labels = []

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            self.process_dataframe(df)

    def process_dataframe(self, df):
        # Calculate features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['volume_delta'] = df['Volume'].pct_change()
        df['spread'] = (df['High'] - df['Low']) / df['Close']
        df['vwap'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        df['vwap_deviation'] = (df['Close'] - df['vwap']) / df['vwap']

        # Technical indicators
        df['ema_9'] = df['Close'].ewm(span=9).mean()
        df['ema_21'] = df['Close'].ewm(span=21).mean()
        df['rsi'] = self.calculate_rsi(df['Close'])

        # Drop NaN values
        df = df.dropna()

        # Create sequences
        features = df[['Open', 'High', 'Low', 'Close', 'Volume',
                       'returns', 'log_returns', 'volume_delta', 'spread',
                       'vwap_deviation', 'ema_9', 'ema_21', 'rsi']].values

        for i in range(len(features) - self.sequence_length - 1):
            self.data.append(features[i:i+self.sequence_length])

            # Label: next period return direction (0=down, 1=up)
            next_return = features[i+self.sequence_length, 5]  # returns column
            self.labels.append(1 if next_return > 0 else 0)

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]])

# ============= Training Functions =============
def train_regime_classifier(model, dataloader, epochs=50):
    """Train the regime classification model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    print("Training Regime Classifier...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.squeeze().to(device)

            optimizer.zero_grad()

            # Synthetic regime labels (in production, use actual regime labels)
            # For now, use simple rule-based labeling
            regime_labels = generate_regime_labels(batch_features)
            regime_labels = regime_labels.to(device)

            outputs = model(batch_features)
            loss = criterion(outputs, regime_labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += regime_labels.size(0)
            correct += predicted.eq(regime_labels).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total

        scheduler.step(avg_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")

    return model

def generate_regime_labels(features):
    """Generate synthetic regime labels based on features"""
    # Simplified regime detection
    # 0: TREND_UP, 1: TREND_DOWN, 2: RANGE_BOUND, 3: LIQUIDITY_EVENT, 4: CHAOTIC_SPIKE

    batch_size = features.size(0)
    labels = torch.zeros(batch_size, dtype=torch.long)

    for i in range(batch_size):
        returns = features[i, :, 5].mean()  # Average returns
        volatility = features[i, :, 5].std()  # Volatility

        if returns > 0.001 and volatility < 0.02:
            labels[i] = 0  # TREND_UP
        elif returns < -0.001 and volatility < 0.02:
            labels[i] = 1  # TREND_DOWN
        elif abs(returns) < 0.0005 and volatility < 0.01:
            labels[i] = 2  # RANGE_BOUND
        elif volatility > 0.03:
            labels[i] = 4  # CHAOTIC_SPIKE
        else:
            labels[i] = 2  # Default to RANGE_BOUND

    return labels

def save_models(models_dict, directory="models"):
    """Save trained models"""
    Path(directory).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for name, model in models_dict.items():
        filepath = f"{directory}/chimera_{name}_{timestamp}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model.__class__.__name__,
            'timestamp': timestamp
        }, filepath)
        print(f"Saved {name} to {filepath}")

def main():
    print("""
    ╔════════════════════════════════════════╗
    ║   🔥 CHIMERA ML TRAINING SYSTEM 🔥     ║
    ╚════════════════════════════════════════╝
    """)

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print("\nLoading training data...")
    csv_files = list(Path('data/raw').glob('*_1m.csv'))[:9]  # Start with 9 symbols

    if not csv_files:
        print("No CSV files found in data/raw/")
        return

    print(f"Found {len(csv_files)} files")

    # Create dataset
    dataset = TradingDataset(csv_files, sequence_length=60)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"Dataset size: {len(dataset)} sequences")

    # Initialize models
    print("\nInitializing models...")
    regime_model = RegimeClassifier(input_dim=13)  # Adjusted for feature count
    moe_model = MixtureOfExperts(input_dim=13)
    pnl_filter = PnLFilter()

    # Train models
    print("\n" + "="*50)
    trained_regime = train_regime_classifier(regime_model, dataloader, epochs=20)

    # Save models
    print("\nSaving models...")
    save_models({
        'regime_classifier': trained_regime,
        'mixture_of_experts': moe_model,
        'pnl_filter': pnl_filter
    })

    print("\n✅ Training complete!")

if __name__ == "__main__":
    main()