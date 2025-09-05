#!/usr/bin/env python3
"""
Fixed training script with proper dimension handling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RegimeClassifier(nn.Module):
    """Fixed regime classifier with proper dimensions"""
    def __init__(self, input_dim=14, hidden_dim=64, num_classes=5):
        super().__init__()
        # Single direction LSTM first to avoid dimension issues
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
        # x shape: (batch_size, sequence_length, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last hidden state
        # h_n shape: (num_layers, batch_size, hidden_dim)
        last_hidden = h_n[-1]  # Take last layer: (batch_size, hidden_dim)

        output = self.classifier(last_hidden)
        return output

class MultiAssetDataset(Dataset):
    """Fixed dataset with consistent dimensions"""
    def __init__(self, data_dir='data/raw', sequence_length=60, max_sequences_per_file=5000):
        self.sequence_length = sequence_length
        self.feature_dim = 14  # Fixed feature dimension
        self.data = []
        self.labels = []
        self.assets = []
        self.timeframes = []

        data_path = Path(data_dir)
        csv_files = sorted(list(data_path.glob('*_usdt_*.csv')))

        print(f"\n📊 Loading data from {len(csv_files)} files...")
        print("=" * 60)

        for csv_file in csv_files:
            parts = csv_file.stem.split('_')
            if len(parts) >= 3:
                asset = parts[0].upper()
                timeframe = parts[2]

                print(f"Loading {asset} {timeframe}...", end=' ')

                try:
                    df = pd.read_csv(csv_file)
                    if len(df) > sequence_length + 10:
                        sequences = self.process_dataframe(df, asset, timeframe, max_sequences_per_file)
                        print(f"✓ {sequences} sequences")
                    else:
                        print(f"✗ Too short ({len(df)} rows)")
                except Exception as e:
                    print(f"✗ Error: {e}")

        # Convert to numpy arrays for consistency
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

        print(f"\n✅ Total sequences loaded: {len(self.data)}")
        print(f"   Data shape: {self.data.shape}")
        print(f"   Assets: {len(set(self.assets))}")
        print(f"   Timeframes: {len(set(self.timeframes))}")

    def process_dataframe(self, df, asset, timeframe, max_sequences):
        """Process dataframe with fixed dimensions"""
        sequences_added = 0

        # Basic features
        df['returns'] = df['Close'].pct_change().clip(-0.1, 0.1)
        df['volume_norm'] = np.log1p(df['Volume'] / df['Volume'].mean())
        df['high_low'] = (df['High'] - df['Low']) / df['Close']
        df['close_open'] = (df['Close'] - df['Open']) / df['Open']

        # Moving averages
        for period in [5, 10, 20]:
            df[f'sma_{period}'] = df['Close'].rolling(period).mean() / df['Close']

        # Price position
        df['price_position'] = df['Close'] / df['Close'].rolling(50).mean()

        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()

        # Volume metrics
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

        # RSI simplified
        df['rsi'] = self.calculate_rsi(df['Close'], 14) / 100.0

        # Asset encoding (2 features)
        df['is_major'] = 1.0 if asset in ['BTC', 'ETH'] else 0.0
        df['is_1m'] = 1.0 if timeframe == '1m' else 0.0

        # Select exactly 14 features
        feature_cols = [
            'returns', 'volume_norm', 'high_low', 'close_open',
            'sma_5', 'sma_10', 'sma_20', 'price_position',
            'volatility', 'volume_ratio', 'rsi',
            'is_major', 'is_1m'
        ]

        # Add one more feature to make 14
        df['momentum'] = df['returns'].rolling(5).mean()
        feature_cols.append('momentum')

        # Clean data
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)

        # Ensure we have all columns
        for col in feature_cols[:self.feature_dim]:
            if col not in df.columns:
                df[col] = 0

        # Get feature matrix
        features = df[feature_cols[:self.feature_dim]].values.astype(np.float32)

        # Sample sequences evenly through the data
        step_size = max(1, (len(features) - self.sequence_length - 10) // max_sequences)

        for i in range(50, len(features) - self.sequence_length - 10, step_size):
            if sequences_added >= max_sequences:
                break

            seq = features[i:i+self.sequence_length]

            # Ensure correct shape
            if seq.shape == (self.sequence_length, self.feature_dim):
                # Calculate label
                future_return = df['returns'].iloc[i+self.sequence_length:i+self.sequence_length+5].mean()
                volatility = df['volatility'].iloc[i+self.sequence_length-1]

                if future_return > 0.001:
                    label = 0  # TREND_UP
                elif future_return < -0.001:
                    label = 1  # TREND_DOWN
                elif volatility < 0.01:
                    label = 2  # RANGE_BOUND
                elif volatility > 0.025:
                    label = 4  # CHAOTIC
                else:
                    label = 3  # LIQUIDITY_EVENT

                self.data.append(seq)
                self.labels.append(label)
                self.assets.append(asset)
                self.timeframes.append(timeframe)
                sequences_added += 1

        return sequences_added

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Ensure consistent tensor shapes
        data = torch.FloatTensor(self.data[idx])  # Shape: (sequence_length, feature_dim)
        label = torch.LongTensor([self.labels[idx]])  # Shape: (1,)
        return data, label

def train_model(model, train_loader, val_loader, epochs=10):
    """Train with validation"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\n🔥 Training model...")
    print("=" * 60)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_features, batch_labels in pbar:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.squeeze().to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_labels.size(0)
            train_correct += predicted.eq(batch_labels).sum().item()

            accuracy = 100. * train_correct / train_total
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy:.2f}%'})

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_features, batch_labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.squeeze().to(device)

                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_labels.size(0)
                val_correct += predicted.eq(batch_labels).sum().item()

        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%")

        scheduler.step(avg_val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc
            }, 'models/best_regime_classifier.pth')

    return model, history

def main():
    print("""
    ╔═══════════════════════════════════════════╗
    ║   🔥 CHIMERA TRAINING (FIXED) 🔥          ║
    ║   All Assets | Optimized Loading          ║
    ╚═══════════════════════════════════════════╝
    """)

    print(f"\n🎮 Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # Create dataset with limited sequences per file for memory efficiency
    dataset = MultiAssetDataset(
        sequence_length=60,
        max_sequences_per_file=150000  # Limit sequences per file
    )

    if len(dataset) == 0:
        print("❌ No data loaded.")
        return

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    print(f"\n📊 Dataset split:")
    print(f"   Training: {train_size} sequences")
    print(f"   Validation: {val_size} sequences")

    # Create dataloaders with appropriate batch size
    batch_size = 64  # Increased batch size since sequences are smaller
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize model
    model = RegimeClassifier(input_dim=14, hidden_dim=64, num_classes=5)
    print(f"\n🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Quick dimension check
    print("\n🔍 Checking dimensions...")
    sample_batch, sample_labels = next(iter(train_loader))
    print(f"   Batch shape: {sample_batch.shape}")
    print(f"   Labels shape: {sample_labels.shape}")

    # Train model
    trained_model, history = train_model(model, train_loader, val_loader, epochs=10)

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path('models').mkdir(exist_ok=True)

    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': {
            'input_dim': 14,
            'hidden_dim': 64,
            'num_classes': 5,
            'sequence_length': 60
        },
        'history': history,
        'timestamp': timestamp
    }, f'models/chimera_regime_{timestamp}.pth')

    print(f"\n✅ Training complete!")
    print(f"   Best validation accuracy: {max(history['val_acc']):.2f}%")

if __name__ == "__main__":
    main()