#!/usr/bin/env python3
"""
Retrain regime classifier with demo trading data
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path

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

def retrain_model():
    print("🔄 Starting model retraining with demo data...")

    # Load demo data
    market_data = pd.read_csv('demo_market_data.csv')
    trades_log = pd.read_csv('demo_trades_log.csv')

    print(f"📊 Loaded {len(market_data)} market samples")
    print(f"📊 Loaded {len(trades_log)} trade outcomes")

    # Process features and create labels from successful trades
    sequences = []
    labels = []

    # Group by symbol and create sequences
    for symbol in market_data['symbol'].unique():
        symbol_data = market_data[market_data['symbol'] == symbol].sort_values('timestamp')

        for i in range(60, len(symbol_data)):
            # Get 60-step sequence
            sequence_data = symbol_data.iloc[i-60:i]
            features_list = [json.loads(f) for f in sequence_data['features']]
            sequences.append(features_list)

            # Create label from successful trades
            # This is simplified - in practice you'd want more sophisticated labeling
            current_time = symbol_data.iloc[i]['timestamp']
            recent_trades = trades_log[
                (trades_log['symbol'] == symbol) &
                (trades_log['timestamp'] >= current_time)
                ].head(1)

            if len(recent_trades) > 0 and recent_trades.iloc[0]['success'] == 1:
                if recent_trades.iloc[0]['action'] == 'BUY':
                    labels.append(0)  # TREND_UP
                else:
                    labels.append(1)  # TREND_DOWN
            else:
                labels.append(2)  # RANGE_BOUND

    if len(sequences) == 0:
        print("❌ No training sequences generated")
        return

    # Convert to tensors
    X = torch.FloatTensor(sequences)
    y = torch.LongTensor(labels)

    print(f"✅ Created {len(sequences)} training sequences")

    # Load existing model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RegimeClassifier()

    # Try to load existing weights
    model_path = 'models/best_regime_classifier.pth'
    if Path(model_path).exists():
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Loaded existing model weights")

    model.to(device)

    # Prepare training
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Fine-tune on demo data
    model.train()
    print("🚀 Fine-tuning model on demo data...")

    for epoch in range(10):  # Few epochs for fine-tuning
        total_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

        accuracy = 100. * correct / total
        print(f"Epoch {epoch+1}/10: Loss={total_loss/len(train_loader):.4f}, Acc={accuracy:.2f}%")

    # Save retrained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_samples': len(sequences),
        'demo_accuracy': accuracy,
        'retrain_date': pd.Timestamp.now().isoformat()
    }, 'models/retrained_regime_classifier.pth')

    print("✅ Model retrained and saved!")
    print(f"📈 Final accuracy: {accuracy:.2f}%")
    print("🚀 Ready for live trading with improved model!")

if __name__ == "__main__":
    retrain_model()