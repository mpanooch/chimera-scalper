#!/bin/bash

echo "🔥 CHIMERA ML Trading System"
echo "============================"
echo ""

# Function to show menu
show_menu() {
    echo "Select an option:"
    echo "1) Train new models on all assets"
    echo "2) Run live trading with ML (Python)"
    echo "3) Run C++ CHIMERA with Bybit feed"
    echo "4) Run complete system (C++ + Python ML)"
    echo "5) Check model performance"
    echo "6) View trading statistics"
    echo "0) Exit"
}

# Check for models
check_models() {
    echo "📂 Checking for trained models..."
    if [ -f "models/best_regime_classifier.pth" ]; then
        echo "✓ Best regime classifier found"
        ls -lh models/best_regime_classifier.pth
    else
        echo "❌ No trained model found. Please train first (option 1)"
        return 1
    fi

    # Show all available models
    echo ""
    echo "Available models:"
    ls -lh models/*.pth 2>/dev/null || echo "No models found"
    echo ""
    return 0
}

# Train models
train_models() {
    echo "🧠 Training ML models on all assets..."
    python train_all_assets_fixed.py

    if [ $? -eq 0 ]; then
        echo "✅ Training complete!"
    else
        echo "❌ Training failed. Check error messages above."
    fi
}

# Run live ML trading
run_ml_trading() {
    if ! check_models; then
        echo "Please train models first!"
        return
    fi

    echo "🚀 Starting ML-powered live trading..."
    python live_trading_with_ml.py
}

# Run C++ CHIMERA
run_cpp_chimera() {
    echo "🚀 Starting C++ CHIMERA with Bybit feed..."

    # Start Bybit feeder in background
    python enhanced_bybit_multi.py &
    FEEDER_PID=$!

    sleep 3

    # Start CHIMERA
    ./build/bin/chimera_scalper &
    CHIMERA_PID=$!

    echo ""
    echo "Systems running:"
    echo "  Bybit Feeder PID: $FEEDER_PID"
    echo "  CHIMERA PID: $CHIMERA_PID"
    echo ""
    echo "Press Ctrl+C to stop"

    # Wait for interrupt
    trap "kill $FEEDER_PID $CHIMERA_PID 2>/dev/null; exit" INT
    wait
}

# Run complete system
run_complete() {
    if ! check_models; then
        echo "Please train models first!"
        return
    fi

    echo "🔥 Starting complete CHIMERA system..."
    echo "  • C++ microstructure engine"
    echo "  • Python ML predictions"
    echo "  • Bybit real-time data"
    echo ""

    # Start all components
    python enhanced_bybit_multi.py &
    FEEDER_PID=$!

    sleep 2

    python live_trading_with_ml.py &
    ML_PID=$!

    sleep 2

    ./build/bin/chimera_scalper &
    CHIMERA_PID=$!

    echo ""
    echo "All systems running:"
    echo "  Bybit Feeder: $FEEDER_PID"
    echo "  ML Trading: $ML_PID"
    echo "  CHIMERA Core: $CHIMERA_PID"
    echo ""
    echo "Press Ctrl+C to stop all"

    # Cleanup function
    cleanup() {
        echo -e "\n🛑 Shutting down all systems..."
        kill $FEEDER_PID $ML_PID $CHIMERA_PID 2>/dev/null
        echo "✓ All systems stopped"
        exit 0
    }

    trap cleanup INT
    wait
}

# Check performance
check_performance() {
    echo "📊 Model Performance Analysis"
    echo "============================="

    python << 'EOF'
import torch
from pathlib import Path

model_files = sorted(Path('models').glob('*.pth'))

for model_file in model_files:
    print(f"\n{model_file.name}:")
    try:
        checkpoint = torch.load(model_file, map_location='cpu')

        if 'val_acc' in checkpoint:
            print(f"  Validation Accuracy: {checkpoint['val_acc']:.2f}%")
        if 'epoch' in checkpoint:
            print(f"  Training Epoch: {checkpoint['epoch']}")
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"  Model Config: {config}")
        if 'history' in checkpoint:
            history = checkpoint['history']
            if 'val_acc' in history:
                print(f"  Best Val Acc: {max(history['val_acc']):.2f}%")
    except Exception as e:
        print(f"  Error loading: {e}")
EOF
}

# View statistics
view_statistics() {
    echo "📈 Trading Statistics"
    echo "===================="

    if [ -f "chimera_trades.db" ]; then
        python << 'EOF'
import sqlite3
import pandas as pd

conn = sqlite3.connect('chimera_trades.db')

# Get recent trades
trades_df = pd.read_sql_query("""
    SELECT * FROM trades
    ORDER BY timestamp DESC
    LIMIT 10
""", conn)

if not trades_df.empty:
    print("\nRecent Trades:")
    print(trades_df[['symbol', 'expert', 'direction', 'entry_price', 'pnl', 'status']])

# Get performance by expert
perf_df = pd.read_sql_query("""
    SELECT expert,
           COUNT(*) as total_trades,
           SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
           AVG(pnl) as avg_pnl
    FROM trades
    WHERE status = 'CLOSED'
    GROUP BY expert
""", conn)

if not perf_df.empty:
    print("\nExpert Performance:")
    print(perf_df)

conn.close()
EOF
    else
        echo "No trading database found. Run live trading first."
    fi
}

# Main loop
while true; do
    show_menu
    read -p "Enter choice: " choice

    case $choice in
        1)
            train_models
            ;;
        2)
            run_ml_trading
            ;;
        3)
            run_cpp_chimera
            ;;
        4)
            run_complete
            ;;
        5)
            check_performance
            ;;
        6)
            view_statistics
            ;;
        0)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid option"
            ;;
    esac

    echo ""
    echo "Press Enter to continue..."
    read
    clear
done
