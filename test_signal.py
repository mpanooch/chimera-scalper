#!/usr/bin/env python3
"""
Test the simple signal monitoring system
"""

import requests
import json
from datetime import datetime

def test_signal_system():
    """Test sending and receiving signals"""
    
    print("🔄 Testing CHIMERA Signal Monitoring System")
    print("=" * 50)
    
    # Test data - just current status
    signal_data = {
        'timestamp': datetime.now().isoformat(),
        'status': 'active',
        'performance': {
            'total_pnl': 125.50,
            'pnl_percent': 1.25,
            'win_rate': 65.5,
            'total_trades': 42,
            'balance': 10125.50
        },
        'regime': {
            'current': 'TRENDING',
            'confidence': 0.85
        },
        'system': {
            'model_accuracy': 72.3,
            'signals_generated': 8
        },
        'latest_trade': {
            'symbol': 'BTCUSDT',
            'action': 'BUY',
            'price': 111500.00,
            'pnl_pct': 2.1,
            'time': datetime.now().strftime('%H:%M:%S')
        }
    }
    
    base_url = "https://tradingtoday.com.au"
    
    # Test 1: Send signal
    print("\n1. Sending signal to website...")
    try:
        response = requests.post(
            f"{base_url}/api/chimera/signal",
            json=signal_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   ✅ Signal sent successfully")
        else:
            print(f"   ❌ Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Check if website can read the signal
    print("\n2. Checking if website received signal...")
    try:
        response = requests.get(f"{base_url}/api/chimera/signal", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Signal received by website")
            print(f"   📊 PnL: {data.get('performance', {}).get('pnl_percent', 0)}%")
            print(f"   🎯 Regime: {data.get('regime', {}).get('current', 'UNKNOWN')}")
            print(f"   🤖 Model Accuracy: {data.get('system', {}).get('model_accuracy', 0)}%")
        else:
            print(f"   ❌ Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n✅ Signal monitoring test completed!")
    print("\nTo integrate with your trading system:")
    print("1. Import send_signal.py in your trading code")
    print("2. Call send_signal_update() when you want to update the website")
    print("3. All detailed data stays on your PC for training")

if __name__ == "__main__":
    test_signal_system()