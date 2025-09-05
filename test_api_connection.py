#!/usr/bin/env python3
"""
Test script to verify CHIMERA API connection to tradingtoday.com.au
"""

import requests
import json
from datetime import datetime

def test_api_connection():
    api_url = "https://chimera-scalper.herokuapp.com"
    
    print("🧪 Testing CHIMERA API Connection")
    print("=" * 50)
    
    # Test 1: Health Check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{api_url}?action=health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check successful!")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    print()
    
    # Test 2: Send sample data
    print("2. Testing data update endpoint...")
    sample_data = {
        'type': 'performance_update',
        'performance': {
            'total_pnl': 150.75,
            'pnl_percent': 1.51,
            'win_rate': 65.2,
            'total_trades': 23,
            'balance': 10150.75
        },
        'regime': {
            'current': 'BULLISH',
            'confidence': 0.85
        },
        'market': {
            'BTCUSDT': 43250.50,
            'ETHUSDT': 2650.25,
            'SOLUSDT': 95.75
        },
        'system': {
            'processing_speed': 1250,
            'total_ticks': 15420,
            'signals_generated': 45,
            'model_accuracy': 0.78
        },
        'recent_trades': [
            {
                'symbol': 'BTCUSDT',
                'direction': 'LONG',
                'entry_price': 43200.00,
                'exit_price': 43350.00,
                'pnl': 15.75,
                'timestamp': datetime.now().isoformat()
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{api_url}?action=update",
            json=sample_data,
            timeout=10,
            headers={'Content-Type': 'application/json'}
        )
        if response.status_code == 200:
            print("✅ Data update successful!")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Data update failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Data update error: {e}")
    
    print()
    
    # Test 3: Retrieve stats
    print("3. Testing stats retrieval...")
    try:
        response = requests.get(f"{api_url}?action=stats", timeout=10)
        if response.status_code == 200:
            print("✅ Stats retrieval successful!")
            data = response.json()
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Performance: {data.get('performance', {})}")
            print(f"   Regime: {data.get('regime', {})}")
        else:
            print(f"❌ Stats retrieval failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Stats retrieval error: {e}")
    
    print()
    print("🏁 API Connection Test Complete!")

if __name__ == "__main__":
    test_api_connection()