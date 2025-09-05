#!/usr/bin/env python3
"""
Test script to verify the web API is working
"""

import requests
import json
import time

# Test data
test_data = {
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
    'market': {
        'BTCUSDT': 111580.70,
        'ETHUSDT': 4336.45,
        'SOLUSDT': 204.51
    },
    'system': {
        'processing_speed': 15.2,
        'total_ticks': 1250,
        'signals_generated': 8,
        'model_accuracy': 72.3
    },
    'data_collection': {
        'market_data_points': 1250,
        'trade_outcomes': 42,
        'db_size_mb': 2.5,
        'learning_rate': 0.001
    },
    'recent_trades': [
        {
            'timestamp': '2024-01-15T10:30:00Z',
            'symbol': 'BTCUSDT',
            'action': 'BUY',
            'price': 111500.00,
            'regime': 'TRENDING',
            'confidence': 0.85,
            'pnl_pct': 2.1
        },
        {
            'timestamp': '2024-01-15T10:25:00Z',
            'symbol': 'ETHUSDT',
            'action': 'SELL',
            'price': 4340.00,
            'regime': 'TRENDING',
            'confidence': 0.78,
            'pnl_pct': -0.5
        }
    ]
}

def test_local_api():
    """Test the API locally using PHP's built-in server"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing CHIMERA Web API")
    print("=" * 50)
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/api/chimera/health")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test update endpoint
    print("\n2. Testing update endpoint...")
    try:
        response = requests.post(
            f"{base_url}/api/chimera/update",
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test stats endpoint
    print("\n3. Testing stats endpoint...")
    try:
        response = requests.get(f"{base_url}/api/chimera/stats")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Data received: {len(json.dumps(data))} bytes")
            print(f"   Performance: PnL {data.get('performance', {}).get('pnl_percent', 0)}%")
            print(f"   Market: BTC ${data.get('market', {}).get('BTCUSDT', 0)}")
            print(f"   Regime: {data.get('regime', {}).get('current', 'UNKNOWN')}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n✅ API test completed!")
    print("\nTo start PHP server: php -S localhost:8000")
    print("Then open: http://localhost:8000/index.html")

def test_production_api():
    """Test the production API"""
    base_url = "https://tradingtoday.com.au"

    print("🌐 Testing Production CHIMERA API")
    print("=" * 50)

    # Test update endpoint
    print("\n1. Testing update endpoint (POST)...")
    try:
        response = requests.post(
            f"{base_url}/api/chimera/index.php?action=update",
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test stats endpoint
    print("\n2. Testing stats endpoint (GET)...")
    try:
        response = requests.get(f"{base_url}/api/chimera/index.php?action=stats", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Data received: {len(json.dumps(data))} bytes")
            print(f"   Last updated: {data.get('last_updated', 'N/A')}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "prod":
        test_production_api()
    else:
        test_local_api()