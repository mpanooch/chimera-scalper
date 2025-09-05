#!/usr/bin/env python3
"""
Simple script to send CHIMERA trading signals to the monitoring website
Keeps all data locally, just sends status updates for monitoring
"""

import requests
import json
import time
from datetime import datetime

def send_signal_update(performance_data, current_regime, latest_trade=None):
    """
    Send a simple status update to the monitoring website
    
    Args:
        performance_data: dict with pnl, win_rate, total_trades, balance
        current_regime: dict with regime and confidence
        latest_trade: dict with latest trade info (optional)
    """
    
    # Simple monitoring data - just current status
    monitoring_data = {
        'timestamp': datetime.now().isoformat(),
        'status': 'active',
        'performance': performance_data,
        'regime': current_regime,
        'system': {
            'model_accuracy': performance_data.get('model_accuracy', 0),
            'signals_generated': performance_data.get('signals_today', 0)
        }
    }
    
    # Add latest trade if provided
    if latest_trade:
        monitoring_data['latest_trade'] = latest_trade
    
    try:
        # Send to your monitoring website
        response = requests.post(
            "https://tradingtoday.com.au/api/chimera/update",
            json=monitoring_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"✅ Signal sent successfully at {datetime.now().strftime('%H:%M:%S')}")
            return True
        else:
            print(f"❌ Failed to send signal: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error sending signal: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # Example: Send current trading status
    current_performance = {
        'total_pnl': 125.50,
        'pnl_percent': 1.25,
        'win_rate': 65.5,
        'total_trades': 42,
        'balance': 10125.50,
        'model_accuracy': 72.3,
        'signals_today': 8
    }
    
    current_regime = {
        'current': 'TRENDING',
        'confidence': 0.85
    }
    
    latest_trade = {
        'symbol': 'BTCUSDT',
        'action': 'BUY',
        'price': 111500.00,
        'pnl_pct': 2.1,
        'time': datetime.now().strftime('%H:%M:%S')
    }
    
    # Send the signal
    send_signal_update(current_performance, current_regime, latest_trade)