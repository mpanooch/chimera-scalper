#!/usr/bin/env python3
"""
Test script for Socket.IO Dashboard Bridge
"""

import time
import random
from datetime import datetime, timedelta
from socketio_dashboard import SocketIODashboardBridge


class MockTradingSystem:
    def __init__(self):
        self.stats = {
            'ticks': 0,
            'signals': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_pnl': 0.0,
            'last_prices': {
                'BTCUSDT': 45000.0,
                'ETHUSDT': 2500.0,
                'SOLUSDT': 100.0,
                'ADAUSDT': 0.5,
                'DOGEUSDT': 0.08
            }
        }
        self.training_start = datetime.now()
        self.training_duration = timedelta(days=14)
        self.start_time = time.time()
        self.current_confidence = 0.75
        self.training_samples_collected = 1500
        
    def simulate_tick(self):
        """Simulate a market tick"""
        self.stats['ticks'] += 1
        
        # Randomly update prices
        for symbol in self.stats['last_prices']:
            change = random.uniform(-0.02, 0.02)  # ±2% change
            self.stats['last_prices'][symbol] *= (1 + change)
            
        # Occasionally generate a signal
        if random.random() < 0.1:  # 10% chance
            self.stats['signals'] += 1
            if random.random() < 0.6:  # 60% success rate
                self.stats['successful_trades'] += 1
                pnl = random.uniform(0.1, 2.0)
            else:
                self.stats['failed_trades'] += 1
                pnl = random.uniform(-1.0, -0.1)
            
            self.stats['total_pnl'] += pnl
            
            return {
                'symbol': random.choice(['BTCUSDT', 'ETHUSDT', 'SOLUSDT']),
                'action': random.choice(['BUY', 'SELL']),
                'price': random.choice(list(self.stats['last_prices'].values())),
                'regime': random.choice(['TREND_UP', 'TREND_DOWN', 'RANGE', 'LIQUIDITY', 'CHAOS']),
                'confidence': random.uniform(0.5, 0.95),
                'expert': random.choice(['ross', 'bao', 'nick', 'fabio']),
                'pnl_pct': pnl,
                'cumulative_pnl': self.stats['total_pnl']
            }
        return None


def main():
    print("🔥 Starting CHIMERA Dashboard Test")
    
    # Create mock trading system
    trading_system = MockTradingSystem()
    
    # Create dashboard bridge
    dashboard = SocketIODashboardBridge(trading_system)
    
    # Start dashboard connection
    dashboard.start()
    
    print("📊 Dashboard bridge started")
    print("🌐 Dashboard should be available at: http://localhost:3001")
    print("⏱️  Simulating trading activity...")
    
    try:
        while True:
            # Simulate market activity
            trade_data = trading_system.simulate_tick()
            
            # Send trade update if there was a signal
            if trade_data:
                dashboard.send_trade_update(trade_data)
                print(f"📈 Trade: {trade_data['action']} {trade_data['symbol']} @ ${trade_data['price']:.2f}")
                
                # Send regime update
                dashboard.send_regime_update({
                    'regime': trade_data['regime'],
                    'confidence': trade_data['confidence']
                })
            
            time.sleep(1)  # 1 second per tick
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down dashboard test...")
        dashboard.disconnect()


if __name__ == "__main__":
    main()