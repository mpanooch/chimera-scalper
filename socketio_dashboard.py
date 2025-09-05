#!/usr/bin/env python3
"""
Socket.IO Dashboard Bridge for CHIMERA Scalper
Connects to the Node.js dashboard server using Socket.IO
"""

import socketio
import threading
import time
import json
from datetime import datetime


class SocketIODashboardBridge:
    def __init__(self, trading_system, server_url='http://localhost:3001'):
        self.trading_system = trading_system
        self.server_url = server_url
        self.sio = socketio.Client()
        self.connected = False
        self.setup_events()
        
    def setup_events(self):
        @self.sio.event
        def connect():
            print("✅ Connected to dashboard server")
            self.connected = True
            # Send initial stats
            self.send_stats_update()
            
        @self.sio.event
        def disconnect():
            print("❌ Disconnected from dashboard server")
            self.connected = False
            
        @self.sio.event
        def connect_error(data):
            print(f"❌ Connection error: {data}")
            
    def start(self):
        """Start the dashboard bridge in a separate thread"""
        def connect_loop():
            while True:
                try:
                    if not self.connected:
                        print(f"🔄 Connecting to dashboard server at {self.server_url}")
                        self.sio.connect(self.server_url)
                        
                    # Send periodic updates
                    if self.connected:
                        self.send_stats_update()
                        
                    time.sleep(5)  # Update every 5 seconds
                    
                except Exception as e:
                    print(f"Dashboard connection error: {e}")
                    self.connected = False
                    time.sleep(5)
                    
        thread = threading.Thread(target=connect_loop, daemon=True)
        thread.start()
        return thread
        
    def send_stats_update(self):
        """Send stats update to dashboard"""
        if not self.connected:
            return
            
        try:
            stats = {
                'ticks': self.trading_system.stats.get('ticks', 0),
                'signals': self.trading_system.stats.get('signals', 0),
                'successful_trades': self.trading_system.stats.get('successful_trades', 0),
                'failed_trades': self.trading_system.stats.get('failed_trades', 0),
                'total_pnl': self.trading_system.stats.get('total_pnl', 0),
                'prices': dict(self.trading_system.stats.get('last_prices', {})),
                'training_start': self.trading_system.training_start.isoformat() if hasattr(self.trading_system, 'training_start') else datetime.now().isoformat(),
                'model_confidence': getattr(self.trading_system, 'current_confidence', 0),
                'tick_rate': self.calculate_tick_rate(),
                'success_rate': self.calculate_success_rate(),
                'data_points': getattr(self.trading_system, 'training_samples_collected', 0),
                'trade_outcomes': self.trading_system.stats.get('successful_trades', 0) + self.trading_system.stats.get('failed_trades', 0),
                'training_progress': self.calculate_training_progress(),
                'days_remaining': self.calculate_days_remaining()
            }
            
            self.sio.emit('stats_update', stats)
            
        except Exception as e:
            print(f"Error sending stats update: {e}")
            
    def send_regime_update(self, regime_data):
        """Send regime update to dashboard"""
        if not self.connected:
            return
            
        try:
            self.sio.emit('regime_update', regime_data)
        except Exception as e:
            print(f"Error sending regime update: {e}")
            
    def send_trade_update(self, trade_data):
        """Send trade update to dashboard"""
        if not self.connected:
            return
            
        try:
            formatted_data = {
                'symbol': trade_data.get('symbol', ''),
                'action': trade_data.get('action', ''),
                'price': trade_data.get('price', 0),
                'regime': trade_data.get('regime', 'UNKNOWN'),
                'confidence': trade_data.get('confidence', 0),
                'expert': trade_data.get('expert', 'AI_Regime'),
                'pnl_pct': trade_data.get('pnl_pct'),
                'cumulative_pnl': trade_data.get('cumulative_pnl', self.trading_system.stats.get('total_pnl', 0))
            }
            
            self.sio.emit('trade_update', formatted_data)
            
        except Exception as e:
            print(f"Error sending trade update: {e}")
            
    def calculate_tick_rate(self):
        """Calculate ticks per minute"""
        try:
            if hasattr(self.trading_system, 'start_time'):
                elapsed = time.time() - self.trading_system.start_time
                if elapsed > 0:
                    return (self.trading_system.stats.get('ticks', 0) / elapsed) * 60
            return 0
        except:
            return 0
            
    def calculate_success_rate(self):
        """Calculate success rate"""
        try:
            successful = self.trading_system.stats.get('successful_trades', 0)
            failed = self.trading_system.stats.get('failed_trades', 0)
            total = successful + failed
            return successful / total if total > 0 else 0
        except:
            return 0
            
    def calculate_training_progress(self):
        """Calculate training progress"""
        try:
            if hasattr(self.trading_system, 'training_start') and hasattr(self.trading_system, 'training_duration'):
                elapsed = time.time() - self.trading_system.training_start.timestamp()
                total = self.trading_system.training_duration.total_seconds()
                return min(1.0, elapsed / total if total > 0 else 0)
            return 0
        except:
            return 0
            
    def calculate_days_remaining(self):
        """Calculate days remaining in training"""
        try:
            if hasattr(self.trading_system, 'training_start') and hasattr(self.trading_system, 'training_duration'):
                elapsed = time.time() - self.trading_system.training_start.timestamp()
                total = self.trading_system.training_duration.total_seconds()
                remaining = max(0, total - elapsed)
                return remaining / 86400  # Convert to days
            return 14  # Default 14 days
        except:
            return 14
            
    def disconnect(self):
        """Disconnect from dashboard server"""
        if self.connected:
            self.sio.disconnect()
            self.connected = False