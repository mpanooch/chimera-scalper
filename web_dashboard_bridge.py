import requests
import json
import time
from datetime import datetime
import logging

class WebDashboardBridge:
    def __init__(self, trading_system=None, api_url="https://chimera-scalper.herokuapp.com"):
        self.trading_system = trading_system
        self.api_url = api_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'CHIMERA-Scalper/1.0'
        })
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"🌐 Web Dashboard Bridge initialized - sending to {self.api_url}")
    
    def start(self):
        """Start the web dashboard bridge"""
        self.logger.info("🚀 Web Dashboard Bridge started")
        # Test the connection
        self.test_connection()
    
    def update_regime(self, regime_data):
        """Update regime detection data"""
        try:
            # Send regime data to the web API
            self.send_data({
                'type': 'regime_update',
                'regime': regime_data
            })
        except Exception as e:
            self.logger.error(f"Failed to update regime: {e}")
    
    def update_performance(self, performance_data):
        """Update performance metrics"""
        try:
            # Send performance data to the web API
            self.send_data({
                'type': 'performance_update',
                'performance': performance_data
            })
        except Exception as e:
            self.logger.error(f"Failed to update performance: {e}")
    
    def update_market_data(self, market_data):
        """Update market data"""
        try:
            # Send market data to the web API
            self.send_data({
                'type': 'market_update',
                'market': market_data
            })
        except Exception as e:
            self.logger.error(f"Failed to update market data: {e}")
    
    def log_trade(self, trade_data):
        """Log a trade"""
        try:
            # Send trade data to the web API
            self.send_data({
                'type': 'trade_log',
                'trade': trade_data
            })
        except Exception as e:
            self.logger.error(f"Failed to log trade: {e}")
    
    def send_data(self, data):
        """Send trading data to the web API"""
        try:
            # Add timestamp
            data['timestamp'] = datetime.now().isoformat()
            
            # Send to API with action=update parameter
            response = self.session.post(
                f"{self.api_url}?action=update",
                json=data,
                timeout=5
            )
            
            if response.status_code == 200:
                self.logger.debug("✅ Data sent successfully to web API")
                return True
            else:
                self.logger.warning(f"⚠️ API returned status {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Failed to send data to web API: {e}")
            return False
    
    def test_connection(self):
        """Test the API connection"""
        try:
            response = self.session.get(f"{self.api_url}?action=health", timeout=5)
            if response.status_code == 200:
                self.logger.info("✅ Web API connection test successful")
                return True
            else:
                self.logger.warning(f"⚠️ Web API health check failed: {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"❌ Web API connection test failed: {e}")
            return False