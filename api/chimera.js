export default function handler(req, res) {
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  const { action } = req.query;
  
  // Sample data
  const chimeraData = {
    status: 'active',
    message: 'CHIMERA system operational',
    performance: {
      total_pnl: 125.50,
      pnl_percent: 1.25,
      win_rate: 65.5,
      total_trades: 42,
      balance: 10125.50
    },
    regime: {
      current: 'TRENDING',
      confidence: 0.85
    },
    market: {
      BTCUSDT: 111580.70,
      ETHUSDT: 4336.45,
      SOLUSDT: 204.51
    },
    system: {
      processing_speed: 15.2,
      total_ticks: 1250,
      signals_generated: 8,
      model_accuracy: 72.3
    },
    data_collection: {
      market_data_points: 1250,
      trade_outcomes: 42,
      db_size_mb: 2.5,
      learning_rate: 0.001
    },
    recent_trades: [
      {
        timestamp: '2024-01-15T10:30:00Z',
        symbol: 'BTCUSDT',
        action: 'BUY',
        price: 111500.00,
        regime: 'TRENDING',
        confidence: 0.85,
        pnl_pct: 2.1
      }
    ],
    last_updated: new Date().toISOString()
  };

  if (action === 'stats') {
    res.status(200).json(chimeraData);
  } else if (action === 'health') {
    res.status(200).json({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      endpoints: {
        'GET ?action=stats': 'Returns current trading stats',
        'GET ?action=health': 'Performs a health check'
      }
    });
  } else if (action === 'update' && req.method === 'POST') {
    // In a real implementation, you'd save this data
    res.status(200).json({ status: 'success', message: 'Data updated' });
  } else {
    res.status(404).json({
      status: 'error',
      message: 'Endpoint not found. Use ?action=stats, ?action=update, or ?action=health.'
    });
  }
}