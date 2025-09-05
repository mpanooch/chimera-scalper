#!/usr/bin/env node
/**
 * CHIMERA AI Stats Dashboard Server — hardened
 * Fixes WS keepalive issues (1011 ping timeout) and improves reconnects.
 */

const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const path = require('path');
const sqlite3 = require('sqlite3').verbose();
const WebSocket = require('ws');
const fs = require('fs');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
    cors: { origin: '*', methods: ['GET', 'POST'] },
    // Socket.IO itself has its own heartbeat, but the Python WS is separate.
});

// =============================
// Configuration
// =============================
const PORT = 3001;
const PYTHON_WS_PORT = 8765; // WebSocket port for Python communication
const PYTHON_WS_URL = process.env.PYTHON_WS_URL || `ws://localhost:${PYTHON_WS_PORT}`; // Optionally override full URL incl. path
const DB_PATH = 'demo_training_data.db';

// WS heartbeat tuning (Node <-> Python)
const WS_PING_INTERVAL_MS = 20000; // how often we ping()
const WS_PONG_TIMEOUT_MS = 10000;  // how long we wait for a pong before terminating

// =============================
// Stats tracking
// =============================
let stats = {
    ticks: 0,
    signals: 0,
    successful_trades: 0,
    failed_trades: 0,
    total_pnl: 0,
    prices: {},
    regime: null,
    regime_confidence: 0,
    training_start: new Date(),
    data_points: 0,
    trade_outcomes: 0,
    last_update: new Date(),
};

let expertStats = {
    ross: { signals: 0, wins: 0 },
    bao: { signals: 0, wins: 0 },
    nick: { signals: 0, wins: 0 },
    fabio: { signals: 0, wins: 0 },
};

// =============================
// Static / JSON
// =============================
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());

// Serve the dashboard.html file as the default page
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'dashboard.html'));
});

app.get('/api/stats', (req, res) => {
    res.json({
        ...stats,
        success_rate: stats.signals > 0 ? stats.successful_trades / stats.signals : 0,
        avg_pnl: stats.signals > 0 ? stats.total_pnl / stats.signals : 0,
        experts: expertStats,
        uptime: Date.now() - stats.training_start.getTime(),
    });
});

app.get('/api/trades', async (req, res) => {
    if (!fs.existsSync(DB_PATH)) return res.json([]);
    const db = new sqlite3.Database(DB_PATH);
    db.all(
        `SELECT * FROM trades ORDER BY timestamp DESC LIMIT 100`,
        (err, rows) => {
            if (err) return res.status(500).json({ error: 'Database error' });
            res.json(rows || []);
            db.close();
        }
    );
});

app.get('/api/market-data', async (req, res) => {
    if (!fs.existsSync(DB_PATH)) return res.json([]);
    const db = new sqlite3.Database(DB_PATH);
    db.all(
        `SELECT symbol, price, timestamp, volume, change_pct FROM market_data ORDER BY timestamp DESC LIMIT 1000`,
        (err, rows) => {
            if (err) return res.status(500).json({ error: 'Database error' });
            res.json(rows || []);
            db.close();
        }
    );
});

// =============================
// WebSocket (Node -> Python)
// =============================
let pythonWs = null;
let reconnectTimer = null;
let heartbeatTimer = null;   // sends ping()
let pongTimeoutTimer = null; // waits for 'pong'
let isPythonConnected = false;

function clearHeartbeatTimers() {
    if (heartbeatTimer) { clearInterval(heartbeatTimer); heartbeatTimer = null; }
    if (pongTimeoutTimer) { clearTimeout(pongTimeoutTimer); pongTimeoutTimer = null; }
}

function scheduleReconnect() {
    if (reconnectTimer) return;
    reconnectTimer = setInterval(connectToPython, 5000);
}

function connectToPython() {
    console.log(`🔄 Connecting to Python WS on ${PYTHON_WS_URL} ...`);

    try {
        // Disable perMessageDeflate to avoid rare stalls on some servers
        pythonWs = new WebSocket(PYTHON_WS_URL, {
            perMessageDeflate: false,
            handshakeTimeout: 15000,
        });

        pythonWs.on('open', () => {
            console.log('✅ Connected to Python trading system');
            isPythonConnected = true;
            if (reconnectTimer) { clearInterval(reconnectTimer); reconnectTimer = null; }

            // Ask for initial stats
            safeSend({ type: 'get_stats' });
            // Try multiple subscription message shapes for compatibility with various Python servers
            safeSend({ type: 'subscribe', channels: ['stats', 'prices', 'regime', 'trades'] });
            safeSend({ action: 'subscribe', channels: ['stats', 'prices', 'regime', 'trades'] });
            safeSend({ op: 'subscribe', channels: ['stats', 'prices', 'regime', 'trades'] });
            safeSend({ type: 'hello' });

            // Start *real* WS heartbeat using control frames (ping/pong)
            startHeartbeat();
        });

        pythonWs.on('pong', () => {
            // got pong in time — clear the watchdog
            if (pongTimeoutTimer) { clearTimeout(pongTimeoutTimer); pongTimeoutTimer = null; }
        });

        pythonWs.on('message', (data) => {
            try {
                const raw = typeof data === 'string' ? data : data.toString();
                const msg = JSON.parse(raw);
                console.debug('⬅️  PY MSG TYPE:', msg?.type || msg?.event || msg?.op || '(none)');
                handlePythonMessage(msg);
            } catch (e) {
                // If Python ever sends a non-JSON frame, we ignore it safely.
                try { console.warn('⚠️  Non-JSON from Python:', String(data).slice(0, 160)); } catch {}
            }
        });

        pythonWs.on('error', (err) => {
            console.error('❌ Python WS error:', err.message);
            isPythonConnected = false;
            // Let 'close' drive the reconnect; some errors don't emit 'close'
        });

        pythonWs.on('close', (code, reason) => {
            console.log(`🔌 Python WS closed: ${code} ${reason || '(no reason)'}`);
            isPythonConnected = false;
            clearHeartbeatTimers();
            scheduleReconnect();
        });

    } catch (err) {
        console.error('❌ WS connect error:', err.message);
        isPythonConnected = false;
        scheduleReconnect();
    }
}

function startHeartbeat() {
    clearHeartbeatTimers();

    const doPing = () => {
        if (!pythonWs || pythonWs.readyState !== WebSocket.OPEN) return;

        try {
            // Send a *control frame* ping. ws will auto-respond to server pings too.
            pythonWs.ping();

            // Arm a watchdog to terminate if no pong in time
            if (pongTimeoutTimer) clearTimeout(pongTimeoutTimer);
            pongTimeoutTimer = setTimeout(() => {
                console.warn('⏱️  No pong from Python within timeout — terminating socket');
                // terminate() immediately destroys the socket; close() tries a clean close
                pythonWs.terminate();
            }, WS_PONG_TIMEOUT_MS);
        } catch (e) {
            console.error('❌ ping() failed:', e.message);
        }
    };

    // ping now, then every interval
    doPing();
    heartbeatTimer = setInterval(doPing, WS_PING_INTERVAL_MS);
}

function normalizePrices(input) {
    if (!input) return null;

    // Case A: already a {SYMBOL: price, ...} map
    if (typeof input === 'object' && !Array.isArray(input)) {
        // Sometimes the object is wrapped (e.g., { data: { BTCUSDT: 123 } })
        // We'll try to find numeric leaves:
        const flat = {};
        for (const [k, v] of Object.entries(input)) {
            if (v && typeof v === 'object') {
                // handle { BTCUSDT: { last: 123 } } or { BTCUSDT: { price: 123 } }
                const price = v.last ?? v.price ?? v.close ?? v.c ?? v.p ?? null;
                if (typeof price === 'number') flat[k] = price;
            } else if (typeof v === 'number') {
                flat[k] = v;
            }
        }
        // If we found any numeric leaves, use that, else maybe input is already flat
        const hasVals = Object.keys(flat).length > 0;
        return hasVals ? flat : input;
    }

    // Case B: array of rows
    if (Array.isArray(input)) {
        const out = {};
        for (const row of input) {
            if (!row || typeof row !== 'object') continue;
            // Accept many common column names
            const sym = row.symbol ?? row.pair ?? row.ticker ?? row.sym ?? row.market;
            const price = row.last ?? row.price ?? row.close ?? row.c ?? row.p;
            if (sym && typeof price === 'number') out[String(sym).toUpperCase().replace('/', '')] = price;
        }
        return Object.keys(out).length ? out : null;
    }

    return null;
}

function safeSend(obj) {
    if (!pythonWs || pythonWs.readyState !== WebSocket.OPEN) return false;
    // Avoid sending if there is excessive backpressure
    if (pythonWs.bufferedAmount > 8 * 1024 * 1024) {
        console.warn('⚠️  Backpressure detected; skipping send to avoid stalls');
        return false;
    }
    try {
        pythonWs.send(JSON.stringify(obj));
        return true;
    } catch (e) {
        console.error('❌ send() failed:', e.message);
        return false;
    }
}

function handlePythonMessage(message) {
    try {
        // 0) Normalize payload containers and "type" aliases
        const t = (message?.type || message?.event || message?.op || '').toLowerCase();
        const payload =
            (message && typeof message === 'object' && (message.data || message.payload || message)) || {};
        const regime = payload.regime ?? message.regime ?? payload.state ?? payload.class ?? payload.market_regime;
        const confidence = (
            payload.confidence ?? payload.regime_confidence ?? payload.score ?? payload.prob ?? message.confidence ?? 0
        );
        if (regime) handleRegimeUpdate({ regime, confidence: Number(confidence) || 0 });

        // 1) Heuristics for common shapes/names your Python might use
        //    (we accept both top-level and nested variants)
        const statsObj =
            payload.stats || message.stats ||
            payload.metrics || message.metrics ||
            (('ticks' in payload || 'signals' in payload || 'total_pnl' in payload) ? payload : null);

        const rawPrices =
            (payload.prices && (payload.prices.prices || payload.prices)) ||
            message.prices ||
            payload.tickers ||
            payload.market ||
            (payload.symbol && payload.price ? { [payload.symbol]: payload.price } : null);

        const normPrices = normalizePrices(rawPrices);
        if (normPrices) handlePriceUpdate({ prices: normPrices });

        if (statsObj) updateStats(statsObj);

        // 2) Trade events
        if (t.includes('trade') || payload.action || payload.side || payload.direction) {
            handleTradeUpdate(payload);
        }

        // 3) Tick events
        if (t.includes('tick') || payload.tick || payload.ticks) {
            stats.ticks = (stats.ticks || 0) + (payload.ticks || payload.tick || 1);
            stats.last_update = new Date();
        }

    } catch (e) {
        console.error('❌ Error handling Python message:', e.message);
    }
}

function handleTradeUpdate(data) {
    // Infer expert from regime or other data
    const expert = data.regime?.toLowerCase().includes('trend')
        ? 'ross'
        : data.regime?.toLowerCase().includes('range')
            ? 'bao'
            : data.regime?.toLowerCase().includes('liquidity')
                ? 'nick'
                : 'fabio';
    expertStats[expert].signals++;
    stats.signals++;
    io.emit('trade_update', { ...data, expert, cumulative_pnl: stats.total_pnl });
}

function handleRegimeUpdate(data) {
    stats.regime = data.regime;
    stats.regime_confidence = data.confidence;
    io.emit('regime_update', data);
}

function handlePriceUpdate(data) {
    stats.prices = { ...stats.prices, ...(data.prices || {}) };
    io.emit('stats_update', { ...stats, prices: stats.prices });
}

function calculateTickRate() {
    const elapsed = Date.now() - stats.training_start.getTime();
    return elapsed > 0 ? Math.round((stats.ticks / elapsed) * 60000) : 0; // ticks/min
}

// =============================
// Database monitoring
// =============================
function monitorDatabase() {
    if (!fs.existsSync(DB_PATH)) return;
    const db = new sqlite3.Database(DB_PATH);
    db.get('SELECT COUNT(*) as count FROM market_data', (err, row) => {
        if (!err && row) stats.data_points = row.count;
    });
    db.get('SELECT COUNT(*) as count FROM trades', (err, row) => {
        if (!err && row) stats.trade_outcomes = row.count;
    });
    try {
        const dbStats = fs.statSync(DB_PATH);
        stats.db_size = (dbStats.size / (1024 * 1024)).toFixed(2);
    } catch (_) {
        stats.db_size = '0';
    }
    db.close();
}

function updateStats(data) {
    stats = {
        ...stats,
        ...data,
        last_update: new Date(),
    };

    io.emit("stats_update", stats);
}

// =============================
// Socket.IO clients (browser dashboard)
// =============================
io.on('connection', (socket) => {
    console.log('A client connected');
    
    const initial = {
        ...stats,
        success_rate: stats.signals > 0 ? stats.successful_trades / stats.signals : 0,
        training_progress: Math.min((Date.now() - stats.training_start.getTime()) / (14 * 24 * 60 * 60 * 1000), 1),
        tick_rate: calculateTickRate(),
        python_connected: isPythonConnected,
    };
    socket.emit('stats_update', initial);

    socket.on('disconnect', () => {
        console.log('A client disconnected');
    });

    socket.on('request_update', () => {
        if (pythonWs && pythonWs.readyState === WebSocket.OPEN) {
            safeSend({ type: 'get_stats' });
        }
    });
});

// Periodic broadcast + DB polling
setInterval(() => {
    monitorDatabase();
    const snapshot = {
        ...stats,
        success_rate: stats.signals > 0 ? stats.successful_trades / stats.signals : 0,
        training_progress: Math.min((Date.now() - stats.training_start.getTime()) / (14 * 24 * 60 * 60 * 1000), 1),
        tick_rate: calculateTickRate(),
        experts: expertStats,
        python_connected: isPythonConnected,
    };
    io.emit('stats_update', snapshot);
}, 5000);

// =============================
// HTTP routes
// =============================
app.get('/health', (req, res) => {
    res.json({
        status: 'ok',
        python_connected: isPythonConnected,
        uptime: Date.now() - stats.training_start.getTime(),
        clients_connected: io.engine.clientsCount,
        stats_summary: { ticks: stats.ticks, signals: stats.signals, last_update: stats.last_update },
    });
});

// =============================
// Startup / Shutdown
// =============================
server.listen(PORT, () => {
    console.log(`\n🔥 CHIMERA AI DASHBOARD SERVER 🔥\n- Dashboard: http://localhost:${PORT}\n- API:       http://localhost:${PORT}/api/stats\n- Health:    http://localhost:${PORT}/health\n`);
    connectToPython();
    monitorDatabase();
});

function gracefulExit(signal) {
    console.log(`\n🛑 ${signal} — shutting down...`);
    clearHeartbeatTimers();
    if (pythonWs && pythonWs.readyState === WebSocket.OPEN) {
        try { pythonWs.close(1001, 'server shutting down'); } catch (_) {}
    }
    server.close(() => {
        console.log('✅ Dashboard server shut down gracefully');
        process.exit(0);
    });
}

process.on('SIGTERM', () => gracefulExit('SIGTERM'));
process.on('SIGINT', () => gracefulExit('SIGINT'));