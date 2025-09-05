#!/usr/bin/env python3
"""
ML Trading System (Public + Private WS) with optional REST orders (Bybit V5, Testnet by default)

Key features:
- Public WebSocket (tickers) for BTCUSDT/ETHUSDT/SOLUSDT
- Private WebSocket (auth) to receive private topics (e.g., "order") if desired
- Optional REST order placement on ML signals (PostOnly limit) when --live is specified
- Robust V5 signing and headers; includes X-BAPI-SIGN-TYPE: 2
- Basic reconnect logic for WebSockets
- Safe defaults: paper trading unless --live is passed

Usage:
  python ml_trading_plus.py --symbols BTCUSDT,ETHUSDT --live
"""

import asyncio
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from collections import deque
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import websockets
import requests

# ------------------ Config / Constants ------------------
BASE_URL_TESTNET = "https://api-testnet.bybit.com"
RECV_WINDOW = "5000"
PUBLIC_WS_TESTNET = "wss://stream-testnet.bybit.com/v5/public/linear"
PRIVATE_WS_TESTNET = "wss://stream-testnet.bybit.com/v5/private"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Model ------------------
class RegimeClassifier(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=64, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=2, batch_first=True, dropout=0.2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        return self.classifier(h_n[-1])

# ------------------ Helpers ------------------
def load_api_config(path: str = "bybit_config.json"):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    test = cfg.get("testnet", {})
    api_key = test.get("api_key", "")
    api_secret = test.get("api_secret", "")
    use_testnet = bool(test.get("use_testnet", True))
    if not api_key or not api_secret:
        raise SystemExit("❌ Missing api_key/api_secret in bybit_config.json under 'testnet'.")
    return api_key, api_secret, use_testnet

def urlencode_sorted(params: Dict[str, Any]) -> str:
    from urllib.parse import urlencode
    items = [(k, v) for k, v in params.items() if v is not None]
    items.sort(key=lambda x: x[0])
    return urlencode(items, doseq=True)

def sign_v5(secret: str, timestamp: str, api_key: str, recv_window: str, payload: str) -> str:
    to_sign = f"{timestamp}{api_key}{recv_window}{payload}"
    return hmac.new(secret.encode("utf-8"), to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

def headers_v5(api_key: str, timestamp: str, signature: str, recv_window: str = RECV_WINDOW) -> Dict[str, str]:
    return {
        "X-BAPI-API-KEY": api_key,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-SIGN": signature,
        "X-BAPI-RECV-WINDOW": recv_window,
        "X-BAPI-SIGN-TYPE": "2",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

def server_time_skew():
    try:
        r = requests.get(f"{BASE_URL_TESTNET}/v5/market/time", timeout=10)
        srv = r.json().get("result", {}).get("timeSecond")
        if srv:
            now_s = int(time.time())
            return abs(now_s - int(srv))
    except Exception:
        pass
    return None

# WebSocket private auth signature (Bybit V5 example approach)
def ws_auth_signature(secret: str, expires_ms: int) -> str:
    # Signature over "GET/realtime{expires_ms}"
    payload = f"GET/realtime{expires_ms}"
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

# ------------------ Trading System ------------------
@dataclass
class Signal:
    symbol: str
    side: str   # "BUY" or "SELL"
    price: float
    regime: str
    confidence: float

class MLTradingSystem:
    def __init__(self, symbols: List[str], live: bool):
        self.symbols = symbols
        self.live = live

        self.api_key, self.api_secret, self.use_testnet = load_api_config("bybit_config.json")
        if not self.use_testnet:
            print("⚠️ Config shows use_testnet=false. This script is intended for TESTNET.")

        self.public_ws_url = PUBLIC_WS_TESTNET
        self.private_ws_url = PRIVATE_WS_TESTNET
        self.base_url = BASE_URL_TESTNET

        # Load model (optional)
        self.sequence_length = 60
        self.model = self._load_model()

        # State
        self.symbol_data: Dict[str, deque] = {s: deque(maxlen=self.sequence_length) for s in self.symbols}
        self.last_prices: Dict[str, float] = {}
        self.last_bid: Dict[str, float] = {}
        self.last_ask: Dict[str, float] = {}
        self.stats = {"ticks": 0, "signals": 0, "buy": 0, "sell": 0}

        # Control
        self._stop = asyncio.Event()

    def _load_model(self) -> Optional[RegimeClassifier]:
        model_path = Path("models/best_regime_classifier.pth")
        if not model_path.exists():
            print("⚠️ No ML model found; will use testing mode (random high-confidence blips).")
            return None
        model = RegimeClassifier()
        ckpt = torch.load(str(model_path), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()
        print(f"✅ Loaded ML model from {model_path}")
        return model

    # ------------------ Feature & Signal ------------------
    def _push_ticker(self, symbol: str, t: Dict[str, Any]):
        price = float(t.get("lastPrice", 0) or 0)
        bid = float(t.get("bid1Price", price) or price)
        ask = float(t.get("ask1Price", price) or price)
        vol24 = float(t.get("volume24h", 0) or 0)
        chg = float(t.get("price24hPcnt", 0) or 0)

        self.last_prices[symbol] = price
        self.last_bid[symbol] = bid
        self.last_ask[symbol] = ask

        f = np.zeros(14, dtype=np.float32)
        f[0] = chg
        f[1] = np.log1p(vol24 / 1e6)
        f[2] = ((ask - bid) / price) * 10000 if price > 0 else 0
        f[11] = 1.0 if symbol in ("BTCUSDT", "ETHUSDT") else 0.0
        f[12] = 1.0

        self.symbol_data[symbol].append(f)

    def _maybe_signal(self, symbol: str) -> Optional[Signal]:
        if len(self.symbol_data[symbol]) < self.sequence_length:
            return None

        price = self.last_prices.get(symbol)
        if not price:
            return None

        if self.model is None:
            # occasionally emit a test signal (very low rate)
            if np.random.random() > 0.99:
                side = "BUY" if np.random.random() > 0.5 else "SELL"
                return Signal(symbol, side, price, "TEST", float(np.random.random()))
            return None

        seq = np.array(self.symbol_data[symbol])
        x = torch.FloatTensor(seq).unsqueeze(0).to(device)
        with torch.no_grad():
            out = self.model(x)
            probs = torch.softmax(out, dim=1)
            regime_idx = int(out.argmax(1).item())
            conf = float(probs[0, regime_idx].item())

        if conf > 0.75:
            names = ["TREND_UP", "TREND_DOWN", "RANGE_BOUND", "LIQ_EVENT", "CHAOTIC"]
            if regime_idx == 0:
                self.stats["buy"] += 1
                side = "BUY"
            elif regime_idx == 1:
                self.stats["sell"] += 1
                side = "SELL"
            else:
                return None
            self.stats["signals"] += 1
            return Signal(symbol, side, price, names[regime_idx], conf)
        return None

    # ------------------ REST: Order placement ------------------
    def place_postonly_order(self, signal: Signal) -> None:
        """Submit a PostOnly limit order far from market to avoid fills (Testnet)."""
        endpoint = "/v5/order/create"
        url = f"{self.base_url}{endpoint}"

        qty = "0.001"  # small default
        bid = self.last_bid.get(signal.symbol, signal.price)
        ask = self.last_ask.get(signal.symbol, signal.price)

        if signal.side == "BUY":
            # place below bid a bit to ensure maker
            price = f"{max(1.0, bid - 50):.2f}"
            side = "Buy"
        else:
            price = f"{ask + 50:.2f}"
            side = "Sell"

        body = {
            "category": "linear",
            "symbol": signal.symbol,
            "side": side,
            "orderType": "Limit",
            "qty": qty,
            "price": price,
            "timeInForce": "PostOnly",
            "reduceOnly": False
        }

        timestamp = str(int(time.time() * 1000))
        payload = json.dumps(body, separators=(",", ":"))
        sig = sign_v5(self.api_secret, timestamp, self.api_key, RECV_WINDOW, payload)
        headers = headers_v5(self.api_key, timestamp, sig)

        r = requests.post(url, data=payload, headers=headers, timeout=20)
        try:
            j = r.json()
        except Exception:
            print(f"❌ Order HTTP {r.status_code}: {r.text[:200]}")
            return

        if r.status_code == 200 and j.get("retCode") == 0:
            oid = j["result"].get("orderId")
            print(f"📝 Order submitted: {signal.symbol} {side} {qty} @ {price} (orderId={oid})")
        else:
            print(f"❌ Order error: HTTP {r.status_code} retCode={j.get('retCode')} retMsg={j.get('retMsg')}")

    # ------------------ WS tasks ------------------
    async def public_ws_task(self):
        backoff = 1
        while not self._stop.is_set():
            try:
                async with websockets.connect(self.public_ws_url, ping_interval=20, ping_timeout=10) as ws:
                    print(f"✅ Public WS connected: {self.public_ws_url}")
                    # subscribe
                    for s in self.symbols:
                        await ws.send(json.dumps({"op": "subscribe", "args": [f"tickers.{s}"]}))
                    print(f"✅ Subscribed: {', '.join(self.symbols)}")

                    backoff = 1  # reset backoff on success

                    while not self._stop.is_set():
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=30)
                        except asyncio.TimeoutError:
                            await ws.send(json.dumps({"op": "ping"}))
                            continue

                        data = json.loads(msg)

                        # heartbeat
                        if data.get("op") == "ping":
                            await ws.send(json.dumps({"op": "pong"}))
                            continue

                        if "topic" in data and "data" in data and "tickers" in data["topic"]:
                            sym = data["topic"].split(".")[-1]
                            # data["data"] can be dict or list depending on library; normalize
                            t = data["data"][0] if isinstance(data["data"], list) else data["data"]
                            self._push_ticker(sym, t)

                            self.stats["ticks"] += 1
                            if self.stats["ticks"] % 100 == 0:
                                prices = ", ".join(f"{s}: {self.last_prices.get(s, 0):,.2f}" for s in self.symbols)
                                print(f"\r📊 Ticks={self.stats['ticks']} Signals={self.stats['signals']} ({self.stats['buy']}/{self.stats['sell']}) | {prices}", end="")

                            sig = self._maybe_signal(sym)
                            if sig:
                                ts = datetime.now().strftime("%H:%M:%S")
                                print(f"\n🎯 [{ts}] {sig.symbol} {sig.side} @ {sig.price:,.2f} ({sig.regime}, conf={sig.confidence:.1%})")
                                if self.live:
                                    self.place_postonly_order(sig)

            except Exception as e:
                print(f"\n⚠️ Public WS error: {e}. Reconnecting soon...")
                await asyncio.sleep(min(60, backoff))
                backoff = min(backoff * 2, 60)

    async def private_ws_task(self):
        """Optional private WS: authenticate and listen to 'order' updates (if desired later)."""
        backoff = 1
        while not self._stop.is_set():
            try:
                async with websockets.connect(self.private_ws_url, ping_interval=20, ping_timeout=10) as ws:
                    print(f"✅ Private WS connected: {self.private_ws_url}")

                    expires = int((time.time() + 10) * 1000)
                    sig = ws_auth_signature(self.api_secret, expires)
                    await ws.send(json.dumps({"op": "auth", "args": [self.api_key, expires, sig]}))

                    # Wait for auth response
                    auth_ok = False
                    try:
                        auth_msg = await asyncio.wait_for(ws.recv(), timeout=5)
                        j = json.loads(auth_msg)
                        if j.get("success") or (isinstance(j.get("retCode"), int) and j.get("retCode") == 0):
                            auth_ok = True
                    except asyncio.TimeoutError:
                        pass

                    if not auth_ok:
                        print("❌ Private WS auth failed (check API key/secret/IP permissions).")
                        return

                    # Subscribe to private topic(s) as needed
                    await ws.send(json.dumps({"op": "subscribe", "args": ["order"]}))
                    print("✅ Subscribed to private topic: order")

                    backoff = 1
                    while not self._stop.is_set():
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=30)
                        except asyncio.TimeoutError:
                            await ws.send(json.dumps({"op": "ping"}))
                            continue
                        data = json.loads(msg)
                        if data.get("op") == "ping":
                            await ws.send(json.dumps({"op": "pong"}))
                            continue
                        # Print order updates
                        if data.get("topic") == "order":
                            print(f"\n🔔 Private update: {data}")

            except Exception as e:
                print(f"\n⚠️ Private WS error: {e}. Reconnecting soon...")
                await asyncio.sleep(min(60, backoff))
                backoff = min(backoff * 2, 60)

    async def run(self, with_private: bool):
        # Print server time skew
        skew = server_time_skew()
        if skew is not None:
            print(f"Server time skew: {skew}s")
            if skew > 5:
                print("⚠️ Large clock skew detected. Sync system time to avoid auth issues.")

        print(f"""
╔═══════════════════════════════════════════════╗
║   🔥 ML TRADING SYSTEM - LIVE (Bybit V5)      ║
║   Mode: {'LIVE orders ON' if self.live else 'Paper only'}              ║
║   Network: Testnet                            ║
║   Symbols: {', '.join(self.symbols):<35}      ║
╚═══════════════════════════════════════════════╝
""")

        tasks = [asyncio.create_task(self.public_ws_task())]
        if with_private:
            tasks.append(asyncio.create_task(self.private_ws_task()))

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass

# ------------------ CLI ------------------
def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="ML Trading System (Bybit V5)")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT", help="Comma-separated symbols")
    p.add_argument("--live", action="store_true", help="Place PostOnly REST orders on signals")
    p.add_argument("--private", action="store_true", help="Connect private WS (auth) to receive order updates")
    return p.parse_args()

async def main():
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    system = MLTradingSystem(symbols=symbols, live=args.live)
    await system.run(with_private=args.private)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
