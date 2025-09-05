# pip install websocket-client
import json, socket, websocket

HOST = "stream-testnet.bybit.com"
PATH = "/v5/public/linear"

# Resolve IPv4 only
ipv4 = next(ai[4][0] for ai in socket.getaddrinfo(HOST, 443, socket.AF_INET, socket.SOCK_STREAM))
URL = f"wss://{ipv4}{PATH}"

def on_open(ws):
    ws.send(json.dumps({"op":"subscribe","args":["tickers.BTCUSDT"]}))

def on_message(ws, msg):
    m = json.loads(msg)
    if m.get("topic") == "tickers.BTCUSDT" and m.get("data"):
        t = m["data"][0]
        print(f"[{t['symbol']}] last={t['lastPrice']} mark={t['markPrice']}")

def on_error(ws, err): print("ERROR:", err)
def on_close(ws, *a):  print("closed")

ws = websocket.WebSocketApp(
    URL,
    header=[f"Host: {HOST}", "User-Agent: bybit-ws/ipv4"],
    on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close
)

# IMPORTANT: keep SNI as original hostname
ws.run_forever(sslopt={"server_hostname": HOST}, ping_interval=20, ping_timeout=10)
