import asyncio
import websockets
import json

async def test_bybit_websocket():
    # Test the exact URL from your system
    url = 'wss://stream-testnet.bybit.com/v5/public/linear'
    
    try:
        print(f"Testing WebSocket: {url}")
        
        async with websockets.connect(url) as ws:
            print("Connected successfully!")
            
            # Send subscription
            subscribe_msg = {
                "op": "subscribe",
                "args": ["tickers.BTCUSDT"]
            }
            
            await ws.send(json.dumps(subscribe_msg))
            print("Subscription sent")
            
            # Listen for response
            response = await asyncio.wait_for(ws.recv(), timeout=10)
            data = json.loads(response)
            print(f"Response: {data}")
            
            # Listen for actual data
            data_msg = await asyncio.wait_for(ws.recv(), timeout=10)
            ticker_data = json.loads(data_msg)
            print(f"Ticker data: {ticker_data}")
            
    except websockets.exceptions.InvalidStatusCode as e:
        print(f"HTTP {e.status_code} - WebSocket endpoint rejected")
        print("This means the URL path is wrong")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_bybit_websocket())
