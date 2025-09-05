#!/usr/bin/env python3
"""
Test Bybit connection and configure API access
"""

import asyncio
import websockets
import json
import ssl
import certifi
import aiohttp

async def test_websocket_connection():
    """Test different WebSocket URLs and configurations"""

    # Different URLs to try
    urls = [
        "wss://stream-testnet.bybit.com/v5/public/linear",
        "wss://stream-testnet.bybit.com/v5/public/spot",
        "wss://stream-testnet.bybit.com/realtime_public",
        "wss://stream.bybit.com/v5/public/linear",  # Mainnet for comparison
    ]

    print("🔍 Testing Bybit WebSocket connections...")
    print("=" * 60)

    for url in urls:
        print(f"\nTesting: {url}")
        print("-" * 40)

        try:
            # Create SSL context
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            # Try to connect with timeout
            async with websockets.connect(
                    url,
                    ssl=ssl_context,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10
            ) as ws:
                print("✓ Connected successfully!")

                # Send a ping
                await ws.send(json.dumps({"op": "ping"}))

                # Wait for response
                response = await asyncio.wait_for(ws.recv(), timeout=5)
                data = json.loads(response)
                print(f"✓ Received: {data}")

                # Try to subscribe to BTCUSDT
                subscribe_msg = {
                    "op": "subscribe",
                    "args": ["orderbook.1.BTCUSDT"]
                }
                await ws.send(json.dumps(subscribe_msg))

                # Wait for subscription response
                for _ in range(3):
                    response = await asyncio.wait_for(ws.recv(), timeout=5)
                    data = json.loads(response)
                    print(f"✓ Response: {data.get('op', data.get('topic', 'data'))}")

                    if 'topic' in data:
                        print("✓ Receiving market data!")
                        break

                await ws.close()
                print("✓ Connection test successful!")
                return url

        except asyncio.TimeoutError:
            print("✗ Connection timeout")
        except Exception as e:
            print(f"✗ Error: {e}")

    return None

async def test_rest_api():
    """Test REST API connection"""
    print("\n🔍 Testing REST API connection...")
    print("=" * 60)

    urls = [
        "https://api-testnet.bybit.com/v5/market/tickers?category=linear&symbol=BTCUSDT",
        "https://api.bybit.com/v5/market/tickers?category=linear&symbol=BTCUSDT",
    ]

    async with aiohttp.ClientSession() as session:
        for url in urls:
            try:
                print(f"\nTesting: {url}")
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('retCode') == 0:
                            result = data.get('result', {})
                            if 'list' in result and result['list']:
                                ticker = result['list'][0]
                                print(f"✓ BTC Price: ${ticker.get('lastPrice', 'N/A')}")
                                print(f"✓ 24h Volume: {ticker.get('volume24h', 'N/A')}")
                            print("✓ REST API working!")
                        else:
                            print(f"✗ API Error: {data.get('retMsg')}")
                    else:
                        print(f"✗ HTTP {response.status}")
            except Exception as e:
                print(f"✗ Error: {e}")

async def main():
    print("""
    ╔════════════════════════════════════════╗
    ║   🔥 BYBIT CONNECTION TESTER 🔥        ║
    ╚════════════════════════════════════════╝
    """)

    # Test WebSocket
    working_url = await test_websocket_connection()

    # Test REST API
    await test_rest_api()

    if working_url:
        print("\n✅ Connection successful!")
        print(f"Working WebSocket URL: {working_url}")
        print("\nUpdate your trading scripts to use this URL.")
    else:
        print("\n❌ All WebSocket connections failed.")
        print("\nPossible issues:")
        print("1. Internet connection problems")
        print("2. Firewall blocking WebSocket connections")
        print("3. Bybit testnet might be down")
        print("\nTry:")
        print("1. Check your internet connection")
        print("2. Try using a VPN")
        print("3. Check Bybit status at: https://www.bybit.com/announcement")

if __name__ == "__main__":
    asyncio.run(main())