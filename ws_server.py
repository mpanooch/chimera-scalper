import asyncio, json, websockets

async def handler(ws):
    i = 0
    while True:
        i += 1
        await ws.send(json.dumps({
            "type": "stats_update",
            "data": {"ticks": i, "signals": 1, "successful_trades": 1, "total_pnl": 0.5}
        }))
        await asyncio.sleep(2)

async def main():
    print("✅ Serving WS on ws://127.0.0.1:8765")
    async with websockets.serve(handler, "127.0.0.1", 8765):
        await asyncio.Future()

asyncio.run(main())
