import asyncio
import json

import websockets


URL = "wss://stream.binance.com:9443/ws/btcusdt@depth@100ms"


async def main():
    async with websockets.connect(
        URL,
        ping_interval=20,
        ping_timeout=20,
    ) as websocket:
        message = await websocket.recv()
        data = json.loads(message)

        print("BINANCE DEPTH CONNECTION OK")
        print(json.dumps(data, indent=2))


if __name__ == "__main__":
    asyncio.run(main())