import asyncio
import json
import time
from pathlib import Path

import websockets


SYMBOL = "btcusdt"
DURATION_SECONDS = 600

OUTPUT_DIR = Path("data/live")
OUTPUT_FILE = OUTPUT_DIR / "btcusdt_raw_10min.jsonl"

URL = (
    "wss://stream.binance.com:9443/stream"
    f"?streams={SYMBOL}@depth@100ms/{SYMBOL}@trade"
)


async def capture():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    messages = 0

    async with websockets.connect(URL) as websocket:
        while time.monotonic() - start < DURATION_SECONDS:
            raw = await websocket.recv()
            received_at = time.time_ns() // 1_000_000

            record = {
                "received_at_ms": received_at,
                "data": json.loads(raw),
            }

            with OUTPUT_FILE.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record) + "\n")

            messages += 1

            if messages % 1000 == 0:
                elapsed = time.monotonic() - start
                print(
                    f"messages={messages:,} "
                    f"elapsed={elapsed:.1f}s"
                )


if __name__ == "__main__":
    asyncio.run(capture())