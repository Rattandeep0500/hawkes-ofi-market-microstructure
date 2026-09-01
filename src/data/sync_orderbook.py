import asyncio
import json
import time
from pathlib import Path
from urllib.request import urlopen

import websockets


SYMBOL = "BTCUSDT"
CAPTURE_SECONDS = 600

WS_URL = (
    "wss://stream.binance.com:9443/stream"
    "?streams=btcusdt@depth@100ms/btcusdt@trade"
)

SNAPSHOT_URL = (
    "https://api.binance.com/api/v3/depth"
    "?symbol=BTCUSDT&limit=5000"
)

OUTPUT_DIR = Path("data/live")
OUTPUT_FILE = OUTPUT_DIR / "btc_usdt_10min.jsonl"


def get_snapshot():
    with urlopen(SNAPSHOT_URL, timeout=10) as response:
        return json.loads(
            response.read().decode("utf-8")
        )


async def capture():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    depth_buffer = []

    synchronized = False
    last_update_id = None

    start_time = time.monotonic()

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as output:

        async with websockets.connect(
            WS_URL,
            ping_interval=20,
            ping_timeout=20,
        ) as websocket:

            print("Connected to Binance.")
            print("Buffering market data...")

            while (
                time.monotonic() - start_time
                < CAPTURE_SECONDS
            ):
                raw = await websocket.recv()

                received_at_ms = (
                    time.time_ns() // 1_000_000
                )

                message = json.loads(raw)

                if "data" not in message:
                    continue

                event = message["data"]
                event_type = event.get("e")

                record = {
                    "received_at_ms": received_at_ms,
                    "data": event,
                }

                if event_type == "depthUpdate":
                    depth_buffer.append(record)

                    if not synchronized:
                        snapshot = get_snapshot()

                        snapshot_id = snapshot[
                            "lastUpdateId"
                        ]

                        depth_buffer = [
                            x
                            for x in depth_buffer
                            if x["data"]["u"] > snapshot_id
                        ]

                        if not depth_buffer:
                            continue

                        first = depth_buffer[0]["data"]

                        if not (
                            first["U"]
                            <= snapshot_id + 1
                            <= first["u"]
                        ):
                            depth_buffer.clear()
                            continue

                        output.write(
                            json.dumps(
                                {
                                    "type": "snapshot",
                                    "received_at_ms": (
                                        time.time_ns()
                                        // 1_000_000
                                    ),
                                    "data": snapshot,
                                }
                            )
                            + "\n"
                        )

                        last_update_id = snapshot_id

                        for buffered in depth_buffer:
                            depth = buffered["data"]

                            if depth["u"] <= last_update_id:
                                continue

                            if depth["U"] > last_update_id + 1:
                                raise RuntimeError(
                                    "Sequence gap during synchronization."
                                )

                            output.write(
                                json.dumps(
                                    {
                                        "type": "depth",
                                        "received_at_ms": buffered[
                                            "received_at_ms"
                                        ],
                                        "data": depth,
                                    }
                                )
                                + "\n"
                            )

                            last_update_id = depth["u"]

                        depth_buffer.clear()
                        synchronized = True

                        print(
                            "Order book synchronized."
                        )

                        print(
                            f"Snapshot update ID: "
                            f"{snapshot_id:,}"
                        )

                        print(
                            f"Current update ID: "
                            f"{last_update_id:,}"
                        )

                        continue

                    if event["u"] <= last_update_id:
                        continue

                    if event["U"] > last_update_id + 1:
                        raise RuntimeError(
                            "Sequence gap detected. "
                            "Local order book must restart."
                        )

                    output.write(
                        json.dumps(
                            {
                                "type": "depth",
                                "received_at_ms": received_at_ms,
                                "data": event,
                            }
                        )
                        + "\n"
                    )

                    last_update_id = event["u"]

                elif event_type == "trade":
                    output.write(
                        json.dumps(
                            {
                                "type": "trade",
                                "received_at_ms": received_at_ms,
                                "data": event,
                            }
                        )
                        + "\n"
                    )

                if (
                    synchronized
                    and int(
                        time.monotonic()
                        - start_time
                    )
                    % 30
                    == 0
                ):
                    print(
                        f"elapsed="
                        f"{time.monotonic() - start_time:.0f}s "
                        f"last_update_id="
                        f"{last_update_id:,}"
                    )

    print("Capture complete.")
    print(f"Saved to: {OUTPUT_FILE}")


async def main():
    await capture()


if __name__ == "__main__":
    asyncio.run(main())