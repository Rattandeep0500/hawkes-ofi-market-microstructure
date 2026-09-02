import asyncio
import json
import time
from pathlib import Path

import requests
import websockets


SYMBOL = "BTCUSDT"
CAPTURE_SECONDS = 600

WS_URL = (
    "wss://stream.binance.com:9443/stream"
    "?streams=btcusdt@depth@100ms/btcusdt@trade"
)

SNAPSHOT_URL = (
    "https://api.binance.com/api/v3/depth"
)

OUTPUT_DIR = Path("data/live")


def next_capture_file():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = sorted(
        OUTPUT_DIR.glob(
            "capture_*.jsonl"
        )
    )

    numbers = []

    for path in existing:
        try:
            number = int(
                path.stem.split("_")[-1]
            )
            numbers.append(number)
        except ValueError:
            continue

    next_number = (
        max(numbers) + 1
        if numbers
        else 1
    )

    return (
        OUTPUT_DIR
        / f"capture_{next_number:02d}.jsonl"
    )


def get_snapshot():
    response = requests.get(
        SNAPSHOT_URL,
        params={
            "symbol": SYMBOL,
            "limit": 1000,
        },
        headers={
            "User-Agent":
                "hawkes-ofi-market-microstructure/1.0"
        },
        timeout=15,
    )

    response.raise_for_status()

    return response.json()


def write_record(
    output,
    record_type,
    data,
    received_at_ms,
):
    output.write(
        json.dumps(
            {
                "type": record_type,
                "received_at_ms":
                    received_at_ms,
                "data": data,
            },
            separators=(
                ",",
                ":",
            ),
        )
        + "\n"
    )


async def capture():
    output_file = next_capture_file()

    depth_buffer = []

    snapshot = None
    snapshot_id = None

    synchronized = False
    last_update_id = None
    synchronized_at = None

    depth_events = 0
    trade_events = 0

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as output:

        async with websockets.connect(
            WS_URL,
            ping_interval=20,
            ping_timeout=20,
            max_size=None,
        ) as websocket:

            print(
                "Connected to Binance."
            )

            print(
                "Buffering depth events..."
            )

            while not synchronized:

                raw = await websocket.recv()

                received_at_ms = (
                    time.time_ns()
                    // 1_000_000
                )

                message = json.loads(raw)

                if "data" not in message:
                    continue

                event = message["data"]

                if (
                    event.get("e")
                    != "depthUpdate"
                ):
                    continue

                depth_buffer.append(
                    (
                        received_at_ms,
                        event,
                    )
                )

                if (
                    snapshot is None
                    and len(depth_buffer) >= 10
                ):
                    print(
                        "Requesting one "
                        "order-book snapshot..."
                    )

                    try:
                        snapshot = (
                            await asyncio.to_thread(
                                get_snapshot
                            )
                        )
                    except Exception as error:
                        print(
                            f"Snapshot request failed: "
                            f"{error}"
                        )

                        snapshot = None
                        depth_buffer.clear()

                        await asyncio.sleep(2)

                        continue

                    snapshot_id = int(
                        snapshot[
                            "lastUpdateId"
                        ]
                    )

                    print(
                        f"Snapshot ID: "
                        f"{snapshot_id:,}"
                    )

                if snapshot_id is None:
                    continue

                candidates = [
                    item
                    for item in depth_buffer
                    if int(item[1]["u"])
                    > snapshot_id
                ]

                if not candidates:
                    continue

                first_event = candidates[0][1]

                first_u = int(
                    first_event["u"]
                )

                first_U = int(
                    first_event["U"]
                )

                if first_u < (
                    snapshot_id + 1
                ):
                    depth_buffer = candidates
                    continue

                if first_U > (
                    snapshot_id + 1
                ):
                    depth_buffer = []

                    snapshot = None
                    snapshot_id = None

                    print(
                        "Snapshot behind "
                        "buffered stream. "
                        "Restarting sync..."
                    )

                    continue

                if not (
                    first_U
                    <= snapshot_id + 1
                    <= first_u
                ):
                    continue

                write_record(
                    output,
                    "snapshot",
                    snapshot,
                    time.time_ns()
                    // 1_000_000,
                )

                last_update_id = (
                    snapshot_id
                )

                for (
                    received_at_ms,
                    depth,
                ) in candidates:

                    event_first = int(
                        depth["U"]
                    )

                    event_last = int(
                        depth["u"]
                    )

                    if event_last <= (
                        last_update_id
                    ):
                        continue

                    if event_first > (
                        last_update_id + 1
                    ):
                        raise RuntimeError(
                            "Sequence gap during "
                            "initial synchronization."
                        )

                    write_record(
                        output,
                        "depth",
                        depth,
                        received_at_ms,
                    )

                    last_update_id = (
                        event_last
                    )

                    depth_events += 1

                depth_buffer.clear()

                synchronized = True

                synchronized_at = (
                    time.monotonic()
                )

                print(
                    "Order book synchronized."
                )

                print(
                    f"Current update ID: "
                    f"{last_update_id:,}"
                )

                print(
                    "Starting 10-minute "
                    "research capture..."
                )

            while (
                time.monotonic()
                - synchronized_at
                < CAPTURE_SECONDS
            ):

                raw = await websocket.recv()

                received_at_ms = (
                    time.time_ns()
                    // 1_000_000
                )

                message = json.loads(raw)

                if "data" not in message:
                    continue

                event = message["data"]

                event_type = event.get("e")

                if event_type == "depthUpdate":

                    event_first = int(
                        event["U"]
                    )

                    event_last = int(
                        event["u"]
                    )

                    if event_last <= (
                        last_update_id
                    ):
                        continue

                    if event_first > (
                        last_update_id + 1
                    ):
                        raise RuntimeError(
                            "Sequence gap detected. "
                            "Local order book is "
                            "invalid."
                        )

                    write_record(
                        output,
                        "depth",
                        event,
                        received_at_ms,
                    )

                    last_update_id = (
                        event_last
                    )

                    depth_events += 1

                elif event_type == "trade":

                    write_record(
                        output,
                        "trade",
                        event,
                        received_at_ms,
                    )

                    trade_events += 1

                elapsed = (
                    time.monotonic()
                    - synchronized_at
                )

                if (
                    depth_events > 0
                    and depth_events % 1000 == 0
                    and (
                        trade_events == 0
                        or depth_events
                        != getattr(
                            capture,
                            "_last_reported_depth",
                            -1,
                        )
                    )
                ):
                    print(
                        f"elapsed={elapsed:.0f}s | "
                        f"depth={depth_events:,} | "
                        f"trades={trade_events:,} | "
                        f"update_id="
                        f"{last_update_id:,}"
                    )

                    capture._last_reported_depth = (
                        depth_events
                    )

    print()
    print(
        "Capture complete."
    )

    print(
        f"Depth events: "
        f"{depth_events:,}"
    )

    print(
        f"Trade events: "
        f"{trade_events:,}"
    )

    print(
        f"Final update ID: "
        f"{last_update_id:,}"
    )

    print(
        f"Saved to: {output_file}"
    )


async def main():
    await capture()


if __name__ == "__main__":
    asyncio.run(main())