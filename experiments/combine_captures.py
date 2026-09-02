import json
from pathlib import Path

import pandas as pd


INPUT_FILES = [
    Path("data/live/capture_02.jsonl"),
    Path("data/live/capture_03.jsonl"),
]

OUTPUT_FILE = Path(
    "data/processed/combined_capture_records.parquet"
)


def load_capture(path, capture_id):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            record = json.loads(line)

            rows.append(
                {
                    "capture_id": capture_id,
                    "type": record["type"],
                    "received_at_ms": int(
                        record["received_at_ms"]
                    ),
                    "data": json.dumps(
                        record["data"],
                        separators=(",", ":"),
                    ),
                }
            )

    if not rows:
        raise RuntimeError(
            f"Capture is empty: {path}"
        )

    return pd.DataFrame(rows)


def validate_capture_boundaries(data):
    for capture_id, frame in data.groupby(
        "capture_id",
        sort=False,
    ):
        if frame.empty:
            raise RuntimeError(
                f"Empty capture: {capture_id}"
            )

        timestamps = frame[
            "received_at_ms"
        ].to_numpy()

        if len(timestamps) > 1:
            if (
                timestamps[1:]
                < timestamps[:-1]
            ).any():
                raise RuntimeError(
                    f"Reception timestamps are not "
                    f"monotonic in {capture_id}."
                )

        snapshots = frame[
            frame["type"] == "snapshot"
        ]

        if len(snapshots) != 1:
            raise RuntimeError(
                f"{capture_id} has "
                f"{len(snapshots)} snapshots."
            )


def print_summary(data):
    print(
        f"Total records: {len(data):,}"
    )

    print()

    print(
        "Records by capture:"
    )

    summary = (
        data
        .groupby(
            ["capture_id", "type"]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    print(
        summary.to_string()
    )

    print()

    for capture_id, frame in data.groupby(
        "capture_id",
        sort=False,
    ):
        start = pd.to_datetime(
            frame["received_at_ms"].min(),
            unit="ms",
            utc=True,
        )

        end = pd.to_datetime(
            frame["received_at_ms"].max(),
            unit="ms",
            utc=True,
        )

        print(
            f"{capture_id}: "
            f"{start} -> {end}"
        )


def main():
    frames = []

    for index, path in enumerate(
        INPUT_FILES,
        start=2,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing capture file: {path}"
            )

        capture_id = (
            f"capture_{index:02d}"
        )

        frame = load_capture(
            path,
            capture_id,
        )

        frames.append(frame)

    data = pd.concat(
        frames,
        ignore_index=True,
    )

    validate_capture_boundaries(
        data
    )

    data = data.sort_values(
        [
            "capture_id",
            "received_at_ms",
        ]
    ).reset_index(
        drop=True
    )

    print_summary(
        data
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_parquet(
        OUTPUT_FILE,
        engine="pyarrow",
        index=False,
    )

    print()

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()