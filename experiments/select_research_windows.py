from pathlib import Path

import numpy as np
import pandas as pd


DEPTH_PATH = Path("data/raw/depth/binance/BTCUSDT/2026-06.parquet")
TRADES_PATH = Path("data/raw/trades/binance/BTCUSDT/2026-06.parquet")
OUTPUT_PATH = Path("data/processed/research_windows.csv")

TARGET_DURATION_SECONDS = 300
MIN_DURATION_SECONDS = 290
MIN_TRADES = 1000
MIN_DEPTH_UPDATES = 1000
N_WINDOWS = 5


def load_continuous_segments():
    depth = pd.read_parquet(
        DEPTH_PATH,
        columns=["timestamp_ms", "first_update_id", "last_update_id"],
        engine="pyarrow",
    )

    updates = (
        depth[
            ["timestamp_ms", "first_update_id", "last_update_id"]
        ]
        .drop_duplicates()
        .sort_values(
            ["first_update_id", "last_update_id"]
        )
        .reset_index(drop=True)
    )

    first_ids = updates["first_update_id"].to_numpy(
        dtype=np.int64
    )

    last_ids = updates["last_update_id"].to_numpy(
        dtype=np.int64
    )

    gaps = first_ids[1:] != last_ids[:-1] + 1
    gap_positions = np.flatnonzero(gaps)

    starts = np.r_[0, gap_positions + 1]
    ends = np.r_[gap_positions, len(updates) - 1]

    segments = pd.DataFrame(
        {
            "start_row": starts,
            "end_row": ends,
        }
    )

    segments["start_update_id"] = updates.loc[
        segments["start_row"],
        "first_update_id",
    ].to_numpy()

    segments["end_update_id"] = updates.loc[
        segments["end_row"],
        "last_update_id",
    ].to_numpy()

    segments["start_timestamp_ms"] = updates.loc[
        segments["start_row"],
        "timestamp_ms",
    ].to_numpy()

    segments["end_timestamp_ms"] = updates.loc[
        segments["end_row"],
        "timestamp_ms",
    ].to_numpy()

    segments["start_time"] = pd.to_datetime(
        segments["start_timestamp_ms"],
        unit="ms",
        utc=True,
    )

    segments["end_time"] = pd.to_datetime(
        segments["end_timestamp_ms"],
        unit="ms",
        utc=True,
    )

    segments["duration_seconds"] = (
        segments["end_time"] - segments["start_time"]
    ).dt.total_seconds()

    segments["n_updates"] = (
        segments["end_row"]
        - segments["start_row"]
        + 1
    )

    return segments


def add_trade_counts(segments):
    trades = pd.read_parquet(
        TRADES_PATH,
        columns=["timestamp_ms"],
        engine="pyarrow",
    )

    trade_times = np.asarray(
        trades["timestamp_ms"],
        dtype=np.int64,
    ).copy()

    trade_times.sort()

    starts_ms = segments[
        "start_timestamp_ms"
    ].to_numpy(dtype=np.int64)

    ends_ms = segments[
        "end_timestamp_ms"
    ].to_numpy(dtype=np.int64)

    left = np.searchsorted(
        trade_times,
        starts_ms,
        side="left",
    )

    right = np.searchsorted(
        trade_times,
        ends_ms,
        side="right",
    )

    segments = segments.copy()

    segments["n_trades"] = (
        right - left
    ).astype(np.int64)

    segments["updates_per_second"] = (
        segments["n_updates"]
        / segments["duration_seconds"]
    )

    segments["trades_per_second"] = (
        segments["n_trades"]
        / segments["duration_seconds"]
    )

    return segments


def filter_candidates(segments):
    candidates = segments[
        (segments["duration_seconds"] >= MIN_DURATION_SECONDS)
        & (segments["n_trades"] >= MIN_TRADES)
        & (segments["n_updates"] >= MIN_DEPTH_UPDATES)
    ].copy()

    candidates["duration_error"] = (
        candidates["duration_seconds"]
        - TARGET_DURATION_SECONDS
    ).abs()

    return candidates


def select_diverse_windows(candidates):
    if len(candidates) < N_WINDOWS:
        raise RuntimeError(
            f"Only {len(candidates)} valid candidates found. "
            f"Need at least {N_WINDOWS}."
        )

    candidates = (
        candidates
        .sort_values("start_time")
        .reset_index(drop=True)
    )

    positions = np.linspace(
        0,
        len(candidates),
        N_WINDOWS + 1,
        dtype=int,
    )

    selected_rows = []

    for i in range(N_WINDOWS):
        start = positions[i]
        end = positions[i + 1]

        block = candidates.iloc[start:end]

        if block.empty:
            continue

        best = (
            block
            .sort_values(
                [
                    "duration_error",
                    "n_trades",
                    "n_updates",
                ],
                ascending=[
                    True,
                    False,
                    False,
                ],
            )
            .iloc[0]
        )

        selected_rows.append(best)

    selected = (
        pd.DataFrame(selected_rows)
        .reset_index(drop=True)
    )

    selected.insert(
        0,
        "window_id",
        [
            f"W{i}"
            for i in range(1, len(selected) + 1)
        ],
    )

    return selected


def main():
    print("Loading depth segments...")

    segments = load_continuous_segments()

    print(
        f"Total continuous segments: "
        f"{len(segments):,}"
    )

    segments = add_trade_counts(segments)

    candidates = filter_candidates(segments)

    print(
        f"Valid research candidates: "
        f"{len(candidates):,}"
    )

    if candidates.empty:
        raise RuntimeError(
            "No valid research windows found."
        )

    selected = select_diverse_windows(
        candidates
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    columns = [
        "window_id",
        "start_time",
        "end_time",
        "duration_seconds",
        "n_updates",
        "n_trades",
        "updates_per_second",
        "trades_per_second",
    ]

    print("\nSelected research windows:\n")

    print(
        selected[columns].to_string(
            index=False
        )
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()