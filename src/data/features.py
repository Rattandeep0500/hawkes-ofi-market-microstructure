from pathlib import Path

import pandas as pd


DEPTH_PATH = Path("data/processed/research_depth.parquet")
OUTPUT_PATH = Path("data/processed/book_features.parquet")


def build_book_features(depth):
    rows = []

    for window_id, window in depth.groupby("window_id", sort=False):
        bids = {}
        asks = {}

        updates = (
            window[
                [
                    "timestamp_ms",
                    "first_update_id",
                    "last_update_id",
                    "side",
                    "price",
                    "quantity",
                ]
            ]
            .drop_duplicates()
            .sort_values(
                ["first_update_id", "last_update_id"]
            )
        )

        for (
            timestamp_ms,
            first_update_id,
            last_update_id,
            side,
            price,
            quantity,
        ), group in updates.groupby(
            [
                "timestamp_ms",
                "first_update_id",
                "last_update_id",
            ],
            sort=False,
        ):
            for _, row in group.iterrows():
                book = bids if row["side"] == "bid" else asks
                price = float(row["price"])
                quantity = float(row["quantity"])

                if quantity <= 0:
                    book.pop(price, None)
                else:
                    book[price] = quantity

            if not bids or not asks:
                continue

            best_bid = max(bids)
            best_ask = min(asks)

            bid_prices = sorted(
                bids.keys(),
                reverse=True,
            )

            ask_prices = sorted(
                asks.keys()
            )

            bid_prices = bid_prices[:10]
            ask_prices = ask_prices[:10]

            bid_quantity_1 = bids[best_bid]
            ask_quantity_1 = asks[best_ask]

            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2

            queue_imbalance = (
                bid_quantity_1 - ask_quantity_1
            ) / (
                bid_quantity_1 + ask_quantity_1
            )

            row_out = {
                "window_id": window_id,
                "timestamp_ms": timestamp_ms,
                "first_update_id": first_update_id,
                "last_update_id": last_update_id,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "mid_price": mid_price,
                "spread": spread,
                "bid_size_1": bid_quantity_1,
                "ask_size_1": ask_quantity_1,
                "queue_imbalance": queue_imbalance,
            }

            for level, price in enumerate(
                bid_prices,
                start=1,
            ):
                row_out[f"bid_price_{level}"] = price
                row_out[f"bid_size_{level}"] = bids[price]

            for level, price in enumerate(
                ask_prices,
                start=1,
            ):
                row_out[f"ask_price_{level}"] = price
                row_out[f"ask_size_{level}"] = asks[price]

            rows.append(row_out)

    if not rows:
        raise RuntimeError(
            "No valid book states were reconstructed."
        )

    return pd.DataFrame(rows)


def main():
    depth = pd.read_parquet(
        DEPTH_PATH,
        engine="pyarrow",
    )

    features = build_book_features(depth)

    features["timestamp"] = pd.to_datetime(
        features["timestamp_ms"],
        unit="ms",
        utc=True,
    )

    features["log_mid_return"] = (
        features.groupby("window_id")["mid_price"]
        .transform(lambda x: (x / x.shift(1)).apply(
            lambda y: None if pd.isna(y) else pd.NA
        ))
    )

    features = features.drop(
        columns=["log_mid_return"]
    )

    features["mid_return"] = (
        features.groupby("window_id")["mid_price"]
        .pct_change()
    )

    features["spread_bps"] = (
        features["spread"]
        / features["mid_price"]
        * 10_000
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_parquet(
        OUTPUT_PATH,
        engine="pyarrow",
        index=False,
    )

    print(
        f"Book states: {len(features):,}"
    )

    print(
        f"Windows: {features['window_id'].nunique()}"
    )

    print(
        "\nSummary:"
    )

    print(
        features[
            [
                "window_id",
                "mid_price",
                "spread",
                "spread_bps",
                "queue_imbalance",
            ]
        ]
        .groupby("window_id")
        .agg(
            states=("mid_price", "size"),
            mean_mid=("mid_price", "mean"),
            mean_spread=("spread", "mean"),
            mean_spread_bps=(
                "spread_bps",
                "mean",
            ),
            mean_qi=(
                "queue_imbalance",
                "mean",
            ),
        )
        .to_string()
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()