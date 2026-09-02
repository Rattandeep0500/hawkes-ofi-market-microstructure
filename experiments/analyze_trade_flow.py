import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path(
    "data/live/btc_usdt_research_capture.jsonl"
)

OUTPUT_FILE = Path(
    "data/processed/trade_flow_summary.csv"
)


def load_trades():
    rows = []

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            record = json.loads(line)

            if record["type"] != "trade":
                continue

            event = record["data"]

            rows.append(
                {
                    "event_time_ms": int(event["T"]),
                    "trade_id": int(event["t"]),
                    "price": float(event["p"]),
                    "quantity": float(event["q"]),
                    "buyer_maker": bool(event["m"]),
                    "received_at_ms": int(
                        record["received_at_ms"]
                    ),
                }
            )

    if not rows:
        raise RuntimeError(
            "No trade events found."
        )

    trades = pd.DataFrame(rows)

    trades = (
        trades
        .sort_values(
            ["event_time_ms", "trade_id"]
        )
        .drop_duplicates(
            subset="trade_id"
        )
        .reset_index(drop=True)
    )

    return trades


def classify_trades(trades):
    result = trades.copy()

    result["side"] = np.where(
        result["buyer_maker"],
        "sell",
        "buy",
    )

    result["signed_quantity"] = np.where(
        result["side"] == "buy",
        result["quantity"],
        -result["quantity"],
    )

    result["notional"] = (
        result["price"]
        * result["quantity"]
    )

    result["signed_notional"] = np.where(
        result["side"] == "buy",
        result["notional"],
        -result["notional"],
    )

    result["interarrival_ms"] = (
        result["event_time_ms"].diff()
    )

    return result


def build_summary(trades):
    interarrival = (
        trades["interarrival_ms"]
        .dropna()
    )

    total_trades = len(trades)

    buy_trades = int(
        (trades["side"] == "buy").sum()
    )

    sell_trades = int(
        (trades["side"] == "sell").sum()
    )

    buy_quantity = trades.loc[
        trades["side"] == "buy",
        "quantity",
    ].sum()

    sell_quantity = trades.loc[
        trades["side"] == "sell",
        "quantity",
    ].sum()

    buy_notional = trades.loc[
        trades["side"] == "buy",
        "notional",
    ].sum()

    sell_notional = trades.loc[
        trades["side"] == "sell",
        "notional",
    ].sum()

    signed_quantity = trades[
        "signed_quantity"
    ].sum()

    signed_notional = trades[
        "signed_notional"
    ].sum()

    summary = {
        "trade_count": total_trades,
        "buy_count": buy_trades,
        "sell_count": sell_trades,
        "buy_fraction": buy_trades / total_trades,
        "sell_fraction": sell_trades / total_trades,
        "total_quantity": trades["quantity"].sum(),
        "buy_quantity": buy_quantity,
        "sell_quantity": sell_quantity,
        "signed_quantity": signed_quantity,
        "total_notional": trades["notional"].sum(),
        "buy_notional": buy_notional,
        "sell_notional": sell_notional,
        "signed_notional": signed_notional,
        "mean_trade_quantity": trades["quantity"].mean(),
        "median_trade_quantity": trades["quantity"].median(),
        "mean_trade_notional": trades["notional"].mean(),
        "median_trade_notional": trades["notional"].median(),
        "mean_interarrival_ms": interarrival.mean(),
        "median_interarrival_ms": interarrival.median(),
        "p01_interarrival_ms": interarrival.quantile(0.01),
        "p05_interarrival_ms": interarrival.quantile(0.05),
        "p25_interarrival_ms": interarrival.quantile(0.25),
        "p50_interarrival_ms": interarrival.quantile(0.50),
        "p75_interarrival_ms": interarrival.quantile(0.75),
        "p95_interarrival_ms": interarrival.quantile(0.95),
        "p99_interarrival_ms": interarrival.quantile(0.99),
        "max_interarrival_ms": interarrival.max(),
    }

    return pd.DataFrame([summary])


def print_summary(trades, summary):
    row = summary.iloc[0]

    print(
        f"Trade events: {len(trades):,}"
    )

    print()

    print(
        "Aggressor side:"
    )

    print(
        trades["side"]
        .value_counts()
        .to_string()
    )

    print()

    print(
        f"Buy fraction: "
        f"{row['buy_fraction']:.6f}"
    )

    print(
        f"Sell fraction: "
        f"{row['sell_fraction']:.6f}"
    )

    print()

    print(
        "Quantity:"
    )

    print(
        f"Total: "
        f"{row['total_quantity']:.8f}"
    )

    print(
        f"Buy: "
        f"{row['buy_quantity']:.8f}"
    )

    print(
        f"Sell: "
        f"{row['sell_quantity']:.8f}"
    )

    print(
        f"Signed: "
        f"{row['signed_quantity']:.8f}"
    )

    print()

    print(
        "Notional:"
    )

    print(
        f"Total: "
        f"{row['total_notional']:.2f}"
    )

    print(
        f"Buy: "
        f"{row['buy_notional']:.2f}"
    )

    print(
        f"Sell: "
        f"{row['sell_notional']:.2f}"
    )

    print(
        f"Signed: "
        f"{row['signed_notional']:.2f}"
    )

    print()

    print(
        "Inter-arrival time (ms):"
    )

    print(
        f"Mean: "
        f"{row['mean_interarrival_ms']:.6f}"
    )

    print(
        f"Median: "
        f"{row['median_interarrival_ms']:.6f}"
    )

    print(
        f"1%: "
        f"{row['p01_interarrival_ms']:.6f}"
    )

    print(
        f"5%: "
        f"{row['p05_interarrival_ms']:.6f}"
    )

    print(
        f"25%: "
        f"{row['p25_interarrival_ms']:.6f}"
    )

    print(
        f"50%: "
        f"{row['p50_interarrival_ms']:.6f}"
    )

    print(
        f"75%: "
        f"{row['p75_interarrival_ms']:.6f}"
    )

    print(
        f"95%: "
        f"{row['p95_interarrival_ms']:.6f}"
    )

    print(
        f"99%: "
        f"{row['p99_interarrival_ms']:.6f}"
    )

    print(
        f"Max: "
        f"{row['max_interarrival_ms']:.6f}"
    )


def main():
    trades = load_trades()

    trades = classify_trades(
        trades
    )

    summary = build_summary(
        trades
    )

    print_summary(
        trades,
        summary,
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()