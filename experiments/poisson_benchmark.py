from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


INPUT_FILE = Path(
    "data/processed/trade_events.parquet"
)

OUTPUT_FILE = Path(
    "data/processed/poisson_benchmark.csv"
)

WINDOW_SIZES = [
    0.1,
    0.5,
    1.0,
    5.0,
    10.0,
]


def load_events():
    events = pd.read_parquet(
        INPUT_FILE,
        engine="pyarrow",
    )

    required = [
        "event_time_s",
        "event_index",
        "side",
    ]

    missing = [
        column
        for column in required
        if column not in events.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    events = (
        events
        .sort_values(
            ["event_time_s", "event_index"]
        )
        .reset_index(drop=True)
    )

    return events


def calculate_counts(
    times,
    duration,
    window_size,
):
    n_windows = int(
        np.floor(
            duration / window_size
        )
    )

    if n_windows < 2:
        raise RuntimeError(
            f"Window size {window_size}s "
            "is too large."
        )

    edges = (
        np.arange(
            n_windows + 1,
            dtype=float,
        )
        * window_size
    )

    counts, _ = np.histogram(
        times,
        bins=edges,
    )

    return counts


def analyze_counts(counts):
    mean_count = counts.mean()
    variance_count = counts.var(
        ddof=1
    )

    fano = (
        variance_count / mean_count
        if mean_count > 0
        else np.nan
    )

    poisson_chi2 = np.nan
    poisson_pvalue = np.nan

    if mean_count > 0:
        expected = np.full(
            len(counts),
            mean_count,
            dtype=float,
        )

        poisson_chi2, poisson_pvalue = (
            stats.chisquare(
                counts,
                f_exp=expected,
            )
        )

    return {
        "mean_count": mean_count,
        "variance_count": variance_count,
        "fano_factor": fano,
        "poisson_chi2": poisson_chi2,
        "poisson_pvalue": poisson_pvalue,
        "n_windows": len(counts),
    }


def analyze_interarrivals(events):
    times = events[
        "event_time_s"
    ].to_numpy(dtype=float)

    interarrivals = np.diff(
        times
    )

    interarrivals = (
        interarrivals[
            interarrivals >= 0
        ]
    )

    if len(interarrivals) == 0:
        raise RuntimeError(
            "No inter-arrival times found."
        )

    mean_dt = interarrivals.mean()

    rate_mle = (
        1.0 / mean_dt
        if mean_dt > 0
        else np.inf
    )

    positive_dt = interarrivals[
        interarrivals > 0
    ]

    exponential_ks = np.nan
    exponential_pvalue = np.nan

    if len(positive_dt) > 0:
        exponential_ks, exponential_pvalue = (
            stats.kstest(
                positive_dt,
                "expon",
                args=(
                    0,
                    positive_dt.mean(),
                ),
            )
        )

    return {
        "mean_interarrival_s": mean_dt,
        "median_interarrival_s": np.median(
            interarrivals
        ),
        "p01_interarrival_s": np.quantile(
            interarrivals,
            0.01,
        ),
        "p05_interarrival_s": np.quantile(
            interarrivals,
            0.05,
        ),
        "p95_interarrival_s": np.quantile(
            interarrivals,
            0.95,
        ),
        "p99_interarrival_s": np.quantile(
            interarrivals,
            0.99,
        ),
        "positive_interarrival_count": len(
            positive_dt
        ),
        "estimated_rate": rate_mle,
        "exponential_ks": exponential_ks,
        "exponential_pvalue": exponential_pvalue,
    }


def analyze_side_counts(
    events,
    window_size,
):
    results = []

    duration = (
        events["event_time_s"].iloc[-1]
        - events["event_time_s"].iloc[0]
    )

    for side in ["buy", "sell"]:
        side_times = events.loc[
            events["side"] == side,
            "event_time_s",
        ].to_numpy(dtype=float)

        counts = calculate_counts(
            side_times,
            duration,
            window_size,
        )

        statistics = analyze_counts(
            counts
        )

        statistics["side"] = side
        statistics["window_size_s"] = (
            window_size
        )

        results.append(
            statistics
        )

    return results


def main():
    events = load_events()

    start = events[
        "event_time_s"
    ].iloc[0]

    end = events[
        "event_time_s"
    ].iloc[-1]

    events = events.copy()

    events["relative_time_s"] = (
        events["event_time_s"]
        - start
    )

    duration = (
        end - start
    )

    total_rate = (
        len(events) / duration
    )

    print(
        f"Events: {len(events):,}"
    )

    print(
        f"Duration: {duration:.6f} s"
    )

    print(
        f"MLE event rate: "
        f"{total_rate:.6f} events/s"
    )

    interarrival_result = (
        analyze_interarrivals(events)
    )

    print()
    print(
        "Inter-arrival analysis:"
    )

    print(
        f"Mean: "
        f"{interarrival_result['mean_interarrival_s']:.9f} s"
    )

    print(
        f"Median: "
        f"{interarrival_result['median_interarrival_s']:.9f} s"
    )

    print(
        f"Estimated Poisson rate: "
        f"{interarrival_result['estimated_rate']:.6f}"
    )

    print(
        f"Positive inter-arrivals: "
        f"{interarrival_result['positive_interarrival_count']:,}"
    )

    print(
        f"Exponential KS statistic: "
        f"{interarrival_result['exponential_ks']}"
    )

    print(
        f"Exponential p-value: "
        f"{interarrival_result['exponential_pvalue']}"
    )

    results = []

    for window_size in WINDOW_SIZES:
        counts = calculate_counts(
            events[
                "relative_time_s"
            ].to_numpy(dtype=float),
            duration,
            window_size,
        )

        statistics = analyze_counts(
            counts
        )

        statistics["window_size_s"] = (
            window_size
        )

        statistics["side"] = "all"

        results.append(
            statistics
        )

        results.extend(
            analyze_side_counts(
                events.assign(
                    event_time_s=events[
                        "relative_time_s"
                    ]
                ),
                window_size,
            )
        )

    results = pd.DataFrame(
        results
    )

    results = results[
        [
            "window_size_s",
            "side",
            "n_windows",
            "mean_count",
            "variance_count",
            "fano_factor",
            "poisson_chi2",
            "poisson_pvalue",
        ]
    ]

    print()
    print(
        "Count-dispersion analysis:"
    )

    print(
        results.to_string(
            index=False
        )
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()