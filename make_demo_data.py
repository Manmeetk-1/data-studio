"""Generate a deliberately messy demo dataset, so the quality checks have something to find.

Usage:  python make_demo_data.py --rows 500000
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "data" / "demo_sales.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=200_000)
    args = ap.parse_args()
    n = args.rows

    random.seed(7)
    products = [f"SKU-{i:04d}" for i in range(1, 300)] + ["POST", "BANK CHARGES"]
    countries = ["Canada", "United Kingdom", "France", "Germany", "Spain", "Netherlands"]

    rows = {
        "order_id": [f"INV{100000 + i}" for i in range(n)],
        "order_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(
            [random.randint(0, 700) for _ in range(n)], unit="D"
        ),
        "customer_id": [
            None if random.random() < 0.24 else f"C{random.randint(1000, 9999)}"
            for _ in range(n)
        ],                                          # ~24% guest checkouts -> missing
        "country": [random.choice(countries) for _ in range(n)],
        "product_code": [random.choice(products) for _ in range(n)],
        "quantity": [
            -random.randint(1, 5) if random.random() < 0.03 else random.randint(1, 40)
            for _ in range(n)
        ],                                          # ~3% returns -> negatives
        "unit_price": [
            round(random.uniform(0.5, 90), 2) if random.random() > 0.002 else 9999.99
            for _ in range(n)
        ],                                          # rare absurd outliers
        "channel": ["web"] * n,                     # constant column
        "notes": ["" if random.random() < 0.4 else "ok" for _ in range(n)],  # blank strings
        "discount_pct": [str(round(random.uniform(0, 0.3), 2)) for _ in range(n)],  # numeric as text
    }

    df = pd.DataFrame(rows)
    df = pd.concat([df, df.head(1200)], ignore_index=True)   # duplicate rows

    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    size_mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT}  ({len(df):,} rows, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
