#!/usr/bin/env python3
"""
Stock Price Difference Calculator
Extracts stock price data from WRDS CRSP database and calculates
forward-looking price differences for Fibonacci day intervals.
"""

import argparse
import sys
from pathlib import Path
from typing import List
import urllib.parse

import polars as pl
import sqlalchemy as sa
from sqlalchemy import Engine


def create_wrds_engine_direct() -> Engine:
    """
    Create a SQLAlchemy engine directly for WRDS database without using wrds.Connection()
    This bypasses all interactive prompts and relies on .pgpass file or environment variables
    """
    # WRDS connection parameters
    WRDS_POSTGRES_HOST = "wrds-pgdata.wharton.upenn.edu"
    WRDS_POSTGRES_PORT = 9737
    WRDS_POSTGRES_DB = "wrds"

    # Try to get credentials from environment variables first
    import os
    username = os.environ.get('PGUSER', 'chriszhang08')
    password = os.environ.get('PGPASSWORD', '')

    # If no password from env vars, try to read from .pgpass file
    if not password:
        pgpass_file = Path.home() / '.pgpass'
        if pgpass_file.exists():
            try:
                with open(pgpass_file, 'r') as f:
                    for line in f:
                        if line.strip() and not line.startswith('#'):
                            parts = line.strip().split(':')
                            if (len(parts) >= 5 and
                                    parts[0] == WRDS_POSTGRES_HOST and
                                    int(parts[1]) == WRDS_POSTGRES_PORT and
                                    parts[2] == WRDS_POSTGRES_DB and
                                    parts[3] == username):
                                password = parts[4]
                                break
            except Exception as e:
                print(f"Warning: Could not read .pgpass file: {e}")

    if not password:
        raise ValueError("No password found in environment variables or .pgpass file")

    # URL encode the password to handle special characters
    password_encoded = urllib.parse.quote_plus(password)

    # Create PostgreSQL URI
    pguri = f"postgresql://{username}:{password_encoded}@{WRDS_POSTGRES_HOST}:{WRDS_POSTGRES_PORT}/{WRDS_POSTGRES_DB}"

    # Create SQLAlchemy engine with WRDS-specific settings
    connect_args = {
        "sslmode": "require",
        "application_name": "stock_price_extractor"
    }

    engine = sa.create_engine(
        pguri,
        isolation_level="AUTOCOMMIT",
        connect_args=connect_args
    )

    return engine


def generate_fibonacci_up_to(max_val: int) -> List[int]:
    """
    Generate Fibonacci sequence up to max_val

    Args:
        max_val: Maximum value for the Fibonacci sequence

    Returns:
        List of Fibonacci numbers (excluding first 1)
    """
    fib_sequence = []
    a, b = 1, 1

    while a <= max_val:
        fib_sequence.append(a)
        a, b = b, a + b

    return fib_sequence[1:]  # Skip first 1


def stream_stock_data(engine: Engine, ticker: str) -> pl.DataFrame | None:
    """
    Stream stock price data for the given ticker from CRSP database

    Args:
        engine: SQLAlchemy engine connected to WRDS
        ticker: Stock ticker symbol

    Returns:
        Polars DataFrame with stock price data or None if error
    """
    # Define CRSP DSF schema for Polars
    CRSP_DSF_SCHEMA = {
        'prc': pl.Float64,
        'vol': pl.Int64,
        'date': pl.Date
    }

    stock_query = f"""
        SELECT dsf.prc, dsf.vol, dsf.date
        FROM crsp.dsf AS dsf
        JOIN crsp.msenames AS msenames
            ON dsf.cusip = msenames.cusip
        WHERE msenames.ticker = '{ticker}'
        ORDER BY dsf.date
    """

    try:
        print(f"🔄 Streaming stock data for {ticker}...")

        stock_df = pl.read_database(
            query=stock_query,
            connection=engine,
            batch_size=100000,
            schema_overrides=CRSP_DSF_SCHEMA,
            infer_schema_length=False
        )

        print(f"✅ Successfully streamed {stock_df.shape[0]:,} stock records for {ticker}")
        return stock_df

    except Exception as e:
        print(f"❌ Error streaming stock data for {ticker}: {str(e)}")
        return None


def calculate_fibonacci_price_differences(stock_df: pl.DataFrame, max_days: int = 233) -> pl.DataFrame:
    """
    Calculate forward-looking price differences for daily stock price.
    Adds columns price_diff_{n}d for all Fibonacci day differences.

    Args:
        stock_df: DataFrame with columns ['date', 'prc', 'vol']
        max_days: Maximum number of days for Fibonacci sequence (default: 233)

    Returns:
        DataFrame with additional price difference columns
    """
    fib_days = generate_fibonacci_up_to(max_days)

    print(f"🔢 Calculating price differences for {len(fib_days)} Fibonacci days: {fib_days}")

    # Sort by date to ensure proper ordering
    stock_df = stock_df.sort('date')

    # Create expressions for all Fibonacci day differences
    # Using negative shift to look forward in time
    chunk_expressions = [
        (pl.col('prc') - pl.col('prc').shift(-n)).alias(f'price_diff_{n}d')
        for n in fib_days
    ]

    # Apply all transformations at once
    result_df = stock_df.with_columns(chunk_expressions)

    # Filter out rows where the maximum day difference is null
    # (these are at the end of the time series)
    max_fib_day = max(fib_days)
    result_df = result_df.filter(pl.col(f'price_diff_{max_fib_day}d').is_not_null())

    print(f"✅ Calculated price differences. Retained {result_df.shape[0]:,} records with complete data")

    return result_df


def save_to_parquet(df: pl.DataFrame, output_path: Path, ticker: str):
    """
    Save DataFrame to parquet file with optimized settings

    Args:
        df: DataFrame to save
        output_path: Directory path to save the file
        ticker: Stock ticker symbol for filename
    """
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{ticker.lower()}_price_differences.parquet"

    print(f"💾 Writing to {output_file}...")

    df.write_parquet(
        output_file,
        compression='zstd',
        statistics=True,
        row_group_size=50000,
        use_pyarrow=True
    )

    print(f"✅ Successfully saved {df.shape[0]:,} records to {output_file}")
    print(f"📊 Columns: {df.columns}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract stock prices from WRDS CRSP and calculate Fibonacci day price differences'
    )
    parser.add_argument(
        '--ticker',
        required=True,
        help='Stock ticker symbol (e.g., AAPL, MSFT)'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for parquet file'
    )
    parser.add_argument(
        '--max-days',
        type=int,
        default=233,
        help='Maximum number of days for Fibonacci sequence (default: 233)'
    )

    args = parser.parse_args()

    try:
        print(f"\n{'=' * 60}")
        print(f"📈 Stock Price Difference Calculator")
        print(f"   Ticker: {args.ticker}")
        print(f"   Max Days: {args.max_days}")
        print(f"{'=' * 60}\n")

        # Step 1: Create database connection
        print("🔌 Connecting to WRDS database...")
        engine = create_wrds_engine_direct()
        print("✅ Connected successfully\n")

        # Step 2: Stream stock data
        stock_df = stream_stock_data(engine, args.ticker)

        if stock_df is None or stock_df.height == 0:
            print(f"❌ No data found for ticker {args.ticker}")
            sys.exit(1)

        # Step 3: Calculate price differences
        result_df = calculate_fibonacci_price_differences(stock_df, args.max_days)

        # Step 4: Save to parquet
        output_path = Path(args.output_dir)
        save_to_parquet(result_df, output_path, args.ticker)

        print(f"\n{'=' * 60}")
        print(f"✅ Processing completed successfully!")
        print(f"{'=' * 60}\n")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)
    finally:
        if 'engine' in locals():
            engine.dispose(close=True)


if __name__ == "__main__":
    main()
