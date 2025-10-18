import argparse
import json
import sys

import polars as pl
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from typing import List, Dict, Union

from polars import DataFrame
from sqlalchemy import Engine

from schema_definitions import OPTIONM_OPPRCD_SCHEMA

import sqlalchemy as sa
import urllib.parse
import os
from pathlib import Path


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
        "application_name": "python_direct_connection"
    }

    engine = sa.create_engine(
        pguri,
        isolation_level="AUTOCOMMIT",
        connect_args=connect_args
    )

    return engine


def parallel_data_chunks_processing(df: pl.DataFrame, n_chunks: int) -> pl.DataFrame:
    """
    Process DataFrame in parallel chunks for compute-intensive operations
    """

    # Split DataFrame into chunks
    chunk_size = max(1, df.height // n_chunks)
    chunks = []

    for i in range(0, df.height, chunk_size):
        chunk = df.slice(i, chunk_size)
        chunks.append(chunk)

    print(f"🔄 Processing {len(chunks)} chunks in parallel...")

    # Process chunks in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=min(n_chunks, mp.cpu_count())) as executor:
        # Apply computationally intensive operations in parallel
        processed_chunks = list(executor.map(process_chunk_intensive, chunks))

    # Concatenate results
    result_df = pl.concat(processed_chunks)
    print(f"✅ Parallel processing completed: {result_df.height} total records")

    return result_df


def stream_options_data(engine: Engine, secids: List[str], year: int) -> DataFrame | None:
    """
    Step 2: Stream options data using Polars with SQLAlchemy engine

    Args:
        engine: SQLAlchemy engine
        secids: List of security IDs
        date_range: Processing configuration

    Returns:
        Polars DataFrame with options data
    """
    secid_str = ','.join([str(s) for s in secids])

    options_query = f"""
    SELECT date, secid, symbol, cp_flag, exdate, strike_price, 
           best_bid, best_offer, volume, open_interest, 
           impl_volatility, delta, vega, theta, forward_price, 
           expiry_indicator
        FROM optionm.opprcd{year}
        WHERE secid IN ({secid_str})
          AND date BETWEEN '{year}-01-01' AND '{year}-12-31'
          AND volume IS NOT NULL
          AND volume > 0
        ORDER BY date, secid, strike_price
        """

    try:
        # Use Polars' native database streaming
        df = pl.read_database(
            query=options_query,
            connection=engine,
            batch_size=100000,
            schema_overrides=dict(OPTIONM_OPPRCD_SCHEMA),
            infer_schema_length=False
        )

        print(f"✅ Successfully streamed {df.shape[0]:,} records")
        return df

    except Exception as e:
        print(f"❌ Error streaming options data: {str(e)}")
        return None


def process_chunk_intensive(chunk: pl.DataFrame) -> pl.DataFrame:
    """
    Apply compute-intensive operations to a data chunk
    This runs in a separate process to utilize multiple CPU cores
    """
    # Apply transformations that benefit from parallelization
    processed = chunk.with_columns([
        # Date-based columns
        pl.col('date').dt.year().alias('year'),
        pl.col('date').dt.month().alias('month'),
        pl.col('date').dt.day().alias('day'),

        # Complex calculations that benefit from SIMD operations
        (pl.col('impl_volatility') * pl.col('delta')).alias('vol_delta_product'),
        (pl.col('prc') / (pl.col('strike_price') / 1000)).alias('moneyness'),

        # Rolling statistics (computationally intensive)
        pl.col('volume').rolling_mean(window_size=5).over('secid').alias('volume_ma5'),
        pl.col('impl_volatility').rolling_std(window_size=10).over('secid').alias('iv_rolling_std'),
    ])

    return processed


def generate_fibonacci_up_to(max_val):
    """Generate Fibonacci sequence up to max_val"""
    fib_sequence = []
    a, b = 1, 1

    while a <= max_val:
        fib_sequence.append(a)
        a, b = b, a + b

    return fib_sequence[1:]


def fetch_stock_data(file_path: Union[str, Path]) -> pl.DataFrame:
    """
    Load stock price data from a parquet file into a Polars DataFrame.

    Args:
        file_path: Path to the parquet file (string or Path object)

    Returns:
        Polars DataFrame containing the stock price data

    Raises:
        FileNotFoundError: If the parquet file doesn't exist
        polars.exceptions.ComputeError: If the file is corrupted or not a valid parquet file

    Example:
        >>> stock_df = fetch_stock_data("underlying_prc/spy_price_differences.parquet")
        >>> print(stock_df.shape)
        (5000, 15)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Expected a file path, but got directory: {file_path}")

    try:
        # Read parquet file into DataFrame
        df = pl.read_parquet(
            file_path,
            use_pyarrow=True,
            memory_map=True
        )

        print(f"✅ Loaded {df.shape[0]:,} rows and {df.shape[1]} columns from {file_path.name}")

        return df

    except Exception as e:
        raise pl.exceptions.ComputeError(
            f"Error reading parquet file {file_path}: {str(e)}"
        )


def process_ticker_optimized(ticker: str, year: int, config: Dict, output_dir: Path, n_cpus: int):
    """
    Process single ticker with full cluster optimization
    """

    print(f"\n{'=' * 60}")
    print(f"🚀 Processing {ticker} with {n_cpus} CPU cores")
    print(f"{'=' * 60}")

    # Create optimized WRDS connection
    db: Engine = create_wrds_engine_direct()

    try:
        # Step 1: Get security IDs (fast operation)
        secid_query = """
                      SELECT DISTINCT secid
                      FROM optionm.secnmd
                      WHERE ticker = %(ticker)s \
                      """

        secid_results = pd.read_sql_query(secid_query, db, params={'ticker': ticker})
        if secid_results.empty:
            print(f"❌ No secids found for {ticker}")
            return

        unique_secids = secid_results['secid'].unique().tolist()
        print(f"✅ Found {len(unique_secids)} secids for {ticker}")

        # Step 2: Stream options data with optimized query
        # Use Polars streaming with large batch size for cluster
        print(f"🔄 Streaming options data...")
        df = stream_options_data(db, unique_secids, year)

        # Step 2.1: Stream stock price data
        stock_df = fetch_stock_data(f'underlying_prc/{ticker}_price_differences.parquet')

        # Step 2.2: Merge stock data into options data
        if stock_df is not None and stock_df.height > 0:
            options_df = df.join(
                stock_df,
                left_on='date',
                right_on='date',
                how='left',
                suffix='_stock'
            )
            print(f"✅ Merged stock data: {options_df.shape[0]:,} records with stock info")

        print(f"✅ Loaded {options_df.shape[0]:,} records")

        # Step 3: Apply volume filtering
        volume_threshold = options_df.select(
            pl.col('volume').quantile(config['volume_percentile'])
        ).item()

        df_filtered = options_df.filter(pl.col('volume') <= volume_threshold)
        print(f"📊 After volume filtering: {df_filtered.shape[0]:,} records")

        # filter out records with null impl_volatility, delta, theta, vega
        df_filtered = df_filtered.filter(
            pl.col('impl_volatility').is_not_null() &
            pl.col('delta').is_not_null() &
            pl.col('theta').is_not_null() &
            pl.col('vega').is_not_null()
        )

        # Create target variable iv_30d (30-day future implied volatility)
        df_filtered = df_filtered.sort(['secid', 'date']).with_columns([
            pl.col('impl_volatility').shift(-30).over('secid').alias('iv_30d')
        ]).filter(pl.col('iv_30d').is_not_null())

        # Step 4: Parallel processing of complex transformations
        df_final = parallel_data_chunks_processing(df_filtered, n_cpus)

        # Step 7: Write to year-specific directory structure
        ticker_year_dir = output_dir / ticker.lower() / str(year)
        ticker_year_dir.mkdir(parents=True, exist_ok=True)

        # Remove temporary columns before writing
        df_clean = df_final.drop(['month', 'day'])

        output_file = ticker_year_dir / f"{ticker.lower()}_{year}_processed.parquet"

        # Write with year-specific optimizations
        df_clean.write_parquet(
            output_file,
            compression='zstd',
            statistics=True,
            row_group_size=50000,
            use_pyarrow=True
        )

        print(f"✅ {ticker} processing completed successfully")

    except Exception as e:
        print(f"❌ Error processing {ticker}: {str(e)}")
        raise
    finally:
        db.dispose(close=True)


def main():
    parser = argparse.ArgumentParser(description='Process single ticker in parallel')
    parser.add_argument('--ticker', required=True, help='Ticker symbol')
    parser.add_argument('--year', type=int, required=True, help='Year to process')
    parser.add_argument('--config-file', required=True, help='Configuration JSON file')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--cpus', type=int, default=1, help='Number of CPUs to use')

    args = parser.parse_args()

    try:
        # Load configuration
        with open(args.config_file, 'r') as f:
            config = json.load(f)

        # Process the ticker
        process_ticker_optimized(
            args.ticker,
            args.year,
            config,
            Path(args.output_dir),
            args.cpus
        )

    except Exception as e:
        print(f"\n❌ ERROR: An exception occurred while processing.\nReason: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
