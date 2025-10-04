import os

import wrds
import polars as pl
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sqlalchemy import create_engine, text, Engine
import json
from datetime import datetime

from schema_definitions import CRSP_DSF_SCHEMA, OPTIONM_OPPRCD_SCHEMA


# Configuration class for better maintainability
class OptionsProcessingConfig:
    def __init__(self,
                 wrds_username: str,
                 date_range: Dict[str, str] = None,
                 output_dir: str = 'options_data_partitioned',
                 volume_percentile: float = 0.95,
                 batch_size: int = 50000):
        self.wrds_username = wrds_username
        self.date_range = date_range or {
            "from_date": "2023-01-01",
            "to_date": "2023-01-31"
        }
        self.output_dir = Path(output_dir)
        self.volume_percentile = volume_percentile
        self.batch_size = batch_size


def create_wrds_engine(wrds_username: str) -> Tuple[object, Engine]:
    """
    Create both WRDS connection and SQLAlchemy engine for optimal performance

    Returns:
        Tuple of (wrds_connection, sqlalchemy_engine)
    """
    print("🔌 Creating WRDS connections...")

    # WRDS connection for metadata queries
    wrds_conn = wrds.Connection(wrds_username=wrds_username)

    return wrds_conn, wrds_conn.engine


def get_ticker_secids(wrds_conn: object, ticker: str) -> Optional[List[str]]:
    """
    Step 1: Get all security IDs for a given ticker

    Args:
        wrds_conn: WRDS connection object
        ticker: Ticker symbol (e.g., 'SPY')

    Returns:
        List of security IDs or None if not found
    """
    print(f"📋 Fetching secids for {ticker}...")

    secid_query = """
                  SELECT DISTINCT secid
                  FROM optionm.secnmd
                  WHERE ticker = %(ticker)s;
                  """

    try:
        secid_results = wrds_conn.raw_sql(secid_query, params={'ticker': ticker})

        if secid_results.empty:
            print(f"❌ No secids found for {ticker}")
            return None

        unique_secids = secid_results['secid'].unique().tolist()
        print(f"✅ Found {len(unique_secids)} unique secids for {ticker}")

        # spy - (7571, 100155, 109820, 115101)
        # qqq - (107899)
        return unique_secids

    except Exception as e:
        print(f"❌ Error fetching secids for {ticker}: {str(e)}")
        return None


def stream_options_data(engine: Engine,
                        secids: List[str],
                        config: OptionsProcessingConfig) -> Optional[pl.DataFrame]:
    """
    Step 2: Stream options data using Polars with SQLAlchemy engine

    Args:
        engine: SQLAlchemy engine
        secids: List of security IDs
        config: Processing configuration

    Returns:
        Polars DataFrame with options data
    """
    secid_str = ','.join([str(s) for s in secids])

    options_query = f"""
    SELECT date, secid, symbol, cp_flag, exdate, strike_price, 
           best_bid, best_offer, volume, open_interest, 
           impl_volatility, delta, vega, theta, forward_price, 
           expiry_indicator
    FROM optionm.opprcd2023
    WHERE secid IN ({secid_str})
      AND date BETWEEN '{config.date_range["from_date"]}' AND '{config.date_range["to_date"]}'
      AND volume IS NOT NULL
      AND volume > 0
    ORDER BY date, strike_price
    """

    try:
        print(f"🔄 Streaming options data with batch size {config.batch_size:,}...")

        # Use Polars' native database streaming
        df = pl.read_database(
            query=options_query,
            connection=engine,
            batch_size=config.batch_size,
            schema_overrides=dict(OPTIONM_OPPRCD_SCHEMA),
            infer_schema_length=False
        )

        print(f"✅ Successfully streamed {df.shape[0]:,} records")
        return df

    except Exception as e:
        print(f"❌ Error streaming options data: {str(e)}")
        return None


def stream_stock_data(engine, ticker, config):
    """
    Stream stock price data for the given ticker

    :param engine:
    :param ticker:
    :param config:
    :return:
    """

    stock_query = f"""
        SELECT dsf.prc, dsf.vol, dsf.date
        FROM crsp.dsf AS dsf
                 JOIN crsp.msenames AS msenames
                      ON dsf.cusip = msenames.cusip
        WHERE msenames.ticker = '{ticker}';
    """

    try:
        print(f"🔄 Streaming stock data for {ticker}...")

        stock_df = pl.read_database(
            query=stock_query,
            connection=engine,
            batch_size=config.batch_size,
            schema_overrides=dict(CRSP_DSF_SCHEMA),
            infer_schema_length=False
        )

        print(f"✅ Successfully streamed {stock_df.shape[0]:,} stock records for {ticker}")
        return stock_df

    except Exception as e:
        print(f"❌ Error streaming stock data for {ticker}: {str(e)}")
        return None


def apply_transformations(df: pl.DataFrame,
                          config: OptionsProcessingConfig) -> Tuple[pl.DataFrame, float]:
    """
    Step 3: Apply data transformations and filtering

    Args:
        df: Raw options data
        config: Processing configuration

    Returns:
        Tuple of (filtered_dataframe, volume_threshold)
    """
    print(f"🔧 Applying transformations...")

    # Add date-based columns for partitioning
    df_transformed = df.with_columns([
        pl.col('date').dt.year().alias('year'),
        pl.col('date').dt.month().alias('month'),
        pl.col('date').dt.day().alias('day')
    ])

    # Calculate volume percentile threshold
    volume_threshold = df_transformed.select(
        pl.col('volume').quantile(config.volume_percentile)
    ).item()

    # Apply volume filter
    df_filtered = df_transformed.filter(pl.col('volume') <= volume_threshold)

    # Create target variable iv_30d (30-day future implied volatility)
    df_filtered = df_filtered.sort(['secid', 'date']).with_columns([
        pl.col('impl_volatility').shift(-30).over('secid').alias('iv_30d')
    ]).filter(pl.col('iv_30d').is_not_null())

    # create columns for price difference over fibonacci days
    df_filtered = create_fibonacci_price_differences_optimized(df_filtered, max_days=250)

    print(f"📊 Volume {config.volume_percentile * 100}th percentile: {volume_threshold:,.0f}")
    print(f"✅ After volume filtering: {df_filtered.shape[0]:,} records "
          f"({df_filtered.shape[0] / df_transformed.shape[0] * 100:.1f}% retained)")

    return df_filtered, volume_threshold


def write_partitioned_parquet(df: pl.DataFrame,
                              ticker: str,
                              config: OptionsProcessingConfig,
                              volume_threshold: float) -> Tuple[List[str], Dict]:
    """
    Step 4: Write partitioned parquet files by year

    Args:
        df: Filtered options data
        ticker: Ticker symbol
        config: Processing configuration
        volume_threshold: Volume filtering threshold used

    Returns:
        Tuple of (files_created, metadata)
    """
    print(f"💾 Writing partitioned parquet files...")

    # Create ticker-specific directory
    ticker_dir = config.output_dir / ticker.lower()
    ticker_dir.mkdir(parents=True, exist_ok=True)

    # Get unique years
    years = df.select(pl.col('year').unique()).to_series().to_list()
    print(f"📅 Years found: {sorted(years)}")

    files_created = []
    total_records_written = 0

    for year in sorted(years):
        year_data = df.filter(pl.col('year') == year)

        if year_data.height > 0:
            # Remove partitioning columns before writing
            year_data_clean = year_data.drop(['year', 'month', 'day'])

            output_file = ticker_dir / f"{ticker.lower()}_{year}.parquet"

            # Write with optimal settings for Great Lakes cluster
            year_data_clean.write_parquet(
                output_file,
                compression='snappy',
                statistics=True,
                row_group_size=100000,  # Optimize for cluster I/O
                use_pyarrow=True
            )

            files_created.append(str(output_file))
            total_records_written += year_data.height

            print(f"  📁 {year}: {year_data.height:,} records → {output_file.name}")
        else:
            print(f"  ⚠️  {year}: No data after filtering")

    # Create comprehensive metadata
    metadata = {
        'ticker': ticker,
        'processing_config': {
            'date_range': config.date_range,
            'volume_percentile': config.volume_percentile,
            'batch_size': config.batch_size
        },
        'data_statistics': {
            'volume_threshold': float(volume_threshold),
            'total_records_raw': int(df.shape[0]),
            'total_records_written': total_records_written,
            'filter_retention_rate': float(total_records_written / df.shape[0]) if df.shape[0] > 0 else 0.0,
            'years_available': sorted(years)
        },
        'files_created': files_created,
        'processing_timestamp': datetime.now().isoformat(),
        'schema': {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
    }

    # Write metadata
    metadata_file = ticker_dir / f"{ticker.lower()}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    return files_created, metadata


def process_single_ticker(ticker: str,
                          wrds_conn: object,
                          engine: Engine,
                          config: OptionsProcessingConfig) -> Dict:
    """
    Process a single ticker through all steps

    Args:
        ticker: Ticker symbol
        wrds_conn: WRDS connection
        engine: SQLAlchemy engine
        config: Processing configuration

    Returns:
        Processing summary dictionary
    """
    print(f"\n{'=' * 60}")
    print(f"Processing {ticker}")
    print(f"{'=' * 60}")

    try:
        # Step 1: Get security IDs
        secids = get_ticker_secids(wrds_conn, ticker)
        if not secids:
            return {'status': 'failed', 'reason': 'No secids found'}

        # Step 2: Stream options data
        options_df = stream_options_data(engine, secids, config)
        if options_df is None:
            return {'status': 'failed', 'reason': 'No options data found'}

        # Step 2.1: Stream stock price data
        stock_df = stream_stock_data(engine, ticker, config)

        # Step 2.2: Merge stock data into options data
        if stock_df is not None and stock_df.height > 0:
            options_df = options_df.join(
                stock_df,
                left_on='date',
                right_on='date',
                how='left',
                suffix='_stock'
            )
            print(f"✅ Merged stock data: {options_df.shape[0]:,} records with stock info")

        # Step 3: Apply transformations
        df_filtered, volume_threshold = apply_transformations(options_df, config)

        # Step 4: Write partitioned files
        files_created, metadata = write_partitioned_parquet(
            df_filtered, ticker, config, volume_threshold
        )

        print(f"✅ {ticker} processing completed: {len(files_created)} files, "
              f"{metadata['data_statistics']['total_records_written']:,} records")

        return {
            'status': 'success',
            'files_created': len(files_created),
            'records_processed': metadata['data_statistics']['total_records_raw'],
            'records_written': metadata['data_statistics']['total_records_written'],
            'volume_threshold': volume_threshold,
            'metadata': metadata
        }

    except Exception as e:
        print(f"❌ Error processing {ticker}: {str(e)}")
        return {'status': 'error', 'error': str(e)}


def process_multiple_tickers_to_parquet(tickers: List[str],
                                        wrds_username: str = 'chriszhang08',
                                        **config_kwargs) -> Tuple[Dict, Path]:
    """
    Main orchestration function - process multiple tickers with streaming

    Args:
        tickers: List of ticker symbols
        wrds_username: WRDS username
        **config_kwargs: Additional configuration parameters

    Returns:
        Tuple of (processing_summary, output_directory)
    """
    # Initialize configuration
    config = OptionsProcessingConfig(wrds_username, **config_kwargs)
    config.output_dir.mkdir(exist_ok=True)

    # Create connections
    wrds_conn, engine = create_wrds_engine(wrds_username)

    processing_summary = {}

    try:
        # Process each ticker
        for ticker in tickers:
            result = process_single_ticker(ticker, wrds_conn, engine, config)
            processing_summary[ticker] = result

        # Print final summary
        print(f"\n{'=' * 60}")
        print("🎉 PROCESSING COMPLETED")
        print(f"{'=' * 60}")
        print(f"📂 Output directory: {config.output_dir}")

        for ticker, summary in processing_summary.items():
            if summary['status'] == 'success':
                print(f"✅ {ticker}: {summary['files_created']} files, "
                      f"{summary['records_written']:,} records")
            else:
                print(f"❌ {ticker}: {summary['status']} - "
                      f"{summary.get('reason', summary.get('error', 'Unknown error'))}")

    finally:
        # Clean up connections
        wrds_conn.close()
        engine.dispose()

    return processing_summary, config.output_dir


# Utility functions for working with the partitioned data
def load_ticker_year_data(ticker: str,
                          year: int,
                          data_dir: Optional[Path] = None) -> pl.DataFrame:
    """Load specific ticker/year combination"""
    if data_dir is None:
        data_dir = Path('options_data_partitioned')

    file_path = data_dir / ticker.lower() / str(year) / f"{ticker.lower()}_{year}_processed.parquet"

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    return pl.read_parquet(file_path)


def get_ticker_metadata(ticker: str,
                        data_dir: Optional[Path] = None) -> Dict:
    """Load ticker processing metadata"""
    if data_dir is None:
        data_dir = Path('options_data_partitioned')

    metadata_path = data_dir / ticker.lower() / f"{ticker.lower()}_metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        return json.load(f)


def generate_fibonacci_up_to(max_val):
    """Generate Fibonacci sequence up to max_val"""
    fib_sequence = []
    a, b = 1, 1  # Start with 1, 1 (skipping 0 for time series)

    while a <= max_val:
        fib_sequence.append(a)
        a, b = b, a + b

    return fib_sequence[1:]  # Skip the first 1 to avoid duplicate

def create_fibonacci_price_differences_optimized(df, max_days=220):
    """
    Optimized version that processes in chunks to avoid memory issues
    """

    # Generate Fibonacci sequence
    fib_days = generate_fibonacci_up_to(max_days)

    # Process in chunks to avoid creating too many columns at once
    chunk_size = 5  # Process 5 Fibonacci numbers at a time
    df_result = df

    for i in range(0, len(fib_days), chunk_size):
        chunk_days = fib_days[i:i + chunk_size]

        # Create expressions for this chunk
        chunk_expressions = [
            (pl.col('prc') -
             pl.col('prc').shift(days).over('secid')).alias(f'price_diff_{days}d')
            for days in chunk_days
        ]

        # Apply chunk transformations
        df_result = df_result.with_columns(chunk_expressions)

        print(f"Processed Fibonacci days: {chunk_days}")

    # Filter at the end
    max_fib_day = max(fib_days)
    df_result = df_result.filter(pl.col(f'price_diff_{max_fib_day}d').is_not_null())

    return df_result


# Usage example
if __name__ == "__main__":
    # Process ETF tickers with custom configuration
    tickers = ['SPY', 'QQQ', 'IWM', 'DIA']

    summary, output_dir = process_multiple_tickers_to_parquet(
        tickers=tickers,
        wrds_username='chriszhang08',
        date_range={
            "from_date": "2023-01-01",
            "to_date": "2023-01-31"
        },
        volume_percentile=0.95,
        batch_size=75000,  # Optimized for Great Lakes
        output_dir='options_data_partitioned'
    )

    # Example: Load specific data
    spy_2023 = load_ticker_year_data('SPY', 2023, output_dir)
    spy_metadata = get_ticker_metadata('SPY', output_dir)

    print(f"\nSPY 2023 data shape: {spy_2023.shape}")
    print(f"SPY volume threshold: {spy_metadata['data_statistics']['volume_threshold']:,.0f}")
