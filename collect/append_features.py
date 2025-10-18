import polars as pl
from pathlib import Path


def add_iv_features_streaming(pl_df):
    """
    Add IV features using group_by aggregation instead of partition_by.
    Groups by date and expiration date for a single symbol (e.g., QQQ).
    More memory efficient for large datasets.
    """
    # Add helper columns for all three calculations to the main dataframe first
    df_with_helpers = pl_df.with_columns([
        ((pl.col('delta').abs() - 0.5).abs()).alias('delta_diff_atm'),
        (pl.col('delta') + 0.25).abs().alias('delta_diff_put25'),
        (pl.col('delta') - 0.25).abs().alias('delta_diff_call25')
    ])

    # Calculate ATM IV for each date-exdate group
    atm_iv = (
        df_with_helpers
        .sort(['date', 'exdate', 'delta_diff_atm'])
        .group_by(['date', 'exdate'], maintain_order=True)
        .agg([
            pl.col('impl_volatility').first().alias('atm_iv')
        ])
    )

    # Calculate 25-delta put IV
    put_iv = (
        df_with_helpers
        .filter(pl.col('cp_flag') == 'P')
        .sort(['date', 'exdate', 'delta_diff_put25'])
        .group_by(['date', 'exdate'], maintain_order=True)
        .agg([
            pl.col('impl_volatility').first().alias('iv_put25')
        ])
    )

    # Calculate 25-delta call IV
    call_iv = (
        df_with_helpers
        .filter(pl.col('cp_flag') == 'C')
        .sort(['date', 'exdate', 'delta_diff_call25'])
        .group_by(['date', 'exdate'], maintain_order=True)
        .agg([
            pl.col('impl_volatility').first().alias('iv_call25')
        ])
    )

    # Join all features
    result = (
        pl_df
        .join(atm_iv, on=['date', 'exdate'], how='left')
        .join(put_iv, on=['date', 'exdate'], how='left')
        .join(call_iv, on=['date', 'exdate'], how='left')
        .with_columns([
            (pl.col('iv_put25') - pl.col('iv_call25')).alias('skew'),
            ((pl.col('iv_put25') + pl.col('iv_call25')) / 2 - pl.col('atm_iv')).alias('curvature')
        ])
        .drop(['iv_put25', 'iv_call25'])
    )

    return result


def process_in_batches(input_path, output_dir, batch_size_months=6):
    """
    Process parquet file in smaller time-based batches.
    Reduced from 2 years to 6 months per batch.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    batch_dir = output_path / "batches"
    batch_dir.mkdir(exist_ok=True)

    print("Scanning parquet file...")
    lf = pl.scan_parquet(input_path)

    # Get date range
    date_stats = lf.select([
        pl.col('date').min().alias('min_date'),
        pl.col('date').max().alias('max_date')
    ]).collect()

    min_date = date_stats['min_date'][0]
    max_date = date_stats['max_date'][0]

    print(f"Date range: {min_date} to {max_date}")

    # Generate month ranges
    from dateutil.relativedelta import relativedelta

    current_date = min_date
    batch_files = []
    batch_num = 0

    while current_date <= max_date:
        end_date = min(current_date + relativedelta(months=batch_size_months), max_date)

        print(f"\nProcessing batch {batch_num}: {current_date} to {end_date}...")

        # Filter for this date range using lazy evaluation
        batch_lf = lf.filter(
            (pl.col('date') >= current_date) &
            (pl.col('date') < end_date)
        )

        # Collect and get row count
        print(f"  Loading data...")
        batch_df = batch_lf.collect(streaming=True)
        row_count = len(batch_df)

        print(f"  Processing {row_count:,} rows...")

        if row_count > 0:
            # Process batch with new method
            result_df = add_iv_features_streaming(batch_df)

            # Save batch result
            batch_file = batch_dir / f"batch_{batch_num:03d}_{current_date.strftime('%Y%m')}_{end_date.strftime('%Y%m')}.parquet"
            print(f"  Saving to {batch_file}...")
            result_df.write_parquet(batch_file)
            batch_files.append(batch_file)

            # Clear memory
            del batch_df, result_df
        else:
            print(f"  No data in this range, skipping...")

        current_date = end_date
        batch_num += 1


if __name__ == "__main__":
    input_file = "consolidated_historical_options_data/spy/spy_historical_2005_2023.parquet"
    output_directory = "consolidated_historical_options_data/spy"

    process_in_batches(
        input_path=input_file,
        output_dir=output_directory,
        batch_size_months=6  # Smaller batches: 6 months instead of 2 years
    )
