import polars as pl


def combine_parquet_files(input_dir, output_file, sample_fraction=None, seed=42):
    """
    Combine all parquet files in a directory into a single parquet file.
    Optionally sample a fraction of the data for testing purposes.

    Args:
        input_dir: Directory containing parquet files
        output_file: Path for the combined output parquet file
        sample_fraction: Fraction of data to keep (e.g., 0.1 for 10%, None for all data)
        seed: Random seed for reproducible sampling
    """
    print(f"Scanning parquet files in {input_dir}...")
    combined = pl.scan_parquet(f"{input_dir}/*.parquet")

    # Sample if fraction is specified
    if sample_fraction is not None:
        print(f"Sampling {sample_fraction * 100:.1f}% of the data (seed={seed})...")
        # Collect to DataFrame, sample, then write
        df = combined.collect()
        df_sampled = df.sample(fraction=sample_fraction, seed=seed, shuffle=False)

        print(f"Original rows: {len(df):,}")
        print(f"Sampled rows: {len(df_sampled):,}")
        print(f"Writing sampled data to {output_file}...")

        df_sampled.write_parquet(output_file)
    else:
        # Write without sampling
        print(f"Writing combined file to {output_file}...")
        combined.sink_parquet(output_file)

    print(f"Done! Combined parquet saved to {output_file}")


# Usage examples
if __name__ == "__main__":
    # For local testing - keep only 10% of data (drop 90%)
    # combine_parquet_files(
    #     input_dir="consolidated_historical_options_data/batches",
    #     output_file="consolidated_historical_options_data/batches/combined_sample.parquet",
    #     sample_fraction=0.1,  # Keep 10%, drop 90%
    #     seed=42  # For reproducibility
    # )

    # For production - use all data
    combine_parquet_files(
        input_dir="consolidated_historical_options_data/batches/sampled/train",
        output_file="consolidated_historical_options_data/batches/train_dataset.parquet",
        sample_fraction=None  # No sampling
    )
