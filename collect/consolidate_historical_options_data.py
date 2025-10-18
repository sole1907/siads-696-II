import json
import datetime as dt
from pathlib import Path
from typing import List, Dict
import polars as pl


def consolidate_ticker_across_years(ticker: str, years: List[int],
                                    input_dir: Path, output_dir: Path) -> Dict:
    """
    Consolidate a single ticker's data across all years by reading all parquet files
    from multiple parquet files located in ticker/year subdirectories.
    """

    print(f"🔄 Consolidating {ticker} across {len(years)} years...")

    ticker_dfs = []
    years_found = []
    total_records = 0

    for year in years:
        year_dir = input_dir / ticker.lower() / str(year)
        if year_dir.exists() and year_dir.is_dir():
            # Find all parquet files in the directory for this year
            parquet_files = list(year_dir.glob("*.parquet"))
            if parquet_files:
                try:
                    # Read and concat all parquet files for the year
                    year_dfs = [pl.read_parquet(f) for f in parquet_files]
                    year_df = pl.concat(year_dfs, how="vertical_relaxed")
                    ticker_dfs.append(year_df)
                    years_found.append(year)
                    total_records += year_df.height
                    print(f"  📁 {year}: {len(parquet_files)} files, {year_df.height:,} records")
                except Exception as e:
                    print(f"  ❌ Error reading {year} parquet files: {str(e)}")
            else:
                print(f"  ⚠️  No parquet files found in {year_dir}")
        else:
            print(f"  ⚠️  Year directory {year_dir} does not exist")

    if not ticker_dfs:
        print(f"  ⚠️  No data files found for {ticker}")
        return {'status': 'no_data', 'years_found': []}

    # Concatenate all years dataframes vertically
    consolidated_df = pl.concat(ticker_dfs, how="vertical_relaxed")

    # Create consolidated output directory and write the consolidated parquet
    consolidated_ticker_dir = output_dir / ticker.lower()
    consolidated_ticker_dir.mkdir(parents=True, exist_ok=True)

    consolidated_file = consolidated_ticker_dir / f"{ticker.lower()}_historical_2005_2023.parquet"
    consolidated_df.write_parquet(
        consolidated_file,
        compression='zstd',
        statistics=True,
        row_group_size=100000,
        use_pyarrow=True
    )

    # Write consolidated metadata JSON
    metadata = {
        'ticker': ticker,
        'years_included': sorted(years_found),
        'total_years': len(years_found),
        'total_records': int(total_records),
        'date_range': {
            'start': min(years_found),
            'end': max(years_found)
        },
        'consolidation_timestamp': dt.datetime.now().isoformat(),
        'consolidated_file': str(consolidated_file)
    }
    metadata_file = consolidated_ticker_dir / f"{ticker.lower()}_consolidated_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ {ticker} consolidated: {total_records:,} records across {len(years_found)} years")

    return {
        'status': 'success',
        'years_found': years_found,
        'total_records': total_records,
        'consolidated_file': str(consolidated_file)
    }


if __name__ == "__main__":
    # Configuration
    tickers = ['QQQ']
    years = list(range(2005, 2024))
    input_dir = Path("./historical_options_data")
    output_dir = Path("./consolidated_historical_options_data")

    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    for ticker in tickers:
        result = consolidate_ticker_across_years(ticker, years, input_dir, output_dir)
        summary[ticker] = result

    # Write overall summary JSON
    summary_file = output_dir / "consolidation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"📄 Consolidation summary written to {summary_file}")