import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import DataLoader
from src.config import settings


def main():
    print("==================================================")
    print(" 1. DATASET INSPECTION TOOL")
    print("==================================================")
    
    loader = DataLoader()
    try:
        data_file = loader.resolve_path()
        print(f"Loading dataset from: {data_file}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    df, stats = loader.inspect_dataset()

    print("\n--- Summary Statistics ---")
    print(f"Row Count:            {stats.total_rows}")
    print(f"Columns ({len(stats.columns)}):     {', '.join(stats.columns)}")
    print(f"Unique Brands:        {stats.unique_brands}")
    print(f"Unique Authors:       {stats.unique_authors}")
    print(f"Date Range:           {stats.date_range[0] if stats.date_range else 'N/A'} to {stats.date_range[1] if stats.date_range else 'N/A'}")
    print(f"Approx Conversations: {stats.unique_conversations}")

    print("\n--- Missing Values Per Column ---")
    for col, count in stats.missing_values.items():
        pct = (count / stats.total_rows) * 100 if stats.total_rows > 0 else 0
        print(f"  - {col:25s}: {count:8d} ({pct:5.1f}%)")

    print("\n--- Sample Records (First 3) ---")
    sample_records = df.head(3).to_dict(orient="records")
    print(json.dumps(sample_records, indent=2, default=str))

    print("\nInspection complete!")


if __name__ == "__main__":
    main()
