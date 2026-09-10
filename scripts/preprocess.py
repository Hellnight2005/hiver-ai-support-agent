import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import DataLoader
from src.config import settings


def main():
    print("==================================================")
    print(" 3. DATA PREPROCESSING")
    print("==================================================")

    loader = DataLoader()
    raw_path = loader.resolve_path()
    print(f"Reading raw data from: {raw_path}")

    tweets = loader.load_tweets()
    print(f"Extracted {len(tweets)} valid cleaned tweets.")

    out_dir = Path(settings.paths.processed_data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "cleaned_tweets.json"

    tweets_data = [t.model_dump() for t in tweets]
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(tweets_data, f, indent=2, ensure_ascii=False)

    print(f"Saved preprocessed tweets to: {out_file}")


if __name__ == "__main__":
    main()
