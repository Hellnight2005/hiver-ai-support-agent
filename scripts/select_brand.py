import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import DataLoader
from src.data.conversations import ConversationReconstructor, analyze_brand_suitability
from src.config import settings


def main():
    print("==================================================")
    print(" 2. BRAND SELECTION ANALYSIS")
    print("==================================================")

    loader = DataLoader()
    try:
        data_path = loader.resolve_path()
        print(f"Analyzing dataset: {data_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("Loading tweets...")
    tweets = loader.load_tweets()
    print(f"Loaded {len(tweets)} tweets.")

    print("Reconstructing conversation trees...")
    reconstructor = ConversationReconstructor(tweets)
    conversations = reconstructor.reconstruct_all()
    print(f"Reconstructed {len(conversations)} conversations.")

    print("\nCalculating brand suitability metrics...")
    stats_list = analyze_brand_suitability(conversations)

    print("\n" + "=" * 90)
    print(f"{'Brand Name':<20} | {'Convs':<7} | {'Tweets':<7} | {'Avg Len':<8} | {'MultiTurn':<10} | {'Suitability':<12}")
    print("=" * 90)
    for s in stats_list:
        print(f"{s.brand_name:<20} | {s.total_conversations:<7} | {s.total_tweets:<7} | {s.avg_conversation_length:<8.2f} | {s.multi_turn_conversations:<10} | {s.suitability_score:<12.2f}")
    print("=" * 90)

    if stats_list:
        best_brand = stats_list[0].brand_name
        print(f"\n[RECOMMENDATION] Recommended Brand with Strongest Data Quality: {best_brand}")
        print(f"To configure this brand, ensure BRAND_NAME={best_brand} in .env or config.yaml.")
    else:
        print("\nNo brand data found.")


if __name__ == "__main__":
    main()
