import sys
import json
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import DataLoader
from src.data.conversations import ConversationReconstructor
from src.config import settings


def main():
    print("==================================================")
    print(" 4. CONVERSATION RECONSTRUCTION & LEAK-FREE SPLIT")
    print("==================================================")

    brand = settings.brand.default_brand
    print(f"Target Brand: {brand}")

    loader = DataLoader()
    tweets = loader.load_tweets()
    
    reconstructor = ConversationReconstructor(tweets)
    all_conversations = reconstructor.reconstruct_all()

    # Filter for target brand
    brand_conversations = [c for c in all_conversations if c.brand.lower() == brand.lower()]
    if not brand_conversations:
        print(f"Warning: No conversations found for brand '{brand}'. Falling back to all conversations.")
        brand_conversations = all_conversations

    print(f"Total reconstructed conversations for '{brand}': {len(brand_conversations)}")

    # Deterministic conversation-level split
    seed = settings.splits.seed
    random.seed(seed)
    shuffled_convs = list(brand_conversations)
    random.shuffle(shuffled_convs)

    n_total = len(shuffled_convs)
    n_dev = int(n_total * settings.splits.dev_ratio)
    n_val = int(n_total * settings.splits.val_ratio)

    dev_convs = shuffled_convs[:n_dev]
    val_convs = shuffled_convs[n_dev:n_dev + n_val]
    eval_convs = shuffled_convs[n_dev + n_val:]

    # Fallback for small sample datasets so all splits have at least 1 item
    if len(shuffled_convs) >= 3 and (not val_convs or not eval_convs):
        dev_convs = shuffled_convs[:-2]
        val_convs = [shuffled_convs[-2]]
        eval_convs = [shuffled_convs[-1]]

    print(f"Splits summary:")
    print(f"  - Development / Retrieval Set (70%): {len(dev_convs)} conversations")
    print(f"  - Validation Set (15%):               {len(val_convs)} conversations")
    print(f"  - Evaluation Set (15%):               {len(eval_convs)} conversations")

    out_dir = Path(settings.paths.processed_data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "dev_conversations.json", "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in dev_convs], f, indent=2)

    with open(out_dir / "val_conversations.json", "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in val_convs], f, indent=2)

    with open(out_dir / "eval_conversations.json", "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in eval_convs], f, indent=2)

    print(f"Successfully saved split conversation datasets to: {out_dir}")


if __name__ == "__main__":
    main()
