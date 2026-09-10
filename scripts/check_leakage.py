import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.retrieval.index import VectorIndex


def main():
    print("==================================================")
    print(" 8. DATA LEAKAGE VERIFICATION TOOL")
    print("==================================================")

    # 1. Load retrieval index conversation IDs
    index_dir = Path(settings.paths.retrieval_index_dir)
    items_file = index_dir / "items.json"
    
    if not items_file.exists():
        print(f"Error: Retrieval index not found at {items_file}. Run scripts/build_retrieval_index.py first.")
        sys.exit(1)

    with open(items_file, "r", encoding="utf-8") as f:
        retrieval_items = json.load(f)
    
    retrieval_conv_ids = set(it["conversation_id"] for it in retrieval_items)
    print(f"Loaded {len(retrieval_conv_ids)} conversation IDs from retrieval index.")

    # 2. Load evaluation split conversation IDs
    eval_file = Path(settings.paths.processed_data_dir) / "eval_conversations.json"
    eval_conv_ids = set()
    if eval_file.exists():
        with open(eval_file, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        eval_conv_ids.update(c["conversation_id"] for c in eval_data)

    # Also load golden set conversation IDs
    golden_file = Path(settings.paths.golden_set)
    if golden_file.exists():
        with open(golden_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    eval_conv_ids.add(item.get("conversation_id", ""))

    print(f"Loaded {len(eval_conv_ids)} conversation IDs from evaluation sets.")

    # 3. Intersection check
    overlap = retrieval_conv_ids.intersection(eval_conv_ids)

    if overlap:
        print("\n" + "!" * 80)
        print(f"[FAIL] CRITICAL LEAKAGE DETECTED! {len(overlap)} conversation IDs overlap!")
        print(f"Overlapping IDs sample: {list(overlap)[:5]}")
        print("!" * 80)
        sys.exit(1)
    else:
        print("\n" + "=" * 80)
        print("PASS: No conversation leakage detected between retrieval index and evaluation sets.")
        print("=" * 80)
        sys.exit(0)


if __name__ == "__main__":
    main()
