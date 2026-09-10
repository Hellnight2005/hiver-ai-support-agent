import json
from pathlib import Path
import pytest
from src.config import settings


def test_zero_conversation_leakage():
    items_file = Path(settings.paths.retrieval_index_dir) / "items.json"
    if not items_file.exists():
        pytest.skip("Retrieval index not yet built.")

    with open(items_file, "r", encoding="utf-8") as f:
        retrieval_items = json.load(f)
    
    retrieval_conv_ids = set(it["conversation_id"] for it in retrieval_items)

    eval_file = Path(settings.paths.processed_data_dir) / "eval_conversations.json"
    eval_conv_ids = set()
    if eval_file.exists():
        with open(eval_file, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        eval_conv_ids.update(c["conversation_id"] for c in eval_data)

    golden_file = Path(settings.paths.golden_set)
    if golden_file.exists():
        with open(golden_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    eval_conv_ids.add(item.get("conversation_id", ""))

    overlap = retrieval_conv_ids.intersection(eval_conv_ids)
    assert len(overlap) == 0, f"Critical Leakage Detected! Overlapping IDs: {overlap}"
