import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.schemas import Conversation
from src.retrieval.index import VectorIndex, RetrievalItem
from src.config import settings


def main():
    print("==================================================")
    print(" 6. BUILD RETRIEVAL INDEX (DEV SET ONLY)")
    print("==================================================")

    dev_file = Path(settings.paths.processed_data_dir) / "dev_conversations.json"
    if not dev_file.exists():
        print(f"Error: {dev_file} not found. Please run scripts/build_conversations.py first.")
        sys.exit(1)

    print(f"Loading dev conversations from: {dev_file}")
    with open(dev_file, "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    conversations = [Conversation(**d) for d in dev_data]
    print(f"Loaded {len(conversations)} dev conversations.")

    items = []
    for conv in conversations:
        cust_msg = conv.initial_customer_message
        agent_reply = conv.first_agent_response
        if cust_msg and agent_reply:
            items.append(RetrievalItem(
                conversation_id=conv.conversation_id,
                customer_message=cust_msg,
                agent_response=agent_reply,
                intent="unknown_other",
                brand=conv.brand
            ))

    print(f"Extracted {len(items)} historical resolution pairs.")
    print("Generating vector embeddings and building FAISS/Vector index...")

    index = VectorIndex()
    index.build_index(items)
    
    out_dir = settings.paths.retrieval_index_dir
    index.save(directory=out_dir)

    print(f"Vector index successfully built and saved to: {out_dir}")


if __name__ == "__main__":
    main()
