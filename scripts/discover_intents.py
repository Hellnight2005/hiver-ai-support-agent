import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import DataLoader
from src.data.conversations import ConversationReconstructor
from src.intents.discovery import IntentDiscoverer
from src.config import settings


def main():
    print("==================================================")
    print(" 5. INTENT DISCOVERY PIPELINE")
    print("==================================================")

    loader = DataLoader()
    tweets = loader.load_tweets()
    
    reconstructor = ConversationReconstructor(tweets)
    conversations = reconstructor.reconstruct_all()

    brand = settings.brand.default_brand
    brand_convs = [c for c in conversations if c.brand.lower() == brand.lower()]
    if not brand_convs:
        brand_convs = conversations

    customer_msgs = [c.initial_customer_message for c in brand_convs if c.initial_customer_message]
    print(f"Extracted {len(customer_msgs)} customer messages for brand '{brand}'.")

    print("\nClustering messages to discover candidate intent themes...")
    discoverer = IntentDiscoverer(n_clusters=8)
    clusters = discoverer.discover_clusters(customer_msgs)

    print("\n--- Discovered Clusters ---")
    for c in clusters:
        print(f"\nCluster #{c['cluster_id']} (Count: {c['size']})")
        print(f"  Top Keywords: {', '.join(c['top_terms'])}")
        print("  Sample Messages:")
        for msg in c['sample_messages']:
            print(f"    - \"{msg}\"")

    print(f"\nFrozen taxonomy is maintained at: {settings.paths.intent_taxonomy}")


if __name__ == "__main__":
    main()
