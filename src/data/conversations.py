from typing import List, Dict, Set, Optional
from collections import defaultdict
from src.data.schemas import Tweet, Message, Conversation, BrandStats


class ConversationReconstructor:
    def __init__(self, tweets: List[Tweet]):
        self.tweets = tweets
        self.tweet_map: Dict[str, Tweet] = {t.tweet_id: t for t in tweets}
        self.parent_map: Dict[str, str] = {}
        self.children_map: Dict[str, List[str]] = defaultdict(list)
        self._build_graph()

    def _build_graph(self):
        for t in self.tweets:
            if t.in_response_to_tweet_id and t.in_response_to_tweet_id in self.tweet_map:
                self.parent_map[t.tweet_id] = t.in_response_to_tweet_id
                self.children_map[t.in_response_to_tweet_id].append(t.tweet_id)
            if t.response_tweet_id:
                resp_ids = [rid.strip() for rid in str(t.response_tweet_id).split(",") if rid.strip()]
                for rid in resp_ids:
                    if rid in self.tweet_map and rid not in self.children_map[t.tweet_id]:
                        self.children_map[t.tweet_id].append(rid)

    def find_root(self, tweet_id: str) -> str:
        curr = tweet_id
        visited = set()
        while curr in self.parent_map and curr not in visited:
            visited.add(curr)
            curr = self.parent_map[curr]
        return curr

    def reconstruct_all(self) -> List[Conversation]:
        # Group tweets by root tweet_id
        root_groups: Dict[str, Set[str]] = defaultdict(set)
        for t in self.tweets:
            root = self.find_root(t.tweet_id)
            root_groups[root].add(t.tweet_id)

        conversations: List[Conversation] = []

        for root_id, tweet_ids in root_groups.items():
            # Get tweets in group
            group_tweets = [self.tweet_map[tid] for tid in tweet_ids if tid in self.tweet_map]
            if not group_tweets:
                continue

            # Sort chronologically or by graph depth
            # To ensure order: root first, then child chain
            ordered_tweets = self._order_tweets(root_id, tweet_ids)

            # Identify brand and customer
            brand_names = [t.author_id for t in ordered_tweets if not t.inbound]
            if not brand_names:
                # If no agent, skip or mark unknown brand
                continue
            brand = brand_names[0]

            customer_ids = [t.author_id for t in ordered_tweets if t.inbound]
            customer_author_id = customer_ids[0] if customer_ids else "unknown_customer"

            messages = []
            for t in ordered_tweets:
                role = "agent" if not t.inbound else "customer"
                messages.append(Message(
                    tweet_id=t.tweet_id,
                    role=role,
                    author_id=t.author_id,
                    text=t.text,
                    timestamp=t.created_at
                ))

            if messages:
                conversations.append(Conversation(
                    conversation_id=f"conv_{root_id}",
                    brand=brand,
                    messages=messages,
                    customer_author_id=customer_author_id
                ))

        return conversations

    def _order_tweets(self, root_id: str, tweet_ids: Set[str]) -> List[Tweet]:
        # Simple BFS / DFS ordering starting from root
        ordered = []
        queue = [root_id]
        visited = set()
        while queue:
            curr = queue.pop(0)
            if curr in visited:
                continue
            visited.add(curr)
            if curr in self.tweet_map:
                ordered.append(self.tweet_map[curr])
                # Add children
                children = [cid for cid in self.children_map.get(curr, []) if cid in tweet_ids]
                queue.extend(children)
        # Append any unvisited in group
        for tid in tweet_ids:
            if tid not in visited and tid in self.tweet_map:
                ordered.append(self.tweet_map[tid])
        return ordered


def analyze_brand_suitability(conversations: List[Conversation]) -> List[BrandStats]:
    brand_convs: Dict[str, List[Conversation]] = defaultdict(list)
    for c in conversations:
        brand_convs[c.brand].append(c)

    stats_list = []
    for brand, convs in brand_convs.items():
        total_convs = len(convs)
        total_tweets = sum(c.turn_count for c in convs)
        customer_msgs = sum(1 for c in convs for m in c.messages if m.role == "customer")
        agent_msgs = sum(1 for c in convs for m in c.messages if m.role == "agent")
        avg_len = total_tweets / total_convs if total_convs > 0 else 0
        multi_turn = sum(1 for c in convs if c.is_multi_turn)

        # Keyword vocabulary size as proxy for intent diversity
        customer_texts = [m.text.lower() for c in convs for m in c.messages if m.role == "customer"]
        vocab = set(" ".join(customer_texts).split())
        intent_diversity = min(10.0, len(vocab) / 50.0)

        # Suitability score calculation
        suitability = (
            (total_convs * 0.4) +
            (multi_turn * 0.3) +
            (avg_len * 10.0) +
            (intent_diversity * 5.0)
        )

        stats_list.append(BrandStats(
            brand_name=brand,
            total_tweets=total_tweets,
            customer_messages=customer_msgs,
            agent_messages=agent_msgs,
            total_conversations=total_convs,
            avg_conversation_length=round(avg_len, 2),
            multi_turn_conversations=multi_turn,
            intent_diversity_score=round(intent_diversity, 2),
            suitability_score=round(suitability, 2)
        ))

    stats_list.sort(key=lambda s: s.suitability_score, reverse=True)
    return stats_list
