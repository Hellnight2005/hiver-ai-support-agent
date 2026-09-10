from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from src.intents.taxonomy import IntentTaxonomy, IntentDefinition


class IntentDiscoverer:
    def __init__(self, n_clusters: int = 8):
        self.n_clusters = n_clusters

    def discover_clusters(self, customer_messages: List[str]) -> List[Dict[str, Any]]:
        if not customer_messages:
            return []

        vectorizer = TfidfVectorizer(max_features=500, token_pattern=r"(?u)\b\w+\b")
        X = vectorizer.fit_transform(customer_messages)
        
        n_clusters = min(self.n_clusters, len(customer_messages))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X)

        feature_names = vectorizer.get_feature_names_out()
        clusters = []

        for cluster_id in range(n_clusters):
            # Top terms in cluster center
            center = kmeans.cluster_centers_[cluster_id]
            top_indices = center.argsort()[-6:][::-1]
            top_terms = [feature_names[i] for i in top_indices]

            # Representative messages
            member_indices = [i for i, label in enumerate(kmeans.labels_) if label == cluster_id]
            sample_msgs = [customer_messages[i] for i in member_indices[:3]]

            clusters.append({
                "cluster_id": cluster_id,
                "size": len(member_indices),
                "top_terms": top_terms,
                "sample_messages": sample_msgs
            })

        clusters.sort(key=lambda c: c["size"], reverse=True)
        return clusters
