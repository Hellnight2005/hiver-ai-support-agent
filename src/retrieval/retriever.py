from typing import List, Dict, Any, Optional
from src.retrieval.index import VectorIndex, RetrievalItem
from src.config import settings


class HistoricalRetriever:
    def __init__(self, index: Optional[VectorIndex] = None):
        self.index = index or VectorIndex.load(directory=settings.paths.retrieval_index_dir)

    def retrieve(
        self,
        customer_message: str,
        top_k: int = 5,
        exclude_conv_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        return self.index.search(
            query=customer_message,
            top_k=top_k,
            exclude_conv_ids=exclude_conv_ids
        )
