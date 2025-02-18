import os
from typing import Optional, List, Any
from qdrant_client.http import models
from qdrant_client import QdrantClient


class QdrantFaceDatabase:
    def __init__(
        self,
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=os.getenv("QDRANT_PORT", 6333),
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._client = self.connect_client(host, port, url, api_key)

    def connect_client(self, host, port, url, api_key):
        if url is not None and api_key is not None:
            # cloud instance
            return QdrantClient(url=url, api_key=api_key)
        elif url is not None:
            # local instance with differ url
            return QdrantClient(url=url)
        else:
            return QdrantClient(host=host, port=port)
        
    def create_colection(self, collection_name="db", dimension=768, distance='cosine') -> None:
        if self._client.collection_exists(collection_name):
            self._client.delete_collection(collection_name)
        # resolve distance
        if distance == 'euclidean':
            distance = models.Distance.EUCLID
        elif distance == 'dot':
            distance = models.Distance.DOT
        elif distance == 'manhattan':
            distance = models.Distance.MANHATTAN
        else: 
            distance = models.Distance.COSINE
        # 
        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=dimension, distance=distance
            ),
        )

    def insert(self, collection_name: str, vectors: list, questions: list, answers: list) -> None:
        """
        self._client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=i,
                    vector=vectors[i],
                    payload={
                        "question": questions[i],
                        "answer": answers[i],
                    },
                )
                for i in range(len(vectors))
            ],
        )
        """
        ...

    def search(self, collection_name, vector, top_k=5, threshold=0.5):
        # Top-k = 5 means 5 results for 5 personal
        """
        return self._client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=top_k,
            score_threshold=threshold,
        )
        """
        ...

    def get_passage(self, collection_name, vector):
        """
        return self._client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=1,
            score_threshold=0.5,
        )
        """
        ...