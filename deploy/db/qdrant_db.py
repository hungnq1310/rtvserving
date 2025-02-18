import os
import logging

from typing import Optional, List, Any
from qdrant_client.http import models
from qdrant_client import QdrantClient

from deploy.db.interface import InterfaceDatabase

DISTANCE_MAPPING = {
    'euclidean': models.Distance.EUCLID,
    'dot': models.Distance.DOT, 
    'manhattan': models.Distance.MANHATTAN,
    'cosine': models.Distance.COSINE
}


class QdrantFaceDatabase(InterfaceDatabase):
    """ 
    Vector database using Qdrant for storing and searching chunks and documents.
    """
    def __init__(
        self,
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=os.getenv("QDRANT_PORT", 6333),
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._client = self.connect_client(host, port, url, api_key)

    def connect_client(self, url, api_key):
        if url is not None and api_key is not None:
            # cloud instance
            return QdrantClient(url=url, api_key=api_key)
        elif url is not None:
            # local instance with differ url
            return QdrantClient(url=url)
        else:
            logging.error("None client connection")
            return None
        
    def create_colection(self, collection_name="db", dimension=768, distance='cosine') -> None:
        if self._client.collection_exists(collection_name):
            self._client.delete_collection(collection_name)
        # resolve distance
        distance = DISTANCE_MAPPING.get(distance, models.Distance.COSINE)
        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=dimension, distance=distance
            ),
        )

    def insert(self, points: list) -> None:
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