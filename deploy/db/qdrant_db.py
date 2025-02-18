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
            logging.info(f"Collection {collection_name} already exists")
            return
        # resolve distance
        distance = DISTANCE_MAPPING.get(distance, models.Distance.COSINE)
        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=dimension, distance=distance
            ),
        )

    def insert(self, points: List[dict], collection_name: str) -> None:
        """ Insert points into collection """
        self._client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=point['id'],
                    vector=point['vector'],
                    payload=point['payload'],
                )
                for point in points
            ],
        )

    def search(self, vector: List[float], collection_name: str, top_k:int = 5):
        # Top-k passages
        return self._client.query_points(
            collection_name=collection_name,
            query_vector=vector,
            limit=top_k,
        )

    def get_document(self, doc_id: str, collection_name: str):
        """ Get document information using doc_id """
        return self._client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id", match=models.MatchValue(value=doc_id)
                    ),
                ],
            ),
        )