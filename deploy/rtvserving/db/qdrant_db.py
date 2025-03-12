import os
import logging

from typing import Optional, List, Any
from qdrant_client.http import models
from qdrant_client import QdrantClient

from rtvserving.db.interface import InterfaceDatabase

DISTANCE_MAPPING = {
    'euclidean': models.Distance.EUCLID,
    'dot': models.Distance.DOT, 
    'manhattan': models.Distance.MANHATTAN,
    'cosine': models.Distance.COSINE
}


class QdrantChunksDB(InterfaceDatabase):
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
        self._client = self.connect_client(url, api_key)

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
        
    def create_colection(self, chunker_id="db", dimension=768, distance='cosine') -> None:
        if self._client.collection_exists(chunker_id):
            logging.info(f"Collection {chunker_id} already exists")
            return
        # resolve distance
        distance = DISTANCE_MAPPING.get(distance, models.Distance.COSINE)
        self._client.create_collection(
            collection_name=chunker_id,
            vectors_config=models.VectorParams(
                size=dimension, distance=distance
            ),
        )

    def insert(self, chunks: List[dict], chunker_id: str) -> None:
        """ Insert points into collection """
        self._client.upsert(
            collection_name=chunker_id,
            points=models.Batch(
                payloads=chunks['payloads'],
                vectors=chunks['vectors'],
            ),
            # ids of chunks are not provided, Qdrant Client will generate them automatically as random UUIDs.
        )

    def search(self, chunk_emb: List[float], chunker_id: str, top_k:int = 5):
        # Top-k passages
        return self._client.query_points(
            collection_name=chunker_id,
            query_vector=chunk_emb,
            limit=top_k,
        )

    def get_chunks_by_doc_id(self, doc_id: str, chunker_id: str):
        """ Get document information using doc_id """
        return self._client.scroll(
            collection_name=chunker_id,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id", match=models.MatchValue(value=doc_id)
                    ),
                ],
            ),
        )
    
    def delete(self, points_ids, **kwagrs):
        ...
    
    def update(self, points_ids, **kwagrs):
        ...