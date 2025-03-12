from typing import List, Dict, Any
from rtvserving.module.module import BaseModule
from rtvserving.services.interface import InterfaceService
from rtvserving.db.interface import InterfaceDatabase

import hashlib

class RetrievalServicesV1(InterfaceService):
    """Manager for Retrieval Services"""
    def __init__(
        self,
        query_module: BaseModule,
        context_module: BaseModule, 
        chunk_db: InterfaceDatabase,
    ):
        super().__init__()
        self.query_module = query_module
        self.context_module = context_module
        self.chunk_db = chunk_db

    def retrieve_chunks(self, query: str, chunker_id, **kwargs) -> List[Dict[str, Any]]:
        # cast List to query here
        query_embed = self.query_module.embed([query])
        # search for chunks
        retrieve_chunks = self.chunk_db.search(
            chunk_emb=query_embed[0], # List[List[float]] -> List[float] 
            chunker_id=chunker_id,
            top_k=kwargs.get('top_k', 5)
        )
        return retrieve_chunks

    def insert_chunks(self, chunks: List[dict], chunker_id: str) -> dict:
        if not chunks:
            return {"Error": "No chunks to insert!"}
        
        # format the chunks
        texts = [chunk['text'] for chunk in chunks]
        ids = [hashlib.md5(text.encode()).hexdigest() for text in texts]
        embeds = self.context_module.embed(texts)
        insert_dicts = {
            'ids': ids,
            'payloads': [chunk['payload'] for chunk in chunks],
            'vectors': embeds,
        }
        try:
            self.chunk_db.insert(insert_dicts, chunker_id)
        except Exception as e:
            return {"Error": f"Error inserting chunks: {e}"}
        return {"Success": "Chunks inserted!"}