
from interface import InterfaceService
from db.interface import QdrantFaceDatabase

from typing import List, Dict, Any
from fastapi.responses import JSONResponse

from module.module import BaseModule

class RetrievalServicesV1(InterfaceService):
    """Manager for Retrieval Services"""
    def __init__(
        self,
        query_module: BaseModule,
        context_module: BaseModule, 
        database_manager,
    ):
        super().__init__()
        self.query_module = query_module
        self.context_module = context_module
        self.database_manager = database_manager

    def check_user(self, token: str):
        return self.database_manager.check_user(token)
    
    def post_docs(self, docs: List[Any]):
        chunks = ...
        texts = [doc['text'] for doc in docs]
        payload = [doc['payload'] for doc in docs]
        embeds = self.context_module.embed(chunks)
        self.database_manager.upload_docs(texts, embeds, payload)

    def retrieve(self, texts: List[str], token: str) -> List[Dict[str, Any]]:
        if not self.check_user(token):
            return JSONResponse(content={"Error": "User not authenticated!"})
        query_embed = self.query_module.embed(texts)
        retrieve_chunks = self.database_manager.search(query_embed)
        return retrieve_chunks
