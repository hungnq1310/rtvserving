
from interface import InterfaceService
from db.interface import InterfaceDatabase

from typing import List, Dict, Any
from fastapi.responses import JSONResponse

from module.module import BaseModule

class RetrievalServicesV1(InterfaceService):
    """Manager for Retrieval Services"""
    def __init__(
        self,
        query_module: BaseModule,
        context_module: BaseModule, 
        user_db: InterfaceDatabase,
        doc_db: InterfaceDatabase,
        chunk_db: InterfaceDatabase,
        session_db: InterfaceDatabase
    ):
        super().__init__()
        self.query_module = query_module
        self.context_module = context_module
        self.user_db = user_db
        self.doc_db = doc_db
        self.chunk_db = chunk_db
        self.session_db = session_db

    def register_user(self, user_name, authen):
        ...
    
    def check_user(self, token: str):
        ...
    
    def post_docs(self, docs: List[Any]):
        chunks = ...
        texts = [doc['text'] for doc in docs]
        payload = [doc['payload'] for doc in docs]
        embeds = self.context_module.embed(chunks)
        ...

    def retrieve(self, texts: List[str], token: str) -> List[Dict[str, Any]]:
        if not self.check_user(token):
            return JSONResponse(content={"Error": "User not authenticated!"})
        query_embed = self.query_module.embed(texts)
        retrieve_chunks = self.chunk_db.search(query_embed)
        return retrieve_chunks

    def update_session_state(self, history: List[Any]):
        ...
    