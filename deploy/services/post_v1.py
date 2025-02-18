
from interface import InterfaceService
from db.interface import QdrantFaceDatabase

from typing import List, Dict, Any
from fastapi.responses import JSONResponse
from schema.module import BaseModule

class PostServiceV1(InterfaceService):
    def __init__(
        self, 
        ctx_module: BaseModule,
        docs_db: QdrantFaceDatabase,
        chunk_db: QdrantFaceDatabase
    ):
        self.ctx_module = ctx_module
        self.docs_db = docs_db
        self.chunk_db = chunk_db

    
    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        text_responses = self.ctx_module.tokenizer(
            texts, 
            padding='max_length', 
            truncation=True, 
            return_tensors="np"
        )
        try:
            outputs: Dict[Any] = self.model.run(data = [
                text_responses['input_ids'], 
                text_responses['attention_mask'], 
                text_responses['token_type_ids']
            ])
            outputs = outputs.values()[0] # BxLx768
            content = content.reshape(len(texts), -1, 768)[:, 0].tolist()
            return JSONResponse(content=content)
        except Exception as e:
            return JSONResponse(content={"Error": "Inference failed with error: " + str(e)})

    def run(self, texts: List[str], **kwargs) -> List[List[float]]:
        chunking = ...

        # texts is docs
        chunks = texts
        chunk_embeds = self.embed(chunks)
        self.docs_db.insert(chunk_embeds)
        return "Success"