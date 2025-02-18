
from interface import InterfaceService
from db.interface import QdrantFaceDatabase

from typing import List, Dict, Any
from fastapi.responses import JSONResponse
from schema.module import BaseModule

class RetrieveServiceV1(InterfaceService):
    def __init__(
        self, 
        qry_module: BaseModule,
        docs_db: QdrantFaceDatabase,
        chunk_db: QdrantFaceDatabase
    ):
        self.qry_module = qry_module
        self.docs_db = docs_db
        self.chunk_db = chunk_db

    
    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        text_responses = self.qry_module.tokenizer(
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
            outputs = outputs.values() # BxLx768
            ## onnx parse
            # content = outputs["embeddings"][:, 0].tolist()
            ## trt parse
            # content = outputs[:, 0].tolist() # Bx768
            content = content.reshape(len(texts), -1, 768)[:, 0].tolist()
            return JSONResponse(content=content)
        except Exception as e:
            return JSONResponse(content={"Error": "Inference failed with error: " + str(e)})

    def run(self, texts: List[str], **kwargs) -> List[List[float]]:
        embeds = self.embed(texts)
        self.chunk_db.search(...)
        return "Success"