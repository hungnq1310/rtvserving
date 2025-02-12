import os
from typing import List, Any
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from transformers import AutoTokenizer
from ray import serve
import ray
from dotenv import load_dotenv

from trism import TritonModel
from db.interface import QdrantFaceDatabase

import asyncio
# from utils.score_compute import compute_similarity

import numpy as np

def compute_similarity(q_reps, p_reps):
    if not isinstance(q_reps, np.ndarray):
        q_reps = np.array(q_reps)
    if not isinstance(p_reps, np.ndarray):
        p_reps = np.array(p_reps)
    return np.matmul(q_reps, p_reps.T)


load_dotenv()
# Parse environment variables
#
query_retriever_name    = os.getenv("QUERY_MODEL_NAME")
query_version = int(os.getenv("QUERY_MODEL_VERSION", ""))
query_batch_size    = int(os.getenv("BATCH_SIZE", 1))
#
ctx_retriever_name    = os.getenv("CTX_MODEL_NAME") 
ctx_version = int(os.getenv("CTX_MODEL_VERSION", ""))
ctx_batch_size    = int(os.getenv("BATCH_SIZE", 10))

#
url           = os.getenv("TRITON_URL", "localhost:6000")
protocol      = os.getenv("PROTOCOL", "HTTP")
verbose       = os.getenv("VERBOSE", "False").lower() in ("true", "1", "t")
async_set     = os.getenv("ASYNC_SET", "False").lower() in ("true", "1", "t")
#
grpc = protocol.lower() == "grpc"
#
use_rerank = os.getenv("USE_RERANK", "False").lower() in ("true", "1", "t")
# 
collection_name = os.getenv("QDRANT_COLLECTION_NAME", "retrieval")
top_k = int(os.getenv("TOP_K", 5))
threshold = float(os.getenv("THRESHOLD", 0.5))
QDRANT_DB     = os.getenv("QDRANT_DB", "")

print(f"Query model: {query_retriever_name}")
print(f"Context model: {ctx_retriever_name}")
print(f"URL: {url}")
print(f"Protocol: {protocol}")
print(f"Verbose: {verbose}")
print(f"Async set: {async_set}")
print(f"Use rerank: {use_rerank}")
print(f"QDRANT_DB: {QDRANT_DB}")


############
# FastAPI
############


app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

## Deploy model
@serve.deployment
class ModelEmbedding:
    def __init__(self, model_name, model_version):
        self.model, self.tokenizer = self._init_model_and_tokenizer(model_name, model_version)
        
    def _init_model_and_tokenizer(self, model_name, model_version):
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join("/models", model_name, str(model_version))
        )
        model = TritonModel(
            model=model_name,                 # Model name.
            version=model_version,            # Model version.
            url=url,                          # Triton Server URL.
            grpc=grpc                         # Use gRPC or Http.
        )
        # View metadata.
        for inp in model.inputs:
            print(f"name: {inp.name}, shape: {inp.shape}, datatype: {inp.dtype}\n")
        for out in model.outputs:
            print(f"name: {out.name}, shape: {out.shape}, datatype: {out.dtype}\n") 

        return model, tokenizer
    
    async def __call__(self, textRequest: List[str]):
        text_responses = self.tokenizer(
            textRequest, 
            padding='max_length', 
            truncation=True, 
            return_tensors="np"
        )
        try:
            start_time = time.time()
            outputs = self.model.run(data = [
                text_responses['input_ids'], 
                text_responses['attention_mask'], 
                text_responses['token_type_ids']
            ])
            end_time = time.time()
            print("Process time: ", end_time - start_time)

            ## onnx parse
            # content = outputs["embeddings"][:, 0].tolist()
            ## trt parse
            content = outputs['last_hidden_state']
            content = content.reshape(len(textRequest), -1, 768)[:, 0].tolist()
            return JSONResponse(content=content)
        except Exception as e:
            return JSONResponse(content={"Error": "Inference failed with error: " + str(e)})

app1 = ModelEmbedding.bind(query_retriever_name, query_version)
app2 = ModelEmbedding.bind(ctx_retriever_name, ctx_version)

@serve.deployment
@serve.ingress(app)
class FastAPIDeployment:
    # FastAPI will automatically parse the HTTP request for us.
    def __init__(self, app1: ModelEmbedding, app2: ModelEmbedding):
        self.app1 = app1
        self.app2 = app2
        self.db = QdrantFaceDatabase(url=QDRANT_DB)

    @app.get("/hello")
    def hello(self, name: str) -> JSONResponse:
        return JSONResponse(content={"message": f"Hello, {name}!"})

    @app.post("/query_embed")
    async def query_embed(self, textRequest: List[str]) -> JSONResponse:
        [outputs] = await asyncio.gather(self.app1.remote(textRequest))
        return outputs

    @app.post("/ctx_embed")
    async def ctx_embed(self, textRequest: List[str]) -> JSONResponse:
        [outputs] = await asyncio.gather(self.app2.remote(textRequest))
        return outputs
    
    @app.post("/search")
    async def search(self, textRequest: List[str]) -> JSONResponse:
        # -------------------INFERENCE--------------------
        query_embed_response: JSONResponse = await self.query_embed(textRequest)
        query_embed = query_embed_response.body
        contexts = self.db.search(
            collection_name=collection_name, 
            vector=query_embed, 
            top_k=top_k, 
            threshold=threshold
        )
        if contexts is None:
            return JSONResponse(content={"Error": "No results found!"})
        
        context_embeds: JSONResponse = await self.ctx_embed(contexts)
        context_passages = self.db.get_passage(context_embeds.body)
        
        if context_passages is None:
            return JSONResponse(content={"Error": "No results found!"})
        
        if use_rerank:
            scores = compute_similarity(query_embed, context_embeds)
            passages_arranged = context_passages[scores.argsort()]
            return JSONResponse(content=passages_arranged)

        return JSONResponse(content=context_passages)


# 2: Deploy the deployment.
mainapp = FastAPIDeployment.bind(app1, app2)