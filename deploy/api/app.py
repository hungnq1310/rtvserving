import os
from typing import List, Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from ray import serve
from dotenv import load_dotenv

from rtvserving.db.qdrant_db import QdrantChunksDB
from rtvserving.module.module import BaseModule
from rtvserving.utils.stuff import _init_model_and_tokenizer
from rtvserving.services.v1 import RetrievalServicesV1

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
# FastAPI Definition
############

#?: flexible version
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

############
# Services Definition
############
@serve.deployment
class ServicesV1: # this into services 
    def __init__(self, 
        query_retriever_name: str,
        query_version: int,
        ctx_retriever_name: str,
        ctx_version: int,
        model_server_url: str,
        is_grpc: bool,
        db_url: str
    ):
        # query
        query_module = self.init_module(
            model_name=query_retriever_name,
            model_version=query_version,
            model_server_url=model_server_url,
            is_grpc=is_grpc
        )
        # ctx
        context_module = self.init_module(
            model_name=ctx_retriever_name,
            model_version=ctx_version,
            model_server_url=model_server_url,
            is_grpc=is_grpc
        )
        # db
        db = QdrantChunksDB(url=db_url)
        # Sevices V1 
        self.services = RetrievalServicesV1(
            query_module=query_module,
            context_module=context_module,
            chunk_db=db
        )

    def init_module(self, model_name, model_version, model_server_url, is_grpc):
        model, tokenizer = _init_model_and_tokenizer(
            model_name=model_name,
            model_version=model_version,
            model_server_url=model_server_url,
            is_grpc=is_grpc
        )
        return BaseModule(tokenizer=tokenizer, model=model)

####################
# Deploy the service
####################
service_app = ServicesV1.bind(
    query_retriever_name=query_retriever_name,
    query_version=query_version,
    ctx_retriever_name=ctx_retriever_name,
    ctx_version=ctx_version,
    model_server_url=url,
    is_grpc=grpc,
    db_url=QDRANT_DB
)

####################
# FastAPI Deployment
####################
@serve.deployment
@serve.ingress(app)
class FastAPIDeployment:
    # FastAPI will automatically parse the HTTP request for us.
    def __init__(self, service_app):
        self.service_app = service_app

    @app.get("/hello")
    def hello(self, name: str) -> JSONResponse:
        return JSONResponse(content={"message": f"Hello, {name}!"})

    @app.post("/retrieve_chunks")
    async def retrieve_chunks(self, query: str, chunker_id: str) -> JSONResponse:
        # add remote with async func
        chunks = self.service_app.services.retrieve_chunks.remote(query, chunker_id)
        if not chunks:
            return JSONResponse(content={"Error": "No chunks found!"})
        return JSONResponse(content=chunks)
    
    @app.post("/insert_chunks")
    async def insert_chunks(self, chunks: List[dict], chunker_id: str) -> JSONResponse:
        # add remote with async func
        response = self.service_app.services.insert_chunks.remote(chunks, chunker_id)
        return JSONResponse(content=response)

# 2: Deploy the deployment.
mainapp = FastAPIDeployment.bind(service_app)