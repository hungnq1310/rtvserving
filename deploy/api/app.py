import os
from typing import List, Any
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from transformers import AutoTokenizer
from ray import serve
import requests
from dotenv import load_dotenv

from trism import TritonModel

from utils.score_compute import compute_similarity

load_dotenv()
# Parse environment variables
#
query_retriever_name    = os.getenv("QUERY_MODEL_NAME")
ctx_retriever_name    = os.getenv("CTX_MODEL_NAME") 


model_version = int(os.getenv("MODEL_VERSION", ""))
batch_size    = int(os.getenv("BATCH_SIZE", 1))
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
collection_name = os.getenv("QDRANT_COLLECTION_NAME", "chitchat_embed")
top_k = int(os.getenv("TOP_K", 5))
threshold = float(os.getenv("THRESHOLD", 0.5))
QDRANT_DB     = os.getenv("QDRANT_DB", "")
ENCODER_URL   = os.getenv("ENCODER", "")

print(f"Query model: {query_retriever_name}")
print(f"Context model: {ctx_retriever_name}")
print(f"Model version: {model_version}")
print(f"Batch size: {batch_size}")
print(f"URL: {url}")
print(f"Protocol: {protocol}")
print(f"Verbose: {verbose}")
print(f"Async set: {async_set}")
print(f"Use rerank: {use_rerank}")
print(f"QDRANT_DB: {QDRANT_DB}")
print(f"ENCODER_URL: {ENCODER_URL}")


# ----------------------------------------------------------
# Create triton model.

#1
query_tokenizer = AutoTokenizer.from_pretrained(
    os.path.join("models", query_retriever_name, str(model_version))
)
query_model = TritonModel(
  model=query_retriever_name,                 # Model name.
  version=model_version,            # Model version.
  url=url,                          # Triton Server URL.
  grpc=grpc                         # Use gRPC or Http.
)
# View metadata.
for inp in query_model.inputs:
  print(f"name: {inp.name}, shape: {inp.shape}, datatype: {inp.dtype}\n")
for out in query_model.outputs:
  print(f"name: {out.name}, shape: {out.shape}, datatype: {out.dtype}\n")

#2
ctx_tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(os.curdir, "models", ctx_retriever_name, str(model_version))
)
ctx_model = TritonModel(
    model=ctx_retriever_name,                 # Model name.
    version=model_version,            # Model version.
    url=url,                          # Triton Server URL.
    grpc=grpc                         # Use gRPC or Http.
    )
# View metadata.
for inp in ctx_model.inputs:
  print(f"name: {inp.name}, shape: {inp.shape}, datatype: {inp.dtype}\n")
for out in ctx_model.outputs:
    print(f"name: {out.name}, shape: {out.shape}, datatype: {out.dtype}\n") 

#3
# Connect to Qdrant
from deploy.db.interface import QdrantFaceDatabase
db = QdrantFaceDatabase(url=QDRANT_DB)


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

@app.get("/hello")
def say_hello(name: str) -> str:
    return f"Hello {name}!"

@app.post("/query_embed")
async def query_embed(textRequest: List[str]) -> JSONResponse:

    # Word-segment the input texts
    text_responses = query_tokenizer(textRequest, padding=True, truncation=True, return_tensors="np")
    print(text_responses)

    # -------------------INFERENCE--------------------
    try:
        start_time = time.time()
        outputs = query_model.run(data = [
            text_responses['input_ids'], 
            text_responses['attention_mask'], 
            text_responses['token_type_ids']
        ])
        end_time = time.time()
        print("Process time: ", end_time - start_time)
        print(outputs['embeddings'].shape)
        return JSONResponse(content=outputs["embeddings"][:, 0].tolist())
    except Exception as e:
        return JSONResponse(content={"Error": "Inference failed with error: " + str(e)})

    # ----------------------------------------------------------------

@app.post("/ctx_embed")
async def ctx_embed(textRequest: List[str]) -> JSONResponse:

    # Word-segment the input texts
    text_responses = ctx_tokenizer(textRequest, padding=True, truncation=True, return_tensors="np")
    print(text_responses)

    # -------------------INFERENCE--------------------
    try:
        start_time = time.time()
        outputs = ctx_model.run(data = [
            text_responses['input_ids'], 
            text_responses['attention_mask'], 
            text_responses['token_type_ids']
        ])
        end_time = time.time()
        print("Process time: ", end_time - start_time)
        return JSONResponse(content=outputs["embeddings"][:, 0].tolist())
    except Exception as e:
        return JSONResponse(content={"Error": "Inference failed with error: " + str(e)})
    

@app.post("/search")
async def search(textRequest: List[str]) -> JSONResponse:
    # -------------------INFERENCE--------------------
    query_embed: List[List[float]] = await query_embed(textRequest)
    contexts = db.search(
        collection_name=collection_name, 
        vector=query_embed, 
        top_k=top_k, 
        threshold=threshold
    )
    context_embeds = await ctx_embed(contexts)
    context_passages = db.get_passages(context_embeds)

    if use_rerank:
        scores = compute_similarity(query_embed, context_embeds)
        passages_arranged = context_passages[scores.argsort()]
        return JSONResponse(content=passages_arranged)

    return JSONResponse(content=context_passages)

