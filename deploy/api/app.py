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
    def __init__(self, model_name):
        self.model, self.tokenizer = self._init_model_and_tokenizer(model_name)
        
    def _init_model_and_tokenizer(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join("models", model_name, str(model_version))
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
    
    def __call__(self, textRequest: List[str]):
        text_responses = self.tokenizer(
            textRequest, 
            padding=True, 
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
            return JSONResponse(content=outputs["embeddings"][:, 0].tolist())
        except Exception as e:
            return JSONResponse(content={"Error": "Inference failed with error: " + str(e)})

app1 = ModelEmbedding.bind(query_retriever_name)
app2 = ModelEmbedding.bind(ctx_retriever_name)

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
    def query_embed(self, textRequest: List[str]) -> JSONResponse:
        return self.app1.remote(textRequest)

    @app.post("/ctx_embed")
    def ctx_embed(self, textRequest: List[str]) -> JSONResponse:
        return self.app2.remote(textRequest)
    
    @app.post("/search")
    def search(self, textRequest: List[str]) -> JSONResponse:
        # -------------------INFERENCE--------------------
        query_embed: List[List[float]] = self.query_embed(textRequest)
        contexts = self.db.search(
            collection_name=collection_name, 
            vector=query_embed, 
            top_k=top_k, 
            threshold=threshold
        )
        context_embeds = self.ctx_embed(contexts)
        context_passages = self.db.get_passages(context_embeds)

        if use_rerank:
            scores = compute_similarity(query_embed, context_embeds)
            passages_arranged = context_passages[scores.argsort()]
            return JSONResponse(content=passages_arranged)

        return JSONResponse(content=context_passages)


# 2: Deploy the deployment.
mainapp = FastAPIDeployment.bind(app1, app2)