import os
from typing import List, Any, Optional, Union

from fastapi import FastAPI, Depends, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2AuthorizationCodeBearer
import requests
from pydantic import BaseModel

from jwt import PyJWKClient
import jwt
from typing import Annotated

from ray import serve
from dotenv import load_dotenv

from rtvserving.db.qdrant_db import QdrantChunksDB
from rtvserving.module.module import BaseModule
from rtvserving.utils.stuff import _init_model_and_tokenizer
from rtvserving.services.v1 import RetrievalServicesV1

import asyncio

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

# Keycloak Configuration
KEYCLOAK_URL = os.getenv("KEYCLOAK_URL", "...")
KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM", "...")
KEYCLOAK_AUDIENCE = os.getenv("KEYCLOAK_AUDIENCE", "...")
# KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID", "...")
# KEYCLOAK_CLIENT_SECRET = os.getenv("KEYCLOAK_CLIENT_SECRET", "...")
# KEYCLOAK_ADMIN_USERNAME = os.getenv("KEYCLOAK_ADMIN_USERNAME", "...")
# KEYCLOAK_ADMIN_PASSWORD = os.getenv("KEYCLOAK_ADMIN_PASSWORD", "...")

# URLs
TOKEN_URL = f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/token"
USER_URL = f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users"
JWKS_URL = f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/certs"
 

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

oauth_2_scheme = OAuth2AuthorizationCodeBearer(
    tokenUrl=f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/token",
    authorizationUrl=f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/auth",
    refreshUrl=f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/token",
)

# Hàm xác thực Access Token với Keycloak
async def valid_access_token(access_token: Annotated[str, Depends(oauth_2_scheme)]):
    optional_custom_headers = {"User-agent": "custom-user-agent"}
    jwks_client = PyJWKClient(JWKS_URL, headers=optional_custom_headers)
    try:
        signing_key = jwks_client.get_signing_key_from_jwt(access_token)
        data = jwt.decode(
            access_token,
            signing_key.key,
            algorithms=["RS256"],
            audience=KEYCLOAK_AUDIENCE,
            options={"verify_exp": True},
        )
        return data
    except jwt.exceptions.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Not authenticated")

# Model cho đăng ký & đăng nhập
class UserRegister(BaseModel):
    username: str
    password: str
    email: str

class UserLogin(BaseModel):
    username: str
    password: str

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:7777",
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
        self.query_module = self.init_module(
            model_name=query_retriever_name,
            model_version=query_version,
            model_server_url=model_server_url,
            is_grpc=is_grpc
        )
        # ctx
        self.context_module = self.init_module(
            model_name=ctx_retriever_name,
            model_version=ctx_version,
            model_server_url=model_server_url,
            is_grpc=is_grpc
        )
        # db
        self.db = QdrantChunksDB(url=db_url)
        # Sevices V1 
        self.services = RetrievalServicesV1(
            query_module=self.query_module,
            context_module=self.context_module,
            chunk_db=self.db
        )

    def init_module(self, model_name, model_version, model_server_url, is_grpc):
        model, tokenizer = _init_model_and_tokenizer(
            model_name=model_name,
            model_version=model_version,
            model_server_url=model_server_url,
            is_grpc=is_grpc
        )
        return BaseModule(tokenizer=tokenizer, model=model)
    
    async def retrieve_chunks(self, query: str, chunker_id: str):
        return self.services.retrieve_chunks(query, chunker_id)
    
    async def insert_chunks(self, chunks: List[dict], chunker_id: str):
        return self.services.insert_chunks(chunks, chunker_id)
    
    async def delete_chunks(self, chunk_ids: Union[str, List[str]], chunker_id: str):        
        return self.db.delete(chunk_ids=chunk_ids, chunker_id=chunker_id)
    
    async def delete_doc_id(self, doc_id: str, chunker_id: str):
        return self.db.delete(doc_id=doc_id, chunker_id=chunker_id)
    
    async def delete_chunker(self, chunker_id: str):
        return self.db.delete_chunker(chunker_id)
    

####################
# Deploy the service
####################
service_app1 = ServicesV1.bind(
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
        # self.insert_app = insert_app

    @app.get("/hello", dependencies=[Depends(oauth_2_scheme)])
    def hello(self, name: str) -> JSONResponse:
        return JSONResponse(content={"message": f"Hello, {name}!"})

    @app.post("/retrieve_chunks", dependencies=[Depends(oauth_2_scheme)])
    async def retrieve_chunks(self, query: str, chunker_id: str) -> JSONResponse:
        # add remote with async func
        chunks = await self.service_app.retrieve_chunks.remote(query, chunker_id)
        if not chunks:
            return JSONResponse(content={"Error": "No chunks found!"})
        return JSONResponse(content=chunks)
    
    @app.post("/insert_chunks", dependencies=[Depends(oauth_2_scheme)])
    async def insert_chunks(self, chunks: List[dict], chunker_id: str) -> JSONResponse:
        # add remote with async func
        response = await self.service_app.insert_chunks.remote(chunks, chunker_id)
        return JSONResponse(content=response)
    
    @app.delete("/delete-chunks", dependencies=[Depends(oauth_2_scheme)])
    async def delete_chunk_ids(self, chunk_ids: List[str], chunker_id: str) -> JSONResponse:        # add remote with async func
        response = await self.service_app.delete_chunks.remote(chunk_ids=chunk_ids, chunker_id=chunker_id)
        return JSONResponse(content=response)
    
    @app.delete("/delete-chunk", dependencies=[Depends(oauth_2_scheme)])
    async def delete_chunk_ids(self, chunk_id: str, chunker_id: str) -> JSONResponse:        # add remote with async func
        response = await self.service_app.delete_chunks.remote(chunk_ids=[chunk_id], chunker_id=chunker_id)
        return JSONResponse(content=response)
    
    @app.delete("/delete-doc", dependencies=[Depends(oauth_2_scheme)])
    async def delete_doc_id(self, doc_id: str, chunker_id: str) -> JSONResponse:
        # add remote with async func
        response = await self.service_app.delete_doc_id.remote(doc_id=doc_id, chunker_id=chunker_id)
        return JSONResponse(content=response)
    
    @app.delete("/delete-chunker", dependencies=[Depends(oauth_2_scheme)])
    async def delete_chunker_id(self, chunker_id: str) -> JSONResponse:
        # add remote with async func
        response = await self.service_app.delete_chunker.remote(chunker_id)
        return JSONResponse(content=response)

# 2: Deploy the deployment.
mainapp = FastAPIDeployment.bind(service_app1)