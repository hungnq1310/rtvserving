import os
from typing import List, Any, Optional, Union

from fastapi import Depends, HTTPException, Form, APIRouter
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2AuthorizationCodeBearer
from pydantic_settings import BaseSettings, SettingsConfigDict

from jwt import PyJWKClient
import jwt
from typing import Annotated


from rtvserving.db.qdrant_db import QdrantChunksDB
from rtvserving.module.module import BaseModule
from rtvserving.utils.stuff import _init_model_and_tokenizer
from rtvserving.services.v1 import RetrievalServicesV1


class Settings(BaseSettings):
    query_model_name: str = None
    query_model_version: int = None
    query_batch_size: int = None
    ctx_model_name: str = None
    ctx_model_version: int = None
    ctx_batch_size: int = None
    rerank_model_name: str = None
    rerank_model_version: int = None
    rerank_batch_size: int = None
    triton_url: str = "localhost:8000"
    protocol: str = "HTTP"
    verbose: bool = False
    async_set: bool = False
    qdrant_db: str = None
    qdrant_collection_name: str = "retrieval"
    top_k: int = 5
    threshold: float = 0.5
    keycloak_url: str = None
    keycloak_realm: str = None
    keycloak_audience: str = None
    algorithm: str = "RS256"

    model_config = SettingsConfigDict(env_file=".env")

# Parse environment variables
settings = Settings()
grpc = settings.protocol.lower() == "grpc"
# URLs
KEYCLOAK_URL = settings.keycloak_url
KEYCLOAK_REALM = settings.keycloak_realm
TOKEN_URL = f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/token"
AUTHORIZE_URL = f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/auth"
JWKS_URL = f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/certs"

############
# FastAPI Definition
############

oauth_2_scheme = OAuth2AuthorizationCodeBearer(
    tokenUrl=TOKEN_URL,
    authorizationUrl=AUTHORIZE_URL,
    refreshUrl=TOKEN_URL,
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
            algorithms=[settings.algorithm],
            audience=settings.keycloak_audience,
            options={"verify_exp": True},
        )
        return data
    except jwt.exceptions.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Not authenticated")

app = APIRouter()


####################
# Init modules
####################
def init_module( model_name, model_version, model_server_url, is_grpc):
    model, tokenizer = _init_model_and_tokenizer(
        model_name=model_name,
        model_version=model_version,
        model_server_url=model_server_url,
        is_grpc=is_grpc
    )
    return BaseModule(tokenizer=tokenizer, model=model)


query_module = init_module(
    model_name=settings.query_model_name,
    model_version=settings.query_model_version,
    model_server_url=settings.triton_url,
    is_grpc=grpc
)
        # ctx
context_module = init_module(
    model_name=settings.ctx_model_name,
    model_version=settings.ctx_model_version,
    model_server_url=settings.triton_url,
    is_grpc=grpc
)
rerank_module = init_module(
    model_name=settings.rerank_model_name,
    model_version=settings.rerank_model_version,
    model_server_url=settings.triton_url,
    is_grpc=grpc
)
# db
db = QdrantChunksDB(url=settings.qdrant_db)

# Sevices V1 
services = RetrievalServicesV1(
    query_module=query_module,
    context_module=context_module,
    rerank_module=rerank_module,
    chunk_db=db
)


####################
# FastAPI Deployment
####################


@app.get("/hello", dependencies=[Depends(oauth_2_scheme)])
def hello(name: str) -> JSONResponse:
    return JSONResponse(content={"message": f"Hello, {name}!"})

@app.post("/retrieve_chunks", dependencies=[Depends(oauth_2_scheme)])
async def retrieve_chunks(query: str, chunker_id: str, is_rerank: bool = False) -> JSONResponse:
    # add remote with async func
    chunks = services.retrieve_chunks(query, chunker_id)
    if not chunks:
        return JSONResponse(content={"Error": "No chunks found!"})
    
    # rerank
    if is_rerank:   
        chunks = services.rerank(
            query=query,
            chunks=chunks,
        )
    return JSONResponse(content=chunks)

@app.post("/insert_chunks", dependencies=[Depends(oauth_2_scheme)])
async def insert_chunks(chunks: List[dict], chunker_id: str) -> JSONResponse:
    # add remote with async func
    response = services.insert_chunks(chunks, chunker_id)
    return JSONResponse(content=response)

@app.delete("/delete-chunks", dependencies=[Depends(oauth_2_scheme)])
async def delete_chunk_ids(chunk_ids: List[str], chunker_id: str) -> JSONResponse:        # add remote with async func
    response = services.chunk_db.delete(chunk_ids=chunk_ids, chunker_id=chunker_id, doc_id=None)
    return JSONResponse(content=response)   

@app.delete("/delete-chunk", dependencies=[Depends(oauth_2_scheme)])
async def delete_chunk_ids(chunk_id: str, chunker_id: str) -> JSONResponse:        # add remote with async func
    response = services.chunk_db.delete(chunk_ids=[chunk_id], chunker_id=chunker_id, doc_id=None)
    return JSONResponse(content=response)

@app.delete("/delete-doc", dependencies=[Depends(oauth_2_scheme)])
async def delete_doc_id(doc_id: str, chunker_id: str) -> JSONResponse:
    # add remote with async func
    response = services.chunk_db.delete(doc_id=doc_id, chunker_id=chunker_id, chunk_ids=None)
    return JSONResponse(content=response)

@app.delete("/delete-chunker", dependencies=[Depends(oauth_2_scheme)])
async def delete_chunker_id(chunker_id: str) -> JSONResponse:
    # add remote with async func
    response = services.chunk_db.delete_chunker(chunker_id)
    return JSONResponse(content=response)


@app.get("/get_config_model", dependencies=[Depends(oauth_2_scheme)])
async def get_config_model(model_name: str, model_version: str) -> JSONResponse:
    """
    Get model config
    """
    if not model_name or not model_version:
        return JSONResponse(content={"Error": "Model name and version are required!"})
    # get config
    config = services.get_config_model(model_name, model_version)
    if not config:
        return JSONResponse(content={"Error": "No config found!"})
    return JSONResponse(content=config)

@app.get("/settings", dependencies=[Depends(oauth_2_scheme)])
async def get_settings() -> JSONResponse:
    """
    Get settings
    """
    return JSONResponse(content=settings.dict())