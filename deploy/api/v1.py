import os
from typing import List, Any, Optional, Union

from fastapi import Depends, HTTPException, Form, APIRouter
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2AuthorizationCodeBearer

from jwt import PyJWKClient
import jwt
from typing import Annotated
from dotenv import load_dotenv

from ..rtvserving.db.qdrant_db import QdrantChunksDB
from ..rtvserving.module.module import BaseModule
from ..rtvserving.utils.stuff import _init_model_and_tokenizer
from ..rtvserving.services.v1 import RetrievalServicesV1

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
ALGORITHM = os.getenv("ALGORITHM", "RS256")
# URLs
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
            algorithms=[ALGORITHM],
            audience=KEYCLOAK_AUDIENCE,
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
    model_name=query_retriever_name,
    model_version=query_version,
    model_server_url=url,
    is_grpc=grpc
)
        # ctx
context_module = init_module(
    model_name=ctx_retriever_name,
    model_version=ctx_version,
    model_server_url=url,
    is_grpc=grpc
)
# db
db = QdrantChunksDB(url=QDRANT_DB)

# Sevices V1 
services = RetrievalServicesV1(
    query_module=query_module,
    context_module=context_module,
    chunk_db=db
)


####################
# FastAPI Deployment
####################


@app.get("/hello", dependencies=[Depends(oauth_2_scheme)])
def hello(name: str) -> JSONResponse:
    return JSONResponse(content={"message": f"Hello, {name}!"})

@app.post("/retrieve_chunks", dependencies=[Depends(oauth_2_scheme)])
async def retrieve_chunks(query: str, chunker_id: str) -> JSONResponse:
    # add remote with async func
    chunks = services.retrieve_chunks(query, chunker_id)
    if not chunks:
        return JSONResponse(content={"Error": "No chunks found!"})
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

