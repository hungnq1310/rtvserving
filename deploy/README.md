# Deployment Guide

This guide explains how to deploy the retrieval models using Triton Inference Server and FastAPI.

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA GPU with CUDA support
- Model files:
    - Query model checkpoint/ONNX
    - Context model checkpoint/ONNX
    - Tokenizer files for both models

## Components

1. **Triton Server**
     - Handles model serving and inference
     - Configured via `docker-compose.yml`
     - Uses port 6000 for HTTP and 6001 for gRPC

2. **FastAPI Application** 
     - Provides REST API endpoints
     - Handles tokenization and post-processing
     - Integrates with Qdrant vector database

## Deployment Steps

NOTES: remember to `cd ./deploy` 

1. **Prepare Model Repository**
     - Place model files in `./models/` directory
     - Configure model settings in `config.pbtxt` files - [Guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html)
     - Set proper execution accelerators and batch sizes

2. **Configure Environment**
    ```bash
    # Required environment variables
    QUERY_MODEL_NAME=`retriever_qry`
    CTX_MODEL_NAME=`retriever_ctx`
    MODEL_VERSION=1
    TRITON_URL=localhost:6000
    QDRANT_DB=<qdrant_url>
    BATCH_SIZE=1
    PROTOCOL=HTTP
    VERBOSE=False
    ASYNC_SET=False
    USE_RERANK=False
    QDRANT_COLLECTION_NAME=chitchat_embed
    TOP_K=5
    THRESHOLD=0.5
    ENCODER=<encoder_url>
     ```

3. **Start Hosting Model Services**
     ```bash
     docker-compose up -d
     ```

4. **Run RayServe**
     ```bash
     serve run api.app:mainapp
     ```

## API Endpoints

- `POST /query_embed`: Generates embeddings for query text
- `POST /ctx_embed`: Generates embeddings for context passages  
- `POST /search`: Performs similarity search using query and context embeddings

## Performance Optimization

- TensorRT acceleration enabled for both models
- Configurable batch sizes and sequence lengths
- Optional model quantization supported
