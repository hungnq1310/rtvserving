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

3. **Config Hosting Model with Triton Server**
     - Modify **access token** of huggingface in `configs/hf.json` file 
          ```
          {
          "token": "...",
          "models": [
               {
               "name": "pythera/triton.mbert-rtvserving",
               "ref": "main",
               "token": "..."
               }
          ]}
          ```
     - Serving model with **specific** config name. Currently using `tensorrt`, which will load config in `models/<model_name>/configs/<config_name>`. [Docs](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#custom-model-configuration)
          ```bash
          bash -c "python3 -u /hf.py && tritonserver --model-repository=/models --model-config-name=tensorrt"
          ```
     - (Optional) Change `config.pbtxt` of model if needed
          ```python
          - default_model_filename: <file_name> # custom name
          - version_policy: { specific: { 
               versions: [...] # multi-models or only specific version
            }}
          - max_batch_size: 0 # dynamic or specific batch
          ```


4. **Start Hosting Model Services**
     ```bash
     docker-compose up -d
     ```

5. **Run RayServe**
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
