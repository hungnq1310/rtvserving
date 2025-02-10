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

2. **Configure Environment For API**
    ```bash
    # Required environment variables
     # Query Model Settings
     QUERY_MODEL_NAME=mbert.query
     QUERY_MODEL_VERSION=2
     BATCH_SIZE=1

     # Context Model Settings
     CTX_MODEL_NAME=mbert.context
     CTX_MODEL_VERSION=2
     BATCH_SIZE=10

     # Server Configuration
     TRITON_URL=localhost:8000
     PROTOCOL=HTTP  # or GRPC
     VERBOSE=False
     ASYNC_SET=False

     # Qdrant Settings
     QDRANT_DB=<qdrant_url>
     QDRANT_COLLECTION_NAME=retrieval
     TOP_K=5
     THRESHOLD=0.5

     # Additional Settings
     USE_RERANK=False
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


## API Endpoints

- `POST /query_embed`: Generates embeddings for query text
- `POST /ctx_embed`: Generates embeddings for context passages  
- `POST /search`: Performs similarity search using query and context embeddings

## Performance Optimization

- TensorRT acceleration enabled for both models
- Configurable batch sizes and sequence lengths
- Optional model quantization supported
