# Deployment Guide

This guide explains how to deploy the retrieval models using Triton Inference Server and FastAPI.

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA GPU with CUDA support
- Model files:
    - Query model checkpoint/ONNX
    - Context model checkpoint/ONNX
    - Rerank model checkpoint/ONNX
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
     - run `pip install -r requiements.txt -e .`
     - Model already downloaded in image, if need to handle model manually, place model files in `./models/` directory and volumn in docker compose.
     - Configure model settings in `config.pbtxt` files - [Guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html)
     - Set proper execution accelerators and batch sizes

2. **Configure Environment For API**
    ```bash
    # Required environment variables
     # Query Model Settings
     QUERY_MODEL_NAME=mbert.query
     QUERY_MODEL_VERSION=1
     QUERY_BATCH_SIZE=0

     # Context Model Settings
     CTX_MODEL_NAME=mbert.context
     CTX_MODEL_VERSION=1
     CTX_BATCH_SIZE=0

     # Rerank Model Settings
     RERANK_MODEL_NAME=mbert.rerank
     RERANK_MODEL_VERSION=1
     RERANK_BATCH_SIZE=0

     # Server Configuration
     RTV_TRITON_URL=localhost:6000
     PROTOCOL=HTTP  # or GRPC
     VERBOSE=true
     ASYNC_SET=False
     ```

3. **Handle model manually,**
     - Create `configs/hf.json` file and Modify **access token** of huggingface following format:
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
     - Volumn `hf.py` and `hf.json` 
          ```bash
          bash -c python3 -u /hf.py && tritonserver --model-repository=/models 
          ```
     - (Optional) Serving model with **specific** config name. Currently using `tensorrt`, which will load config in `models/<model_name>/configs/<config_name>`. [Docs](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#custom-model-configuration)
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

## Run fastapi
```
fastapi dev mainapp.py
```
NOTE: duo to authorization, you need to handle keycloak token to access these API 

## API Endpoints

- `POST /query_embed`: Generates embeddings for query text
- `POST /ctx_embed`: Generates embeddings for context passages  
- `POST /search`: Performs similarity search using query and context embeddings
- `GET /get_config_model`: Retrieves the configuration of a specific model version.
- `GET /settings`: Retrieves the current application settings.
- `POST /retrieve_chunks`: Retrieves chunks based on a query and optionally reranks them.
- `POST /insert_chunks`: Inserts chunks into the database.
- `DELETE /delete-chunks`: Deletes multiple chunks by their IDs.
- `DELETE /delete-chunk`: Deletes a single chunk by its ID.
- `DELETE /delete-doc`: Deletes a document by its ID.
- `DELETE /delete-chunker`: Deletes a chunker by its ID.


## Performance Optimization

- TensorRT acceleration enabled for both models
- Configurable batch sizes and sequence lengths
- Optional model quantization supported

### Openvino - Query Model - Batch1
```
Inferences/Second vs. Client p95 Batch Latency
Concurrency: 1, throughput: 8.56111 infer/sec, latency 143606 usec
Concurrency: 2, throughput: 8.51688 infer/sec, latency 270469 usec
Concurrency: 3, throughput: 8.51644 infer/sec, latency 403345 usec
Concurrency: 4, throughput: 8.48926 infer/sec, latency 527736 usec
```