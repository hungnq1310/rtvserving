# rtvserving
## Description
This project provides a comprehensive guide for TensorRT model optimization and deployment. Key experimental features include:

- ONNX dynamic quantization
- PyTorch-based calibration with Pytorch Quantization
- TensorRT-LLM BertAttentionPlugin integration
- Custom TensorRT calibration pipeline for low-resource models
- Deployment with Rayserve

The project aims to convert retriever models through various quantization method while maintaining accuracy and performance. The another goal is to leverage TensorRT engines for high scale. 

## Onnx Dynamic Quantization
 

## TensorRT-LLM BertAttentionPlugin


## TensorRT Calibration API 


## TensorRT Custom Network Lowcode

## Deploy
For detailed deployment instructions and configuration, please refer to the [`deploy`](./deploy/README.md) folder which contains:

- FastAPI application for serving models
- Triton server configuration
- Docker compose setup
- Database interface and utilities

## Reference
