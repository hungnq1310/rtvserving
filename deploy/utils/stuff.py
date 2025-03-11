from transformers import AutoTokenizer
from trism import TritonModel
import os

def _init_model_and_tokenizer(
    model_name, 
    model_version,
    model_server_url,
    is_grpc=False
):
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join("/models", model_name, str(model_version))
    )
    model = TritonModel(
        model=model_name,                 # Model name.
        version=model_version,            # Model version.
        url=model_server_url,              # Triton Server URL.
        grpc=is_grpc                      # Use gRPC or Http.
    )
    # View metadata.
    for inp in model.inputs:
        print(f"name: {inp.name}, shape: {inp.shape}, datatype: {inp.dtype}\n")
    for out in model.outputs:
        print(f"name: {out.name}, shape: {out.shape}, datatype: {out.dtype}\n") 

    return model, tokenizer