from trism import TritonModel
from transformers import AutoTokenizer
from dataclasses import dataclass

@dataclass
class BaseModule:
    tokenizer: AutoTokenizer = None
    model: TritonModel = None
