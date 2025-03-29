from typing import List, Dict, Any, Tuple
from trism import TritonModel
from transformers import AutoTokenizer
import numpy as np


class BaseModule:
    """Factory class"""
    def __init__(self, tokenizer: AutoTokenizer, model: TritonModel):
        self.tokenizer = tokenizer
        self.model = model

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        text_responses = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors="np"
        )
        try:
            outputs: Dict[Any] = self.model.run(data = [
                text_responses['input_ids'], 
                text_responses['attention_mask'], 
                text_responses['token_type_ids']
            ])
            outputs = list(outputs.values())[0] # BxLx768
            outputs = outputs.reshape(len(texts), -1, 768)[:, 0].tolist() # Bx768
            return outputs
        except Exception as e:
            print(f"Error embedding text: {e}")
            return []
        
    # Encode text
    def rerank(self, query: str, context: str):

        # Tokenize sentences
        encoded_pair = self.tokenizer(
            query,
            context,
            return_tensors="pt",
        )
        for key in encoded_pair:
            encoded_pair[key] = encoded_pair[key].numpy()
        # Compute token embeddings
        score = self.model.run(
            data=[
                encoded_pair['input_ids'],
                encoded_pair['attention_mask'],
                encoded_pair['token_type_ids']
            ]
        )['logits']
        # Get the score
        score = score.reshape(-1).tolist()
        return score
