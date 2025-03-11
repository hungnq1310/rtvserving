from typing import List, Dict, Any
from trism import TritonModel
from transformers import AutoTokenizer


class BaseModule:
    """Factory class"""
    def __init__(self, tokenizer: AutoTokenizer, model: TritonModel):
        self.tokenizer = tokenizer
        self.model = model

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        text_responses = self.tokenizer(
            texts, 
            padding='max_length', 
            truncation=True, 
            return_tensors="np"
        )
        try:
            outputs: Dict[Any] = self.model.run(data = [
                text_responses['input_ids'], 
                text_responses['attention_mask'], 
                text_responses['token_type_ids']
            ])
            outputs = outputs.values()[0] # BxLx768
            content = content.reshape(len(texts), -1, 768)[:, 0].tolist() # Bx768
            return content
        except Exception as e:
            return []