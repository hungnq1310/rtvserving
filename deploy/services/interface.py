from typing import List, Any
from abc import ABC, abstractmethod


class InterfaceService(ABC):
    '''Interface for Face Recognition Service'''

    @abstractmethod
    def embed_query(self, texts: List[Any], **kwargs) -> List[Any]:
        ...

    @abstractmethod 
    def embed_context(self, texts: List[Any], **kwargs) -> List[Any]:
        ...

    @abstractmethod
    def post_documents(self, texts: List[Any], **kwargs) -> List[Any]:
        ...

    @abstractmethod
    def retrieve(self, query: List[Any], **kwargs) -> List[Any]:
        ...