from typing import List, Any
from abc import ABC, abstractmethod


class InterfaceService(ABC):
    '''Interface for Face Recognition Service'''

    @abstractmethod 
    def insert_chunks(self, chunks: List[Any], **kwargs) -> None:
        ...

    @abstractmethod
    def retrieve_chunks(self, query: str, **kwargs) -> List[Any]:
        ...