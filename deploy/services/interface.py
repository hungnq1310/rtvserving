from typing import List, Any
from abc import ABC, abstractmethod


class InterfaceService(ABC):
    '''Interface for Face Recognition Service'''

    @abstractmethod 
    def embed(self, texts: List[Any], **kwargs) -> List[Any]:
        ...

    @abstractmethod
    def run(self, texts: List[Any], **kwargs) -> List[Any]:
        ...