from typing import List, Any
from abc import ABC, abstractmethod


class InterfaceService(ABC):
    '''Interface for Face Recognition Service'''

    @abstractmethod 
    def post_docs(self, texts: List[Any], **kwargs) -> None:
        ...

    @abstractmethod
    def retrieve(self, texts: List[Any], **kwargs) -> List[Any]:
        ...