import os
from typing import Optional, List, Any
from qdrant_client.http import models
from abc import ABC, abstractmethod

class InterfaceDatabase(ABC):
    '''Interface for Database Service'''

    @abstractmethod
    def connect_client(self, url, **kwargs) -> None:
        ...

    @abstractmethod
    def insert(self, points: List[Any], **kwagrs) -> None:
        ...

    @abstractmethod
    def update(self, points: List[Any], **kwagrs) -> None:
        ...

    @abstractmethod
    def delete(self, points_ids: List[int], **kwagrs) -> None:
        ...
