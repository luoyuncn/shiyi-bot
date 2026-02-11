"""Channel base class"""
from abc import ABC, abstractmethod


class BaseChannel(ABC):
    """Channel abstract base class"""

    @abstractmethod
    async def start(self):
        """Start channel"""
        pass

    @abstractmethod
    async def stop(self):
        """Stop channel"""
        pass
