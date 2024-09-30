""" Absract class """

from typing import List
from abc import ABC, abstractmethod


class BaseStorage(ABC):
    """Abstract class for storage"""

    @abstractmethod
    def load_data(self, texts: List):
        """Load data"""

    @abstractmethod
    def similarity_search(self, query, k=4):
        """Search for similar data"""
