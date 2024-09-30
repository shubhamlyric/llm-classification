"""
FAISS Implementation
"""

from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from .base import BaseStorage


class FaissStorage(BaseStorage):
    """
    FAISS storage class
    """

    def __init__(self, embedding_function):
        self.embedding_function = embedding_function
        self.db = None

    def load_data(self, texts: List):
        """
        Add texts to the FAISS index

        # Document object contains text and metadata
        # incase if needed this can be extended
        """
        if self.db is None:
            self.db = FAISS.from_documents(
                [Document(page_content=text) for text in texts], self.embedding_function
            )
        else:
            self.db.add_documents([Document(page_content=text) for text in texts])

    def similarity_search(self, query, k=4):
        """
        Perform similarity search on the FAISS index
        """
        return self.db.similarity_search(query, k)
