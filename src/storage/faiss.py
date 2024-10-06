"""
FAISS Implementation
"""

from typing import List, Dict, Optional
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

    def load_data(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """
        Add texts to the FAISS index with optional metadata.
        """
        documents = []
        for idx, text in enumerate(texts):
            metadata = metadatas[idx] if metadatas is not None else {}
            documents.append(Document(page_content=text, metadata=metadata))
        
        if self.db is None:
            self.db = FAISS.from_documents(documents, self.embedding_function)
        else:
            self.db.add_documents(documents)

    def similarity_search(self, query: str, k: int = 4, query_embedding: Optional[List[float]] = None):
        """
        Perform similarity search on the FAISS index and return texts with metadata.
        """
        if query_embedding is None:
            results = self.db.similarity_search(query, k)
        else:
            results = self.db.similarity_search_by_vector(query_embedding, k)
        
        # Extract texts and metadata
        formatted_results = []
        for i, item in enumerate(results, 1):
            text = item.page_content
            metadata = item.metadata
            formatted_item = {
                "rank": i,
                "text": text,
                "metadata": metadata
            }
            formatted_results.append(formatted_item)
        
        return formatted_results

    def get_embeddings(self, texts: List[str]):
        """
        Get embeddings for the texts
        """
        return self.embedding_function.embed_documents(texts)
