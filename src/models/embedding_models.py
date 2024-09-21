from abc import ABC, abstractmethod
import openai
from sentence_transformers import SentenceTransformer
import anthropic
from langchain_community.embeddings import OpenAIEmbeddings
import os
# Import other necessary libraries for different models

class EmbeddingModel(ABC):
    @abstractmethod
    def get_embeddings(self, texts):
        pass

class OpenAIEmbedding(EmbeddingModel):
    def __init__(self, api_key= 'apikey', model="text-embedding-3-large"):
        os.environ["OPENAI_API_KEY"] = api_key
        self.embeddings = OpenAIEmbeddings(model=model)

    def get_embeddings(self, texts):
        return self.embeddings.embed_documents(texts)

class HuggingFaceEmbedding(EmbeddingModel):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, texts):
        return self.model.encode(texts)

class ClaudeEmbedding(EmbeddingModel):
    def __init__(self, api_key, model="claude-3-opus-20240229"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def get_embeddings(self, texts):
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embeddings.append(response.embedding)
        return embeddings

def get_embedding_model(model_type='openai', **kwargs):

    if model_type.lower().startswith("gpt"):
        return OpenAIEmbedding(**kwargs)
    elif model_type.lower().startswith("claude"):
        return ClaudeEmbedding(**kwargs)
    else:
        return HuggingFaceEmbedding(**kwargs)
