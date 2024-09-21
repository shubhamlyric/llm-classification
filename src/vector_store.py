from langchain.vectorstores import Pinecone, Chroma, FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import pinecone
import os

class VectorStore:
    def __init__(self, db_type, embedding_type):
        self.db_type = db_type
        self.embedding_type = embedding_type
        self.embeddings = self._get_embeddings()
        self.vector_store = self._initialize_db()

    def _get_embeddings(self):
        if self.embedding_type == "openai":
            return OpenAIEmbeddings()
        elif self.embedding_type == "huggingface":
            return HuggingFaceEmbeddings()
        else:
            raise ValueError("Unsupported embedding type")

    def _initialize_db(self):
        if self.db_type == "pinecone":
            pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENV'))
            return Pinecone.from_existing_index(index_name=os.getenv('PINECONE_INDEX'), embedding=self.embeddings)
        elif self.db_type == "chroma":
            return Chroma(embedding_function=self.embeddings, persist_directory="./chroma_db")
        elif self.db_type == "faiss":
            return FAISS(embedding_function=self.embeddings)
        else:
            raise ValueError("Unsupported database type")

    def add_texts(self, texts):
        self.vector_store.add_texts(texts)

    def similarity_search(self, query, k=4):
        return self.vector_store.similarity_search(query, k=k)

    def get_relevant_documents(self, query):
        return self.vector_store.get_relevant_documents(query)

    def get_qa_chain(self):
        retriever = self.vector_store.as_retriever()
        llm = OpenAI()
        return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)