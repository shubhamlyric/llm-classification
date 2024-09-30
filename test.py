from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
raw_documents = TextLoader("../../../state_of_the_union.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db = FAISS.from_documents(documents, OpenAIEmbeddings())


query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)
print(docs[0].page_content)


"""
PARAMETERS = (
    llm_model_name="meta-llama/Llama-2-7b-hf",
    db_type="pinecone",
    embedding_type="openai",
    dataset_name="titanic-passenger-survival",
    target_column="Survived",
)
"""
