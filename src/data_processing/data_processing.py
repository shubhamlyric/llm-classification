import polars as pl
import numpy as np
from typing import List, Dict, Any
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import os
from pinecone import Pinecone, ServerlessSpec
from src.models.embedding_models import get_embedding_model  # Import the get_embedding_model function
# from langchain.vectorstores import Pinecone as PineconeVectorStore
# from langchain_community.vectorstores import Pinecone
# from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import Pinecone, ServerlessSpec

from dotenv import load_dotenv

# Load environment variables at the top of the file
load_dotenv()
def preprocess_data(df: pl.DataFrame, feature_columns: List[str]) -> pl.DataFrame:

 
    # Handle missing values
    for col in feature_columns:
        if df[col].dtype == pl.Utf8:
            df = df.with_columns(pl.col(col).fill_null("Unknown"))
        else:
            imputer = SimpleImputer(strategy='mean')
            df = df.with_columns(pl.col(col).fill_null(strategy="mean"))
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in feature_columns:
        if df[col].dtype == pl.Utf8:
            encoded = le.fit_transform(df[col].to_numpy())
            df = df.with_columns(pl.Series(name=col, values=encoded))
    
    # Normalize numerical features
    scaler = StandardScaler()
    numerical_columns = [col for col in feature_columns if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
    scaled_data = scaler.fit_transform(df.select(numerical_columns).to_numpy())
    
    for i, col in enumerate(numerical_columns):
        df = df.with_columns(pl.Series(name=col, values=scaled_data[:, i]))
    
    return df

def create_text_description(row: Dict[str, Any], feature_columns: List[str]) -> str:
    return ", ".join([f"{col}={row[col]}" for col in feature_columns])

def vectorize_data(df: pl.DataFrame, feature_columns: List[str], embedding_model: str, **kwargs) -> np.ndarray:
    # Create text descriptions from feature columns
    print('type of df', type(df))

    # Create the text description using all feature columns
    text_expr = pl.concat_str([
        pl.lit(f"{col}="),
        pl.col(col).cast(pl.Utf8),
        pl.lit(", ")
    ] for col in feature_columns).arr.join("")

    df = df.with_columns(
        text_expr.alias('text_description')
    )

    text_data = df["text_description"].to_list()
    print('text_data', text_data)

    # Use the specified embedding model to create embeddings
    model = get_embedding_model(model_type=embedding_model, **kwargs)
    embeddings = model.get_embeddings(text_data)
    
    return np.array(embeddings)

def initialize_pinecone_index(index_name, dimension=1536, metric='cosine'):
    # Initialize Pinecone

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Check if the index already exists
    if index_name not in pc.list_indexes().names():
        # Create a new index
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud='gcp',
                region=os.getenv("PINECONE_ENVIRONMENT")
            )
        )
        print(f"Created new Pinecone index: {index_name}")
    else:
        print(f"Pinecone index {index_name} already exists")

    # Get the index
    return pc.Index(index_name)

def store_in_pinecone(df: pl.DataFrame, embeddings: np.ndarray, feature_columns: List[str], index_name: str):
    # Initialize or get existing Pinecone index
    index = initialize_pinecone_index(index_name, dimension=embeddings.shape[1])

    # Prepare metadata
    metadata = df.select(feature_columns).apply(
        lambda row: {
            'text': create_text_description(row, feature_columns),
            **{col: str(row[col]) for col in feature_columns}
        }
    ).to_list()

    # Upsert data into Pinecone
    vectors = [
        (f"id_{i}", embedding.tolist(), meta)
        for i, (embedding, meta) in enumerate(zip(embeddings, metadata))
    ]

    index.upsert(vectors=vectors)

def process_data(
    input_file: str,
    target_column: str,
    index_name: str,
    configs: Dict[str, Any],
    embedding_model: str,
) -> int:
    # Get the total number of rows by reading only the first column
    data = pl.read_csv(input_file, batch_size=1)
    feature_columns = [col for col in data.columns if col != target_column]
    total_rows = pl.read_csv(input_file, columns=[0]).height

    chunk_size = configs['batch_size']
    processed_rows = 0

    # Initialize Pinecone index
    initialize_pinecone_index(index_name)

    # Process the data in chunks
    for chunk_start in range(0, total_rows, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_rows)
        
        # Read a chunk of data
        chunk_df = pl.read_csv(input_file, skip_rows=chunk_start, n_rows=chunk_end - chunk_start)

        feature_columns = [col for col in chunk_df.columns if col != target_column]

        # Preprocess the data
        processed_chunk = preprocess_data(chunk_df, feature_columns)
        
        # Vectorize the data using the specified embedding model
        embeddings = vectorize_data(
            processed_chunk, 
            feature_columns, 
            embedding_model
        )
        
        # Store in Pinecone
        store_in_pinecone(processed_chunk, embeddings, feature_columns, index_name)
        
        processed_rows += len(processed_chunk)
        print(f"Processed and stored {processed_rows} out of {total_rows} rows...")

    print(f"Finished processing {processed_rows} rows.")
    return processed_rows

# Remove the perform_sample_similarity_search function as it's not needed for this approach
