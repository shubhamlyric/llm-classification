import polars as pl
import numpy as np
from typing import List, Dict, Any
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import pinecone
from models import get_embedding_model  # Import the get_embedding_model function
from langchain.vectorstores import Pinecone as PineconeVectorStore

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

def create_text_description(row: pl.Series, feature_columns: List[str]) -> str:
    return ", ".join([f"{col}={row[col]}" for col in feature_columns])

def vectorize_data(df: pl.DataFrame, feature_columns: List[str], embedding_model: str, **kwargs) -> np.ndarray:
    # Create text descriptions
    text_data = df.select(feature_columns).apply(lambda row: create_text_description(row, feature_columns))
    
    # Use the specified embedding model to create embeddings
    model = get_embedding_model(model_type=embedding_model, **kwargs)
    embeddings = model.get_embeddings(text_data.to_list())
    
    return np.array(embeddings)

def store_in_pinecone(df: pl.DataFrame, embeddings: np.ndarray, feature_columns: List[str], index_name: str):
    # Initialize Pinecone
    pinecone.init(api_key="b709e216-9b1c-455b-997d-525349516113",
                  environment="YOUR_ENVIRONMENT")
    
    # Create or connect to an existing index
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=embeddings.shape[1])
    
    index = pinecone.Index(index_name)

    # Prepare metadata
    metadata = df.select(feature_columns).apply(
        lambda row: {
            'text': create_text_description(row, feature_columns),
            **{col: str(row[col]) for col in feature_columns}
        }
    ).to_list()

    # Upsert data into Pinecone
    vectors = zip(
        [f"id_{i}" for i in range(len(df))],
        embeddings.tolist(),
        metadata
    )

    index.upsert(vectors=list(vectors))

def process_data(
    input_file: str,
    feature_columns: List[str],
    index_name: str,
    configs: Dict[str, Any]
) -> int:
    processed_rows = 0

    for chunk_df in pl.read_csv(input_file, batch_size=configs['processing']['batch_size']):
        # Preprocess the data
        processed_chunk = preprocess_data(chunk_df, feature_columns)
        
        # Vectorize the data using the specified embedding model
        embeddings = vectorize_data(
            processed_chunk, 
            feature_columns, 
            configs['embedding']['model'],
            **configs['embedding']['params']
        )
        
        # Generate IDs for the vectors
        ids = [f"id_{i+processed_rows}" for i in range(len(processed_chunk))]
        
        # Store in Pinecone
        store_in_pinecone(processed_chunk, embeddings, feature_columns, index_name)
        
        processed_rows += len(processed_chunk)
        print(f"Processed and stored {processed_rows} rows so far...")
    
    print(f"Finished processing {processed_rows} rows.")
    return processed_rows

# Remove the perform_sample_similarity_search function as it's not needed for this approach
