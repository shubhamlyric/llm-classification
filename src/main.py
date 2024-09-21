from vector_store import VectorStore
import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import pandas as pd

class InputData(BaseModel):
    input_file: str
    text_columns: List[str]
    target_column: str
    metadata_columns: Optional[List[str]] = None

    class Config:
        arbitrary_types_allowed = True

class Parameters(BaseModel):
    model_name: str = Field(..., description="Name of the LLM model to use")
    db_type: str = Field(..., description="Type of vector database to use")
    embedding_type: str = Field(..., description="Type of embedding to use")

    class Config:
        allow_population_by_field_name = True

class OutputData(BaseModel):
    processed_rows: int
    vector_ids: List[str]
    sample_similarity_scores: Optional[List[float]] = None

    class Config:
        arbitrary_types_allowed = True

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return json.load(f)

def run(inputs: InputData, parameters: Parameters, configs: Dict[str, Any]) -> OutputData:
    # Initialize the vector store
    vs = VectorStore(
        db_type=parameters.db_type,
        embedding_type=parameters.embedding_type
    )
    
    # Process the CSV file in batches
    processed_rows = 0
    vector_ids = []
    
    for chunk in pd.read_csv(inputs.input_file, chunksize=configs['processing']['batch_size']):
        # Combine text columns
        texts = chunk[inputs.text_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        
        # Prepare metadata
        if inputs.metadata_columns:
            metadata = chunk[inputs.metadata_columns].to_dict('records')
        else:
            metadata = None
        
        # Process and add texts to the vector store
        for i, text in enumerate(texts):
            # Simple chunking (you might want to implement more sophisticated chunking)
            chunks = [text[i:i+configs['processing']['chunk_size']] for i in range(0, len(text), configs['processing']['chunk_size']-configs['processing']['overlap'])]
            
            # Add chunks to vector store
            ids = vs.add_texts(chunks, metadatas=[metadata[i] if metadata else None] * len(chunks))
            vector_ids.extend(ids)
        
        processed_rows += len(chunk)
        print(f"Processed {processed_rows} rows so far...")
    
    print(f"Finished processing {processed_rows} rows.")
    
    # Perform a sample similarity search
    if vector_ids:
        sample_text = vs.similarity_search(vector_ids[0], k=configs['processing']['top_k'])
        sample_similarity_scores = [score for _, score in sample_text if score >= configs['processing']['similarity_threshold']]
    else:
        sample_similarity_scores = None
    
    return OutputData(
        processed_rows=processed_rows,
        vector_ids=vector_ids,
        sample_similarity_scores=sample_similarity_scores
    )

if __name__ == "__main__":
    # Example usage
    input_data = InputData(
        input_file="large_input.csv",
        text_columns=["text_column1", "text_column2"],
        target_column="target",
        metadata_columns=["id", "category"]
    )
    parameters = Parameters(
        model_name="gpt-3.5-turbo",
        db_type="pinecone",
        embedding_type="openai"
    )
    configs = load_config('config.json')
    
    output = run(input_data, parameters, configs)
    print(output)