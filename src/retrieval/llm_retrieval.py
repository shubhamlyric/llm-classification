from langchain.agents import Tool, initialize_agent
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from src.utils.config import Parameters
import polars as pl
from typing import List, Dict

def create_search_tool(vectorstore, dataset_name):
    def retrieve_similar_items(query, k=5):
        results = vectorstore.similarity_search(query, k=k)
        return results

    return Tool(
        name=f'{dataset_name}SimilaritySearch',
        func=lambda q: retrieve_similar_items(q),
        description=f'Use this tool to find similar {dataset_name.lower()}s based on their features.'
    )

def get_llm(parameters: Parameters):
    if parameters.model_name.lower() == "gpt-3.5-turbo":
        return OpenAI(
            model_name=parameters.model_name,
            temperature=0,
            max_tokens=150
        )
    # Add more conditions here for other model types
    else:
        raise ValueError(f"Unsupported model: {parameters.model_name}")

def create_prompt_template(dataset_name: str):
    return PromptTemplate(
        input_variables=["input_features", "similar_items"],
        template=f"""
Given the following {dataset_name} details:
{{input_features}}

Here are similar {dataset_name}s and their target values:
{{similar_items}}

Based on this information, what is the likely target value for this {dataset_name}? Provide a concise answer.
"""
    )

def create_agent(search_tool, llm):
    return initialize_agent(
        tools=[search_tool],
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True
    )

def setup_retrieval_tool(vectorstore, parameters: Parameters):
    dataset_name = parameters.dataset_name
    
    # Set up the LLM
    llm = get_llm(parameters)
    
    # Create the search tool
    search_tool = create_search_tool(vectorstore, dataset_name)
    
    # Create the prompt template
    prompt_template = create_prompt_template(dataset_name)
    
    # Create the agent
    agent = create_agent(search_tool, llm)

    return llm, search_tool, prompt_template, agent  # Return all four components

def process_and_predict(new_data: pl.DataFrame, agent, prompt_template, parameters: Parameters) -> List[Dict]:
    predictions = []

    for row in new_data.iter_rows(named=True):
        input_text = create_input_text(row, parameters)
        
        # Agent retrieves similar items
        similar_items = agent.run(f"Find items similar to: {input_text}")
        
        # Format similar items information
        similar_info = format_similar_items(similar_items)
        
        # Construct the prompt
        prompt = prompt_template.format(
            input_features=input_text,
            similar_items=similar_info
        )
        
        # Get prediction from the agent
        prediction = agent.run(prompt).strip()
        
        predictions.append({
            'Id': row.get('Id', 'Unknown'),  # Adjust this based on your actual ID column name
            'Prediction': prediction
        })

    return predictions

def create_input_text(row: Dict, parameters: Parameters) -> str:
    # Create a string representation of the input features
    # Adjust this based on your actual data structure
    return ", ".join([f"{k}={v}" for k, v in row.items() if k != parameters.target_column])

def format_similar_items(similar_items: str) -> str:
    # Format the similar items string returned by the agent
    # You might need to adjust this based on the actual format of the agent's output
    return similar_items

def setup_retrieval_and_prediction(vectorstore, parameters: Parameters):
    dataset_name = parameters.dataset_name
    
    llm = get_llm(parameters)
    search_tool = create_search_tool(vectorstore, dataset_name)
    prompt_template = create_prompt_template(dataset_name)
    agent = create_agent(search_tool, llm)

    return llm, search_tool, prompt_template, agent, process_and_predict


