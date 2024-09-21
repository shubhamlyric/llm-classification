from langchain.agents import Tool, initialize_agent
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from src.utils.config import Parameters
import polars as pl
from typing import List, Dict
from collections import Counter

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
        input_variables=["input_features", "similar_items", "most_common_target", "vote_count", "total_votes"],
        template=f"""
Given the following {dataset_name} details:
{{input_features}}

Here are similar {dataset_name}s and their target values:
{{similar_items}}

The most common target value among similar items is: {{most_common_target}}
This value appeared {{vote_count}} times out of {{total_votes}} similar items.

Based on this voting result and the similarity of features, what is the most likely target value for this {dataset_name}?
Please provide a concise prediction, considering both the voting result and any relevant feature similarities or differences. The answer should be a single value.
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
        
        # Extract target values from similar items
        target_values = extract_target_values(similar_items)
        
        # Perform voting
        vote_result = Counter(target_values).most_common(1)[0]
        most_common_target, vote_count = vote_result
        
        # Format similar items information
        similar_info = format_similar_items(similar_items)
        
        # Construct the prompt
        prompt = prompt_template.format(
            input_features=input_text,
            similar_items=similar_info,
            most_common_target=most_common_target,
            vote_count=vote_count,
            total_votes=len(target_values)
        )
        
        # Get final prediction from the agent
        prediction = agent.run(prompt).strip()
        
        predictions.append({
            'Id': row.get('Id', 'Unknown'),
            'Prediction': prediction,
            'MostCommonTarget': most_common_target,
            'VoteCount': vote_count,
            'TotalVotes': len(target_values)
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

def extract_target_values(similar_items: str) -> List[str]:
    # This function should parse the similar_items string and extract the target values
    # The implementation will depend on the exact format of the similar_items string
    # Here's a placeholder implementation:
    target_values = []
    for item in similar_items.split('\n'):
        if 'target=' in item:
            target_values.append(item.split('target=')[1].strip())
    return target_values

def setup_retrieval_and_prediction(vectorstore, parameters: Parameters):
    dataset_name = parameters.dataset_name
    
    llm = get_llm(parameters)
    search_tool = create_search_tool(vectorstore, dataset_name)
    prompt_template = create_prompt_template(dataset_name)
    agent = create_agent(search_tool, llm)

    return llm, search_tool, prompt_template, agent, process_and_predict


