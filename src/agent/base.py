"""
LLM retrieval module
"""

from langchain.agents import AgentType, Tool, initialize_agent
from langchain.prompts import PromptTemplate

from src.configs import Parameters

OPENAI_MODELS = ["gpt", "gpt-4"]


class BaseAgent:
    """Base class for the agent"""

    def __init__(
        self, parameters: Parameters, vectorstore, llm, prompt: PromptTemplate
    ):
        self.parameters = parameters
        self.vectorstore = vectorstore
        self.llm = llm
        self.prompt = prompt

    def create_agent(self):
        """Create a search agent"""
        tools = [
            Tool(
                name="Make Prediction",
                func=lambda x: x,  # tool doesn't do anything, it's just a placeholder
                description="Use this tool to make your final prediction based on "
                "the given information. The prediction must be either 0 or 1.",
            )
        ]

        return initialize_agent(
            tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=1,  # Limit to one iteration to force a direct answer
            prompt=self.prompt,
        )
