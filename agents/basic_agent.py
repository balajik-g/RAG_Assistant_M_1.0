#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import os

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise EnvironmentError("Environment variable 'OPENROUTER_API_KEY' is not set. Please set it to proceed.")

llm = ChatOpenAI(
    openai_api_key=api_key,  # Set this as env variable
    openai_api_base="https://openrouter.ai/api/v1",  # OpenRouter's endpoint
    model="mistralai/mixtral-8x7b-instruct",          # or use other models like gpt-4, llama-3, etc.
    temperature=0.2
)

def run_agent(prompt):
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content
