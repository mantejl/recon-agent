from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Load .env from repo root 
load_dotenv(Path(__file__).resolve().parent / ".env")

# just a test to check if LLM calls are workign
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
answer = llm.invoke([HumanMessage(content="Reply with exactly one word: OK.")])
print(answer.content)
