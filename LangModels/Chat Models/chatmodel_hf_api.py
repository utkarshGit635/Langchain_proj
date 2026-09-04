import os
from pathlib import Path

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
if not huggingface_token:
    raise RuntimeError(
        "Set HUGGINGFACEHUB_API_TOKEN or HF_TOKEN in LangModels/.env"
    )

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    provider="featherless-ai",
    huggingfacehub_api_token=huggingface_token,
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("What is the capital of India?")
print(result.content)