import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv(Path(__file__).resolve().parent / ".env")

token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
if not token:
    raise RuntimeError("Set HUGGINGFACEHUB_API_TOKEN or HF_TOKEN in .env")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    provider="featherless-ai",
    huggingfacehub_api_token=token,
    max_new_tokens=256,
    temperature=0.2,
    repetition_penalty=1.05,
)

model = ChatHuggingFace(llm=llm)

chat_history = []

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))

    if user_input.lower() in {"exit", "quit"}:
        print("Exiting the chat. Goodbye!")
        break

    response = model.invoke([
        SystemMessage(content="You are a helpful, concise assistant. Answer the user's question directly."),
        *chat_history,
    ])
    chat_history.append(response)
    print(f"Bot: {response.content}")

print(chat_history)