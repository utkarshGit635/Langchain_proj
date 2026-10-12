from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    provider="featherless-ai",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=256,
    temperature=0.2,
    repetition_penalty=1.05,
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template = "Write a detailed report on the following topic: {topic}",
    input_variables = ["topic"]
)

template2 = PromptTemplate(
    template = "Write a 5 line summary on the following topic: \n {text}",
    input_variables = ["text"]
)

prompt1 = template1.invoke({"topic": "Black holes and their impact on space-time."})

result = model.invoke(prompt1)

prompt2 = template2.invoke({"text": result.content})

final_result = model.invoke(prompt2)

print("Final Result:", final_result.content)