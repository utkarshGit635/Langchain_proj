import os

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

os.environ['HF_HOME'] = 'E:/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 128, "temperature": 0.7}
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("Who is the Prime Minister of Israel?")
print(result.content)