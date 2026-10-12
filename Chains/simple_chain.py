from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 256,
        "temperature": 0.2,
        "do_sample": True,
        "repetition_penalty": 1.05,
    },
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template = "Generate 5 facts about {topic}.",
    input_variables = ["topic"]
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "KKR"})

print(result)

chain.get_graph().print_ascii()