from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
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

parser = JsonOutputParser()

template = prompt = PromptTemplate(
        template = "Give me a name, age and address of a fictional person in JSON format \n {format_instructions}",
        input_variables = [],
        partial_variables = {'format_instructions': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({})

print(result)