from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    provider="featherless-ai",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=1024,
    temperature=0.2,
    repetition_penalty=1.05,
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="summary", description="A brief summary of the topic"),
    ResponseSchema(name="detailed_report", description="A detailed report on the topic"),
    ResponseSchema(name="key_points", description="Key points related to the topic")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Write a detailed report on the following topic: {topic} \n {format_instructions}",
    input_variables=["topic"],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

chain = template | model | parser

final_result = chain.invoke({"topic": "ICC Cricket World Cup 2023"})

print(final_result)