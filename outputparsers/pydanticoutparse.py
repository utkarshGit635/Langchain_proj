from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
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

class PersonInfo(BaseModel):
    name: str = Field(..., description="The person's full name")
    age: int = Field(gt = 18, description="The person's age in years")
    city: str = Field(..., description="The person's city of residence")

parser = PydanticOutputParser(pydantic_object=PersonInfo)

template = PromptTemplate(
    template="Generate the name of a historical {place} war criminal, their age, and the city they lived in. \n {format_instructions}",
    input_variables=["place"],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({"place": "Germany"})
print(result)