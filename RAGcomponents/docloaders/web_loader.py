from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 512,
        "do_sample": False,
        "return_full_text": False,
    }
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="Answer the following question based on the content of the webpage:\n\n{input_text}",
    input_variables=["input_text"]
)

parser = StrOutputParser()

url = "https://github.com/utkarshGit635"
loader = WebBaseLoader(url)
docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({'Question': 'What is the purpose of this GitHub repository?', 'input_text': docs[0].page_content})
