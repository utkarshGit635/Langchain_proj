from pathlib import Path
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

loader = PyPDFLoader("Unit-1 Training Neural Networks.pdf")

docs = loader.load()
print(len(docs))