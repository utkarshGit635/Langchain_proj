from pathlib import Path
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

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

prompt_template = PromptTemplate(
    template="Write a summary of the following text:\n\n{input_text}",
    input_variables=["input_text"]
)

parser = StrOutputParser()

loader = TextLoader(Path(__file__).with_name("random.txt"), encoding="utf8")

docs = loader.load()
print(type(docs))

print(len(docs))
print(docs[0].metadata)

chain = prompt_template | model | parser

print(chain.invoke({"input_text": docs[0].page_content}))