from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 128,
        "do_sample": False,
        "return_full_text": False,
    },
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate.from_template("""
Use only the verified information below. Return exactly 5 numbered facts.
Do not include an introduction, conclusion, or any unverified claim.

Verified information:
- Kolkata Knight Riders (KKR) is an Indian Premier League franchise.
- KKR was founded in 2008.
- Its home venue is Eden Gardens, Kolkata.
- KKR won IPL titles in 2012, 2014, and 2024.
- Knight Riders Sports Private Limited owns KKR.

Topic: {topic}
""")

chain = prompt | model | StrOutputParser()

print(chain.invoke({"topic": "Kolkata Knight Riders"}))