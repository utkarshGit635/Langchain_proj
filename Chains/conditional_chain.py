from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 256,
        "do_sample": False,
        "return_full_text": False,
    },
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(..., description="The sentiment of the text")
    confidence: float = Field(..., description="The confidence score of the sentiment classification")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template = 'Classify the sentiment of the following text as positive, negative, or neutral: {text}\n {format_instructions}',
    input_variables = ['text'],
    partial_variables = {'format_instructions': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template = 'Write an appropriate response to this positive review: {feedback}',
    input_variables = ['feedback']
)

prompt3 = PromptTemplate(
    template = 'Write an appropriate response to this negative review: {feedback}',
    input_variables = ['feedback']
)

prompt4 = PromptTemplate(
    template = 'Write an appropriate response to this neutral review: {feedback}',
    input_variables = ['feedback']
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | parser),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser),
    (lambda x: x.sentiment == "neutral", prompt4 | model | parser),
    RunnableLambda(lambda x: "Invalid sentiment classification")
)

chain = classifier_chain | branch_chain

result = chain.invoke({"text": "I hate the new update. It is full of bugs and crashes all the time."})

print(result)