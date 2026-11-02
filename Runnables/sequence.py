from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence

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
    template=(
        "Translate the following text from {input_language} to {output_language}. "
        "Return only the translation, without explanations or additional text.\n\n"
        "Text: {input_text}"
    ),
    input_variables=["input_language", "output_language", "input_text"]
)

parser = StrOutputParser()

chain = RunnableSequence(prompt_template, model, parser)

print(
    chain.invoke(
        {
            "input_language": "English",
            "output_language": "French",
            "input_text": "Where is the nearest train station?",
        }
    )
)
