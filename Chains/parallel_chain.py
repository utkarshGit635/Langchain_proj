from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm1 = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 256,
        "do_sample": False,
        "return_full_text": False,
    },
)

llm2 = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 256,
        "do_sample": False,
        "return_full_text": False,
    },
)

model1 = ChatHuggingFace(llm=llm1)
model2 = ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(
    template = 'Generate short and simple notes on the following topic: {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'Generate 5 short questions and their answers on the following topic: {topic}',
    input_variables = ['topic']
)

prompt3 = PromptTemplate(
    template = 'Merge the following notes and questions into a single document: {notes} {quiz}',
    input_variables = ['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quiz": prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

topic = """
The 2023 ICC Men’s Cricket World Cup was held in India from 5 October to 19 November 2023. It was the 13th edition of the ODI World Cup and the first to be hosted entirely by India. Ten teams participated: India, Australia, England, New Zealand, Pakistan, South Africa, Sri Lanka, Bangladesh, Afghanistan, and the Netherlands. The tournament used a round-robin format, meaning each team played every other team once before the top four sides advanced to the semi-finals.

India had an outstanding league stage, winning all nine of its matches. The Indian side performed strongly in both batting and bowling, with Rohit Sharma providing aggressive starts, Virat Kohli scoring heavily, and bowlers such as Mohammed Shami, Jasprit Bumrah, and Ravindra Jadeja making major contributions. Kohli had a historic tournament, finishing as the leading run-scorer and surpassing Sachin Tendulkar’s record for the most centuries in ODI cricket.

South Africa also had an impressive campaign, producing several dominant batting performances. Quinton de Kock scored many runs, while the team’s pace attack remained dangerous throughout the competition. Australia had an uneven beginning, losing its first two league matches, but recovered strongly and built a long winning streak. New Zealand, another consistent World Cup team, also qualified for the semi-finals.

Afghanistan was one of the most admired teams of the tournament. It defeated several strong opponents, including England, Pakistan, and Sri Lanka, and showed major progress in international cricket. The Netherlands also produced memorable moments, including a surprising victory over South Africa.

The semi-finals featured India against New Zealand and Australia against South Africa. India defeated New Zealand in Mumbai, with Virat Kohli scoring his 50th ODI century and Mohammed Shami taking seven wickets. Australia defeated South Africa in a tense match at Kolkata, earning a place in the final.

The final was played at Narendra Modi Stadium in Ahmedabad on 19 November 2023. India batted first and scored 240 runs. Australia’s bowling attack, especially Mitchell Starc, restricted India after a promising beginning. In response, Australia lost early wickets, but Travis Head played a brilliant innings of 137 runs. Marnus Labuschagne supported him with a patient partnership. Australia won the match by six wickets and became ODI World Cup champions for the sixth time.

Australia’s victory demonstrated its ability to perform under pressure in major tournaments. Travis Head was named Player of the Match in the final, while Virat Kohli was named Player of the Tournament for his exceptional batting performances. The 2023 World Cup was remembered for India’s dominant league campaign, Kohli’s record-breaking achievements, Afghanistan’s rise, and Australia’s successful comeback after a difficult start.
"""

result = chain.invoke({"topic": topic})

print(result)