from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful {role} assistant."),
    ("user", "Explain the difference between {topic1} and {topic2}.")
])

prompt = chat_template.invoke({
    "role": "concise",
    "topic1": "langchain",
    "topic2": "langgraph"
})
