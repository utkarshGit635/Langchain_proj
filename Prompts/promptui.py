import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

huggingface_token = (
    os.getenv("HUGGINGFACEHUB_API_TOKEN")
    or os.getenv("HF_TOKEN")
)

if not huggingface_token:
    st.error("Set HUGGINGFACEHUB_API_TOKEN or HF_TOKEN before starting the app.")
    st.stop()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    provider="featherless-ai",
    huggingfacehub_api_token=huggingface_token,
    temperature=0.7,
    max_new_tokens=1024,
)

model = ChatHuggingFace(llm=llm)

st.header("Chat with TinyLlama")

paper_input = st.selectbox(
    "Select research paper to discuss:",
    [
        "Select ...",
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "GPT-3: Language Models are Few-Shot Learners",
    ],
)

style_input = st.selectbox(
    "Select style of discussion:",
    ["Select ...", "Academic", "Casual", "Professional"],
)

length_input = st.selectbox(
    "Select length of discussion:",
    ["Select ...", "Short", "Medium", "Long"],
)

template = load_prompt("paper_explainer.json")

if st.button("Start Discussion"):
    if "Select ..." in (paper_input, style_input, length_input):
        st.warning("Please select all options before starting the discussion.")
    else:
        prompt = template.format(
            paper=paper_input,
            style=style_input.lower(),
            length=length_input.lower(),
        )

        with st.spinner("Generating discussion..."):
            try:
                response = model.invoke(prompt)
                st.markdown("### Discussion")
                st.markdown(response.content)
            except Exception as exc:
                st.error(f"Could not generate a response: {exc}")