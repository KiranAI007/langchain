import os
from constants import openai_key
from langchain.llms import openai

import streamlit as st

# setting up the openai environment
os.environ["OPENAI_API_KEY"] = openai_key

# streamlit framework
st.title('Langchain demo with OPENAI API')
input_text = st.text_input('Search the text that you want')

# OPENAI LLMs
llm = openai.OpenAI()

if input_text:
    st.write(llm(input_text))