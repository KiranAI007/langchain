import os
from constants import openai_key
from langchain.llms import openai
from langchain import prompts
from langchain.chains import LLMChain

from langchain.chains import SequentialChain

from langchain.memory import ConversationBufferMemory

import streamlit as st

# setting up the openai environment
os.environ["OPENAI_API_KEY"] = openai_key

# streamlit framework
st.title('Celebrity search result')
input_text = st.text_input('Search the text that you want')

# prompt template - 1
first_input_prompt = prompts.PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)

# memory
person_memory = ConversationBufferMemory(input_key='name',memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')

# OPENAI LLMs
llm = openai.OpenAI()
# chain 1
chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key='person')

# prompt template - 2
second_input_prompt = prompts.PromptTemplate(
    input_variables=['person'],
    template="when was {person} born"
)
# chain 2
chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key='dob')

# prompt template - 3
third_input_prompt = prompts.PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events happends around {dob} in the world"
)
# chain 3
chain3 = LLMChain(llm=llm, prompt=third_input_prompt, verbose=True, output_key='description')

# combine all the chain 
parent_chain = SequentialChain(
    chains=[chain,chain2, chain3], input_variables=['name'], output_variables=['person','dob','description'], verbose=True
)

if input_text:
    st.write(parent_chain({'name':input_text}))

    with st.expander('Person Name'): 
        st.info(person_memory.buffer)

    with st.expander('Major Events'): 
        st.info(descr_memory.buffer)