import os
import streamlit as st
from langchain.llms import Replicate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

class StreamHandler(BaseCallbackHandler):
    def __init__(self, initial_text=""):
        self.ai = st.chat_message("ai")
        with self.ai:
            self.container = st.empty()
        self.text = initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.write(self.text)

def write_ai(text):
    ai = st.chat_message("ai")
    ai.write(text)

def write_human(text):
    human = st.chat_message("human")
    human.write(text)

def update_messages():
    if  "messages" not in st.session_state:
        st.session_state.messages = []
    else:
        for message in st.session_state.messages:
            write_human(message["Human"])
            write_ai(message["AI"])

def current_memory():
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages= True)
    for message in st.session_state.messages:
        memory.save_context({"Human": message["Human"]}, {"AI": message["AI"]})
    return memory

def prompt_template():
    template = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot having a conversation with a human."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{text}")
    ])
    return template

def return_llm():
    llm = Replicate(
    streaming = True,
    callbacks=[StreamHandler()],
    model="meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d",
    model_kwargs={"temperature": 0.75, "max_length": 1500, "top_p": 1},
    verbose= True
    )
    return llm

def return_llm_chain():
    memory = current_memory()
    llm = return_llm()
    template = prompt_template()
    chain = LLMChain(llm=llm, prompt=template, memory = memory)
    return chain

def return_ai_response(prompt):
    chain = return_llm_chain()
    response = chain.predict(text = prompt)
    return response

st.set_page_config(page_title = "Chatbot",layout='centered')
st.title("Chatbot")
write_ai("Hello human. How can I help you?")
update_messages()

API_TOKEN = st.sidebar.text_input("Replicate API Token", type="password")

prompt = st.chat_input("Say something")

if prompt:
    if API_TOKEN:
        os.environ["REPLICATE_API_TOKEN"] = API_TOKEN
        try:
            write_human(prompt)
            response = return_ai_response(prompt)
            #write_ai(response)
            st.session_state.messages.append({"Human":prompt,"AI":response})
        except Exception as e:
            st.error(str(e))
        #response = "response"
    else:
        st.warning("NO API TOKEN IS PROVIDED")

    
