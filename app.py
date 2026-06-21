import os

import streamlit as st
# from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from db_logic import get_db_implementation, DB_PATH

# load_dotenv() #stored in .env
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') #loading OPEN AI key

st.sidebar.header("🔑 OpenAI API Key")
st.session_state["OPENAI_API_KEY"] = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if not st.session_state["OPENAI_API_KEY"]:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()
    
def render_streamlit_heading():
    st.set_page_config(page_title="Chat with PDF 📄", page_icon="🤖", layout="wide")

    st.title("📄 Chat with Machine Learning ")
    st.markdown("Ask any question based on the Machine learning content.")

def get_chat_prompt_template():
    ## Design ChatPrompt Template
    ## We are defining a prompt for LLM on how should it behave.

    # system prompt to define the behaviour of the LLM
    prompt = ChatPromptTemplate.from_template(""" 
    You can greet the user and make friendly response.
    Answer the following question based only on provided context.
    The Question  may include previous history conversation to get you idea of previous step.
    think step by step before providing a detailed answer.
    If question cannot be found from the context, just say you dont know, dont try to make up.
    We need to strictly stick to the context.
    <context>
    {context}
    </context>

    Question:{input}

    """)
    
    return prompt

def get_llm(model="gpt-3.5-turbo"):
    # Learn about temperature, its range (0,1), high temperature will result into more random response or out-of-the-box thinking.
    llm = ChatOpenAI(model=model, temperature = 0.7, api_key = st.session_state["OPENAI_API_KEY"])  # or model="gpt-4"
    return llm

def get_answer(context, rag):
    response = rag.invoke({"input" : str(context)})
    return response['answer']


render_streamlit_heading()
# Load RAG components
with st.spinner("Loading vector store and LLM..."):
    db = get_db_implementation(api_key = st.session_state['OPENAI_API_KEY'])
    prompt =  get_chat_prompt_template()
    llm = get_llm()
    generator = create_stuff_documents_chain(llm, prompt) #its implentation has been already defined in langchain library
    retriever = db.as_retriever() # Here we are defining the db as retreiver, i.e. db should act as a knowledge base or vector store that can be retrieved from.
    rag = create_retrieval_chain(retriever, generator)

print("Session state: ", st.session_state) # {}
if "history" not in st.session_state.keys():
    st.session_state["history"] = []
print("Session state: ",st.session_state) # {'history': []}

if ("user", None) in st.session_state["history"]:
    st.session_state["history"].remove(('user', None))

# Chat input
question = st.chat_input("Ask a question from the Machine learning...") # streamlit run app.py
st.session_state["history"].append(("user", question))
if question:
    with st.spinner("Thinking..."):
        history_len = 5 if len(st.session_state['history']) >= 5 else len(st.session_state['history']) 
        context = st.session_state['history'][-history_len:]
        answer = get_answer(context, rag)
        
        # if len(st.session_state['history']) >= 5:
        #     context = st.session_state['history'][-5:]
        #     answer = get_answer(context, rag)
        # else:
        #     answer = get_answer(question, rag)
        # answer = "this is default answer!"
        st.session_state["history"].append(("assistant", answer))
        
        for role, msg in st.session_state['history']:  

            if role == "user":
                st.markdown(f"""
                    <div style='text-align: right;'>
                        <div style='display: inline-block; background-color: #0000FF; color: black; padding: 10px 15px;
                                    border-radius: 12px; margin: 8px 0; max-width: 80%;'>
                            {msg}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            elif role == "assistant":
                st.markdown(f"""
                    <div style='text-align: left;'>
                        <div style='display: inline-block; background-color: #F1F0F0; color: black; padding: 10px 15px;
                                    border-radius: 12px; margin: 8px 0; max-width: 80%;'>
                            {msg}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            

# print("Session starts...")
# for i in range(3):
#     question = input("Please Input your Question: ") #python app.py
#     if question == "exit":
#         break
    
#     answer = get_answer(question, rag)
#     print("ANSWER: ", answer)

print("Your session ends!")


