from langchain_community.chat_models import JinaChat
from langchain_core.messages import HumanMessage, SystemMessage
import streamlit as st
from datasets import load_dataset
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import pinecone
import google.generativeai as genai
import PIL.Image
from model import full_data, id_to_text, model, index

from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)






st.cache_data()
#########################KNOWLEDGE BASE#######################################
with st.sidebar:
    st.header("Enter your Jina Chat API Key here")
    j = st.text_input("Jina Chat API Key", key="chatbot_api_key", type="password")
    st.text("Make sure to enter your correct API Key")

# load_dotenv(r'C:\Users\91982\Desktop\Taskformer\.env')
# j = os.getenv('J')
gen_model = genai.GenerativeModel('gemini-1.5-pro')

def query_documents(query_text, top_k=5):

    query_embedding = model.encode([query_text])[0]
    query_results = index.query(query_embedding.tolist(), top_k=top_k)
    matching_ids = [match['id'] for match in query_results['matches']]
    return matching_ids

def generate_response(messages, jinachat_api_key):
    chat = JinaChat(temperature=0, jinachat_api_key=j)
    response = chat(messages)
    return response.content

def summarize_image_with_gemini(image, prompt):
    # Use the generative model to summarize the image
    response = gen_model.generate_content([prompt, image])
    summary = response.text
    return summary

image_summary_prompt = """
You are a helpful medical assistant that gives medical summarizations of images.
These summaries will be embedded and used to retrieve the raw image.
Describe concisely the characteristics of the image along with what diseases it could entail. Analyse properly which part of the human body it is and list all possible diseases; don't just give one.
Describe the characteristics and the predicted disease. If it seems too bad, advise the user to go to a doctor.
"""

st.title("💬 MedChat")
st.caption("🚀 A Streamlit medical chatbot powered by JinaChat")

uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
text_input = st.text_input("Enter your query or description")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    similar_ids = query_documents(prompt)
    context = "\n".join([id_to_text.get(id, '') for id in similar_ids])
    
    # Prepare messages for JinaChat
    messages = [
        SystemMessage(
            content=f"""You are a helpful medical assistant that gives advice on any and all medical related queries. You try to advise someone if the need immediate help and need to go to the doctor. You use the context to learn.
            Context={context}"""
        ),
        HumanMessage(content=prompt)
    ]
    

    jinachat_api_key = j
    response = generate_response(messages, jinachat_api_key)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
