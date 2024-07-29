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
from PIL import Image

from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)






st.cache_data()
#########################KNOWLEDGE BASE#######################################
with st.sidebar:
    st.title("Before using ChatMed, create your own API keys for free")
    st.header("Enter your Jina Chat API Key here")
    j = st.text_input("Jina Chat API Key", key="chatbot_api_key", type="password")
    st.text("Make sure to enter your correct JinaChat API Key")
    st.header("Enter your Gemini API Key here")
    g= st.text_input("Gemini API Key", key="google", type="password")
    st.text("Make sure to enter your correct Gemini API Key")


# load_dotenv(r'C:\Users\91982\Desktop\Taskformer\.env')
# j = os.getenv('J')
gen_model = genai.GenerativeModel('gemini-1.5-pro')
genai.configure(api_key=g)

def query_documents(query_text, top_k=5):

    query_embedding = model.encode([query_text])[0]
    query_results = index.query(query_embedding.tolist(), top_k=top_k)
    matching_ids = [match['id'] for match in query_results['matches']]
    return matching_ids

def generate_response(messages):
    chat = JinaChat(temperature=0, jinachat_api_key=j)
    response = chat(messages)
    return response.content

def summarize_image_with_gemini(image, prompt):
    response = gen_model.generate_content([prompt, image])
    summary = response.text
    return summary

image_summary_prompt = """
You are a helpful medical assistant that gives medical summarizations of images.
These summaries will be embedded and used to retrieve the raw image.
Describe concisely the characteristics of the image along with what diseases it could entail. Analyse properly which part of the human body it is and list all possible diseases; don't just give one.
Describe the characteristics and the predicted disease. If it seems too bad, advise the user to go to a doctor.
"""

st.title("💬 ChatMed")
st.caption("🚀 A Streamlit medical chatbot powered by JinaChat and Gemini")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if prompt := st.chat_input("How are you feeling?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        summarized_text = summarize_image_with_gemini(image, image_summary_prompt)
        prompt = summarized_text + "\n" + prompt

    similar_ids = query_documents(prompt)
    context = "\n".join([id_to_text.get(id, '') for id in similar_ids])

    messages = [
        SystemMessage(
            content=f"You are a helpful medical assistant that gives advice on any and all medical related queries. You try to advise someone if they need immediate help and need to go to the doctor.\nContext={context}"
        ),
        HumanMessage(content=prompt)
    ]
    

    response = generate_response(messages)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(summarized_text+response)