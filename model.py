from langchain_community.chat_models import JinaChat
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
import os
from dotenv import load_dotenv
from datasets import load_dataset
import import_ipynb


#########################KNOWLEDGE BASE#######################################
ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")

load_dotenv(r'C:\Users\91982\Desktop\Taskformer\.env')


j = os.getenv('J')
print(j)
chat = JinaChat(temperature=0, jinachat_api_key=j)

messages = [
    SystemMessage(
        content="You are a helpful medical assistant that gives advice on any and all medical related queries. You try to advise someone if the need immediate help and need to go to the doctor."
    ),
    HumanMessage(
        content="Hey, I've been feeling very tired and drowsy for the past week. Is that normal?"
    ),
]
c=chat(messages)
print(c.content)