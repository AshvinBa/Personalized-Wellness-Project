import streamlit as st

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------
# CONFIG
# -----------------------------
COLLECTION_NAME = "GENERAL_GUIDELINES"

# -----------------------------
# EMBEDDINGS (MUST MATCH INGESTION)

# -----------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

embedding_function=load_embeddings()
# -----------------------------
# LOAD CHROMA DB
# -----------------------------
@st.cache_resource
def load_chroma():
    return Chroma(
        persist_directory="./chroma",
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function
    )
chroma = load_chroma() 


# -----------------------------
# LLM (GENERATION ONLY)
# # -----------------------------

llm = ChatOllama(
    model="llama3:8b",
    temperature=0.2
)
# -----------------------------
# RAG FUNCTIONS
# # -----------------------------
# def retrieve_docs(query: str):
#     return chroma.similarity_search(query, k=5)

# def build_prompt(query, docs):
#     context = "\n\n".join(d.page_content for d in docs)
#     prompt = ChatPromptTemplate.from_template(PROMPT)
#     return prompt.format_messages(
#         query= query,
#         context=context
        
#     )
def run_rag(prompt):
    docs = chroma.similarity_search(prompt, k=3)
    context = "\n".join([d.page_content for d in docs])
    return context, llm.invoke(prompt + "\n\n" + context).content

# prompts.py

def build_rag_new_prompt(ml_output, chat_history, user_question=None):

    history_text = ""
    for q, a in chat_history:
        history_text += f"User: {q}\nAssistant: {a}\n"

    follow_up = f"\nUser Question: {user_question}" if user_question else ""

    return f"""
You are a wellness assistant.

Health status is already determined by a machine learning model.
DO NOT reclassify or calculate health metrics.

Health Status: {ml_output['user_status']}
Risk Factors: {", ".join(ml_output['risk_factors'])}

Conversation so far:
{history_text}

{follow_up}

Use the context below to:
- Answer the user's question
- Give general wellness advice
- Stay consistent with previous answers

If info is missing, say "I don't know".
"""


def build_rag_prompt(ml_output):

    return f"""
You are a wellness assistant.

Health status is already determined by a machine learning model.
DO NOT reclassify or calculate health metrics.

Health Status: {ml_output['user_status']}
Risk Factors: {", ".join(ml_output['risk_factors'])}

Use the context below to:
- Explain what these risk factors mean
- Give general wellness advice
- Suggest simple habit improvements

If info is missing, say "I don't know".
"""


def generate_answer(prompt):
    return llm.invoke(prompt).content


