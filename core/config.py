import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# First try Streamlit Secrets
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "llama-3.3-70b-versatile"

CHROMA_PATH = "vector_db"