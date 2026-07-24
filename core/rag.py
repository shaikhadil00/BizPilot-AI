from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

from core.config import GROQ_API_KEY
from core.config import LLM_MODEL


def build_rag(vectordb):

    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 20
        }
    )

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=LLM_MODEL,
        temperature=0,
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )

    return qa