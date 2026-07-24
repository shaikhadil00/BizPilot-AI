import gc

from langchain_chroma import Chroma

from core.embedding import load_embeddings
from core.config import CHROMA_PATH


def create_vector_db(chunks):

    if not chunks:
        raise ValueError("No document chunks found.")

    embeddings = load_embeddings()

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )

    return vectordb


def load_vector_db():

    embeddings = load_embeddings()

    vectordb = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )

    return vectordb


# -------------------------------------------------
# FIXED: Explicitly closes native persistent file handles
# -------------------------------------------------

def close_vector_db(vectordb):

    try:

        if vectordb is not None:
            
            # 1. Target the underlying native Chroma persistent clients if exposed
            if hasattr(vectordb, "_client") and vectordb._client is not None:
                try:
                    # In newer chromadb builds, .close() drops background thread references
                    vectordb._client.close()
                except Exception:
                    pass
                    
            # 2. Wipe the core LangChain references
            del vectordb
            
            # 3. Force the Python garbage collector to reclaim leaked handle references
            gc.collect()

    except Exception:

        pass
