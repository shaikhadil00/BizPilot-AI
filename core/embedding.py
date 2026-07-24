from langchain_community.embeddings import HuggingFaceEmbeddings
from core.config import EMBEDDING_MODEL


def load_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )

    return embeddings