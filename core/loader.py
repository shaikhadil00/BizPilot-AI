import os

from langchain_community.document_loaders import UnstructuredExcelLoader

from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
)


def load_document(file_path):

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)

    elif ext == ".csv":
        loader = CSVLoader(file_path)

    elif ext in [".xlsx", ".xls"]:
        loader = UnstructuredExcelLoader(file_path)

    elif ext == ".docx":
        loader = UnstructuredWordDocumentLoader(file_path)

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return loader.load()