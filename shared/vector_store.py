import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def build_vector_store(file_path: str):
    print(f"Loading document: {file_path}")
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding='utf-8')
    
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    return vectorstore.as_retriever(search_kwargs={"k": 4})
