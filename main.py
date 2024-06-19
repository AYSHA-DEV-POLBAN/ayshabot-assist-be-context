from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
from langchain_community.callbacks import get_openai_callback
import io
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import CharacterTextSplitter
# from PyPDF2 import PdfReader
# import io
import requests
import os
# from langchain_openai import OpenAIEmbeddings
import chromadb
from chromadb import Client
from chromadb.config import Settings
import uuid
from urllib.parse import urlparse

app = FastAPI()

headers = {
    'Content-Type': 'application/json'  # Pastikan header ini jika Anda mengirim data JSON
}

load_dotenv()

knowledge_base = None

embedding_function = OpenAIEmbeddings()

chroma_client = chromadb.HttpClient(host='127.0.0.1', port=8000)    
persist_directory="/db_knowledge_base/chroma_db"
# persist_directory="/content/db_knowledge_base/chroma_db"
collection_name="informasi_aysha"
# vectorstore = Chroma(client=chroma_client,collection_name=agent, embedding_function=embeddings)
# vector_store = Chroma(collection_name = collection_name, embedding_function = embedding_function, persist_directory = persist_directory, client=chroma_client)
vector_store = Chroma(client = chroma_client, collection_name = collection_name, embedding_function = embedding_function)
# vector_store = Chroma(collection_name = collection_name, embedding_function = embedding_function, client=chroma_client)
    
class Question(BaseModel):
    question: str


@app.post("/add_document/")
async def add_document(id_information: int, file_path: str, last_chunks_ids: int):
    # print(id_information)
    # print(file_path)
    # print(last_chunks_ids)
    loader = PyPDFLoader(file_path)
    documents = loader.load_and_split()
    # print(documents)

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)

    chunks_document = text_splitter.split_documents(documents)
    # print(chunks_document)
    # len(chunks_document)


    last_chunk_ids = last_chunks_ids
    next_chunks_ids = last_chunk_ids + 1

    ids = [str(i) for i in range(next_chunks_ids, next_chunks_ids + len(chunks_document))]

    # print(ids)

    # array ids terkahir di chunk baru
    # print(ids[-1])

    # add_data = vector_store.from_documents(chunks_document, embedding_function, ids=ids, persist_directory=persist_directory, client=chroma_client)
    try:
        add_data = vector_store.from_documents(chunks_document, embedding_function, ids=ids, client=chroma_client, collection_name=collection_name)

        url = f'http://localhost:8001/api/v1/information/edit_chroma/{id_information}'  # Ganti dengan URL endpoint API yang ingin Anda gunakan
        payload = {
            'chunk_total': len(chunks_document),
            'chunk_ids_min': ids[0],
            'chunk_ids_max': ids[-1]
        }
        
        try:
            response = requests.put(url, json=payload, headers=headers)
            response.raise_for_status()  # Ini akan menimbulkan error untuk status code 4xx/5xx
            data = response.json()  # Mengurai JSON dari respons
            print(data)
        except requests.exceptions.HTTPError as errh:
            print(f"HTTP Error: {errh}")
        except requests.exceptions.ConnectionError as errc:
            print(f"Error Connecting: {errc}")
        except requests.exceptions.Timeout as errt:
            print(f"Timeout Error: {errt}")
        except requests.exceptions.RequestException as err:
            print(f"Request Exception: {err}")

        return {"documents": documents, "chunks_document": chunks_document, "chunk_total": len(chunks_document), "ids": ids, "chunk_ids_min":ids[0], "chunk_ids_max":ids[-1], "id_information": id_information}
    except Exception as e:
        return {"success": False, "error": str(e)}



    # try:
        # global knowledge_base  # Gunakan klausa global
        # load_dotenv()
        # contents = await file.read()
        # pdf_reader = PdfReader(io.BytesIO(contents))
        # text = ""
        # for page in pdf_reader.pages:
        #     text += page.extract_text()
        
        # split into chunks
        # text_splitter = CharacterTextSplitter(
        #     separator="\n",
        #     chunk_size=1000,
        #     chunk_overlap=200,
        #     length_function=len
        # )
        
        # chunks = text_splitter.split_text(text)   
        
        # create embedding
        # embeddings = OpenAIEmbeddings() #embedding model
        # knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        
        # look_embeddings = embeddings.embed_documents(chunks)
        # embedded_query = embeddings.embed_query(chunks)
        # embedded_query[:5]
        
        # return {"filename": file.filename, "content": text, "chunks": chunks, "len_array_embedding": len(look_embeddings), "len_embedding": len(look_embeddings[0]), "embedding" : look_embeddings}
# @app.get("/test_2/")
# async def test_2():
#     return {"message": "ok"}

# @app.get("/test_chroma/")
# async def test_chroma():
#     id_information = 2
#     url = f'http://localhost:8001/api/v1/information/edit_chroma/{id_information}'  # Ganti dengan URL endpoint API yang ingin Anda gunakan
#     payload = {
#         'chunk_total': 25,
#         'chunk_ids_min': 1,
#         'chunk_ids_max': 25
#     }

#     try:
#         response = requests.put(url, json=payload, headers=headers)
#         response.raise_for_status()  # Ini akan menimbulkan error untuk status code 4xx/5xx
#         data = response.json()  # Mengurai JSON dari respons
#         print(data)
#     except requests.exceptions.HTTPError as errh:
#         print(f"HTTP Error: {errh}")
#     except requests.exceptions.ConnectionError as errc:
#         print(f"Error Connecting: {errc}")
#     except requests.exceptions.Timeout as errt:
#         print(f"Timeout Error: {errt}")
#     except requests.exceptions.RequestException as err:
#         print(f"Request Exception: {err}")

@app.post("/get_document_by_ids/")
async def get_document_by_ids(ids: str):
    # print(ids)
    get_data = vector_store.get(ids=[ids])
   
    return {"get_data": get_data}

@app.post("/delete_document_by_ids/")
async def delete_document_by_ids(id_information:int, chunk_ids_min: int, chunk_ids_max: int):
    ids_chunk_for_delete = [str(i) for i in range(chunk_ids_min, chunk_ids_max+1)]
    vector_store.delete(ids=ids_chunk_for_delete)

    id_information = id_information
    url = f'http://localhost:8001/api/v1/information/delete_chroma/{id_information}'  # Ganti dengan URL endpoint API yang ingin Anda gunakan
    payload = {
        'chunk_total': None,
        'chunk_ids_min': None,
        'chunk_ids_max': None
    }

    try:
        response = requests.put(url, json=payload, headers=headers)
        response.raise_for_status()  # Ini akan menimbulkan error untuk status code 4xx/5xx
        data = response.json()  # Mengurai JSON dari respons
        print(data)
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"Request Exception: {err}")


@app.get("/delete_collection/")
async def delete_collection():
    delete_collection = vector_store.delete_collection()

# @app.post("/uploadfile/")
# async def create_upload_file(file: UploadFile = File(...)):
    # try:
        # global knowledge_base  # Gunakan klausa global
        # load_dotenv()
        # contents = await file.read()
        # pdf_reader = PdfReader(io.BytesIO(contents))
        # text = ""
        # for page in pdf_reader.pages:
        #     text += page.extract_text()
        
        # split into chunks
        # text_splitter = CharacterTextSplitter(
        #     separator="\n",
        #     chunk_size=1000,
        #     chunk_overlap=200,
        #     length_function=len
        # )
        
        # chunks = text_splitter.split_text(text)   
        
        # create embedding
        # embeddings = OpenAIEmbeddings() #embedding model
        # knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        
        # look_embeddings = embeddings.embed_documents(chunks)
        # embedded_query = embeddings.embed_query(chunks)
        # embedded_query[:5]
        
        # return {"filename": file.filename, "content": text, "chunks": chunks, "len_array_embedding": len(look_embeddings), "len_embedding": len(look_embeddings[0]), "embedding" : look_embeddings}

@app.post("/question_answer/")
async def question_answer(question: Question):
    # if knowledge_base is None:
    #     return {"response": "Error! Knowledge base is not available yet. Please upload a file first."}
    
    # load_dotenv()
    user_question = question.question
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    
    # Fetch documents from knowledge base
    # docs = knowledge_base.similarity_search(user_question)
    docs = vector_store.similarity_search(user_question)
    
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)
        print(cb)
        
    return {"response": response, "docs": docs}
