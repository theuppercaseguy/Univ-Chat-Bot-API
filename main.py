from fastapi import FastAPI, Request,Body,HTTPException
from typing import Dict
from typing import List
from fastapi.responses import HTMLResponse,JSONResponse
from fastapi.templating import Jinja2Templates
from langchain.chains import ConversationalRetrievalChain
from langchain import HuggingFaceHub
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import os
from langchain.embeddings import HuggingFaceEmbeddings
import sqlite3
import pickle



# with open("cleanScrape.txt", "r", encoding="utf-8") as file:
#     content = file.read()

loader = TextLoader("scrapped_Data.txt")
docs = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs= text_splitter.split_documents(docs)


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
embeddings = HuggingFaceEmbeddings()


from langchain.chains import ConversationalRetrievalChain
from langchain import HuggingFaceHub
from langchain.vectorstores import Chroma



docsearch = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
repo_id="google/flan-t5-base"


qa = ConversationalRetrievalChain.from_llm(
    llm = HuggingFaceHub( repo_id=repo_id, model_kwargs={"temperature": 0, "max_length":512}),
    retriever=docsearch.as_retriever(),
    return_source_documents=True,
  )

import sys
history = []
past = []


app = FastAPI()


# Create a templates object using Jinja2Templates
templates = Jinja2Templates(directory="./templates")



@app.get("/",response_class=HTMLResponse)
async def root(request: Request):
   return templates.TemplateResponse("index.html",context={"request":request,})
  #return {"example":"example return value", "data":99}


def store_in_database(query: str, answer: str):
    conn = sqlite3.connect("chatbot_database.db")
    cursor = conn.cursor()

    # Store the question and answer in the chat_history table
    cursor.execute("INSERT INTO chat_history (question, answer) VALUES (?, ?)", (query, answer))

    conn.commit()
    conn.close()


def process_query(query: str):
    if not query:
        return None
    
    result = qa({'question': query, 'chat_history': history})
    history.append((query, result['answer']))
    past.append({"question": query, "answer": result['answer']})

    data = open(f"./history.txt", "a")
    data.write("Q: " + query + ". ANS: " + history[-1][1] + "\n")  # Write the last answer in the history
    data.close()
    store_in_database(query, result['answer'])

    return result['answer']


# Create a connection to the SQLite database
conn = sqlite3.connect("chatbot_database.db")
cursor = conn.cursor()


@app.get("/chat_me", response_class=HTMLResponse)
async def get_chat_me(request: Request, query: str = None):
    response_text = process_query(query)
    return templates.TemplateResponse("index.html", {"request": request, "response_text": response_text, "chat_history": past})
