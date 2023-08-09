from fastapi import FastAPI, Request,Body,HTTPException
from fastapi.responses import HTMLResponse,JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request
import json

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain import HuggingFaceHub
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import PromptTemplate
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware

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

class UserData:
    def __init__(self, users):
        self.users = users


template = """As a GIKI website chat bot, your goal is to provide accurate and helpful information about GIKI,
an educational institute located in Pakistan.

Instructions:
1. Greeting: If the user greets, respond with a greeting.
2. Answering Questions: To answer a question, go through your training data and use the following context:
    <ctx>
    {context} 
    </ctx>
    Question: {question}

3. Unknown Answers: If you don't know the answer, tell the user that you don't know.

4. User Name: If the user tells you their name, ask them for their phone number.

5. Phone Number: If the user tells you their phone number, ask them if they have any more questions.

Remember to avoid making up answers and always provide accurate information.

Example Usage:
- User: Hello
  ChatBot: Hi there! How can I assist you today?

- User: What courses does GIKI offer?
  ChatBot: <ctx> GIKI offers a wide range of courses in engineering, computer science, and more. </ctx> Question: What courses does GIKI offer?

- User: What is the capital of Pakistan?
  ChatBot: <ctx> Pakistan is a country in South Asia. </ctx> Question: What is the capital of Pakistan?

- User: My name is John
  ChatBot: Nice to meet you, John! Could you please provide your phone number?

- User: 123-456-7890
  ChatBot: Thank you, John! Do you have any more questions?

Remember to follow these instructions to provide the best assistance to the users. Good luck!
"""


docsearch = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
repo_id="google/flan-t5-base"

llm = HuggingFaceHub( repo_id=repo_id, model_kwargs={"temperature": 0, "max_length":512})

qa = ConversationalRetrievalChain.from_llm(
    llm = llm,
    retriever=docsearch.as_retriever(),
    return_source_documents=True,
  )

chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=docsearch.as_retriever(),
                chain_type_kwargs={
                "prompt": PromptTemplate(
                template=template,
                input_variables=["context","question"],
            ),
        },)

import sys
history = []
past = []


app = FastAPI()


# Create a templates object using Jinja2Templates
templates = Jinja2Templates(directory="./templates")

def read_database():
    try:
        with open("./database.json", "r") as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        return {"users": {}}

def write_database(data):
    with open("./database.json", "w") as file:
        json.dump(data, file)

def store_in_database(user_id: str, query: str, answer: str):
    data = read_database()
    if user_id not in data["users"]:
        data["users"][user_id] = []
    data["users"][user_id].append({"question": query, "answer": answer})
    write_database(data)

def get_user_history(user_id: str):
    data = read_database()
    if user_id in data["users"]:
        return UserData(users=data["users"])  # Pass the whole data["users"] as the argument
    else:
        return UserData(users={user_id: []})

@app.get("/",response_class=HTMLResponse)
async def root(request: Request):
   return templates.TemplateResponse("index.html",context={"request":request,})
  
default_user_id = "123"
def process_query(user_id: str, query: str, user_history: UserData):
    if not query:
        return None

    chat_history = [(entry["question"], entry["answer"]) for entry in user_history.users[user_id]]
    
    #result = qa({'question': query, 'chat_history': chat_history})
    result = chain.run(query)
    store_in_database(user_id, query, result)

    return result
    #return chat_history

@app.get("/chat_me", response_class=HTMLResponse)
async def get_chat_me(request: Request, query: str = None, user_id: str = default_user_id):

    print(f"Received User ID: {user_id}")

    user_history = get_user_history(str(user_id))
    
    response_text = process_query(user_id, query, user_history)

    return templates.TemplateResponse("index.html", {
        "user_id": user_id,
        "request": request, 
        "response_text": response_text, 
        "chat_history": user_history.users[user_id],  # Pass the chat history for the specific user
    })



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat")
async def chat(request_data:list):
    print(request_data)
    user_id = "123"  
    user_history = get_user_history(user_id)
    # response_text = process_query(user_id, query, user_history)
    # return {"response": "response_text"}
    
    for item in request_data:
        query = item.get("content")
        role = item.get("role")
        if query:
            # Process the query here and generate a response
            response_text = process_query(user_id, query, user_history)
            item["response"] = response_text
            item["role"] = role

    return request_data
