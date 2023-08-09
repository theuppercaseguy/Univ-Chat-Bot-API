from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain,RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
import os
from fastapi import FastAPI, Request, HTTPException
import json
from langchain import PromptTemplate, LLMChain
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
#from pydantic import BaseModel
PromptTemplate
from fastapi.middleware.cors import CORSMiddleware
os.environ["OPENAI_API_KEY"] = "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"

class UserData:
  def __init__(self, users):
    self.users = users



loader = TextLoader("scrapped_Data.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


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


qa = ConversationalRetrievalChain.from_llm(
   OpenAI(temperature=0.5), 
   vectorstore.as_retriever(), 
   memory=memory, 
   max_tokens_limit=4000
   )


chat_history = []
def QuestionAnswers(que, chat_history):
  query = que
  result = qa({"question": query, "chat_history": chat_history})
  return result["answer"]


app = FastAPI()


# Create a templates object using Jinja2Templates
templates = Jinja2Templates(directory="./templates")

@app.get("/",response_class=HTMLResponse)
async def root(request: Request):
   return templates.TemplateResponse("index.html",context={"request":request,})
  
"""
##########################################################
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.chains import create_qa_with_sources_chain



embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(documents, embeddings)

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo",)
qa_chain = create_qa_with_sources_chain(llm )

doc_prompt = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)

final_qa_chain = StuffDocumentsChain(
    llm_chain=qa_chain,    
    document_variable_name="context",
    document_prompt=doc_prompt,
  
)

retrieval_qa = RetrievalQA(
    retriever=docsearch.as_retriever(), 
    combine_documents_chain=final_qa_chain,
    
)

query = "What did the president say about russia"


print(retrieval_qa.run(query))

##########################################################
"""

def read_database():
    try:
        with open("./database.json", "r") as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        return {"users": {}}

def write_database(data):
    database_file_path = os.path.join(os.getcwd(), "database.json")
    # print("write: ", database_file_path, data)
    with open(database_file_path, "w") as file:
        json.dump(data, file)

def store_in_database(user_id: str, query: str, answer: str):
  data = read_database()
  # print("database: ", data)
    
  if user_id not in data["users"]:
    data["users"][user_id] = []
    
  if isinstance(data["users"][user_id], list):
    data["users"][user_id].append({"question": query, "answer": answer})
  else:
    existing_entry = {
      "question": data["users"][user_id]["question"],
      "answer": data["users"][user_id]["answer"]
    }
    data["users"][user_id] = [existing_entry, {"question": query, "answer": answer}]
  write_database(data)
  # print("dataLLLL", data)

def get_user_history(user_id: str):
    data = read_database()
    if user_id in data["users"]:
        return UserData(users=data["users"])  # Pass the whole data["users"] as the argument
    else:
        return UserData(users={user_id: []})

default_user_id = "123"
def process_query(user_id: str, query: str, user_history: UserData):
    if not query:
        return None

    user_chat_history = user_history.users[user_id]
    # print("321: ",user_chat_history)

    chat_history = list()
    for chat in user_chat_history:
      chat_history.append(chat)

    # print("Chat history2: ",chat_history)
    
    result = QuestionAnswers(query, chat_history)
    store_in_database(user_id, query, result)

    return result
    #return chat_history

@app.get("/chat_me", response_class=HTMLResponse)
async def get_chat_me(request: Request, query: str = None, user_id: str = default_user_id):

    # print(f"Received User ID: {user_id}")

    user_history = get_user_history((user_id))
    # print("user_history:\t", user_history.users)
    response_text = process_query(user_id, query, user_history)

    return templates.TemplateResponse("index.html", {
        "user_id": user_id,
        "request": request, 
        "response_text": response_text, 
        "chat_history": user_history.users[user_id] 
    })



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat")
async def chat(request_data:list):
    # print(request_data)
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







