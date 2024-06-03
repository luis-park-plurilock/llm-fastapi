from fastapi import FastAPI, UploadFile, File,  HTTPException
from pydantic import BaseModel
from enum import Enum
import mariadb
import sys
from ollama import Client
import ollama
from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from fastapi.responses import JSONResponse
import inspect



# Connect to MariaDB Platform
def connect():
    try:
        conn = mariadb.connect(
            user="root",
            password="S3cret",
            host="db",
            #this is port for whats running inside the db container, not the exposed port
            port=3306, 
            database="todo_db"
        )
        return conn
    except mariadb.Error as e:
        print(f"Error connecting to MariaDB Platform: {e}")
        sys.exit(1)

conn = connect()
cursor = conn.cursor()
# Create the files_contents table
cursor.execute("CREATE TABLE files_contents (filename TEXT, contents TEXT)")
cursor.close()
conn.close()

app = FastAPI()

@app.get("/todos/llama/{prompt}")
async def llama(prompt: str):
    client = Client(host='llama')
    response = client.chat(model='llama3', messages=[
    {
        'role': 'user',
        'content': prompt,
    }, ])
    return {'message': response['message']['content']}


@app.post("/createModel/{modelName}")
async def createModel(modelName: str, system: str, model: str):
    client = Client(host='llama')
    modelfile = f'''FROM {model}\nSYSTEM {system}'''
    try:
        response = client.create(model=modelName, modelfile=modelfile)
        return response
    except ollama.ResponseError as e:
        return e.error


@app.delete("/deleteModel/{modelName}")
async def deleteModel(modelName: str):
    client = Client(host='llama')
    try:
        response = client.delete(model=modelName)
        return response
    except ollama.ResponseError as e:
        return e.error

@app.post("/importPDF/")
async def importPDF(modelName: str, question: str, file: UploadFile = File(...)):
    # Ensure the uploaded file is a PDF
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")
    
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid content type. Only PDF files are allowed.")
    # Save the uploaded file to the specified path
    try:
        file_name = file.filename
        filepath = f"/code/app/{file_name}"
        with open(filepath, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})


    loader = PyPDFLoader(file_path=filepath)
    data = loader.load()
    # Split and chunk 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    # Add to vector database
    vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=OllamaEmbeddings(model=modelName,  base_url = "http://llama:11434", show_progress=True),
    collection_name="local-rag"
    )
    llm = ChatOllama(model=modelName,  base_url = "http://llama:11434")
    QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)
    retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), 
    llm,
    prompt=QUERY_PROMPT
)
    # RAG prompt
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )
    x = chain.invoke(question)
    return x




