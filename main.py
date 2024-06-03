from fastapi import FastAPI, UploadFile, File,  HTTPException
from pydantic import BaseModel
from enum import Enum
import mariadb
import sys
from ollama import Client
import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from fastapi.responses import JSONResponse
import os


app = FastAPI()

@app.get("/generate/{prompt}")
async def llama(user_prompt: str, model: str, document: str = ""):
    client = Client(host='llama')
    try: 
        client.show(model)
    except:
        return {"Message": f"Model {model} not found"}
    
    if document:
        embedding_function=OllamaEmbeddings(model=model,  base_url = "http://llama:11434", show_progress=True)
        directory_path = f"/code/{model}"
        if not os.path.exists(directory_path):
            return {"Mesaage": f"No pdfs saved for model {model}"}
        directory_path += f"/{document}"
        if not os.path.exists(directory_path):
            return {"Message": f"Document {document} not imported"}
        print(directory_path)
        db = Chroma(persist_directory= directory_path, embedding_function=embedding_function, collection_name="local-rag")
        print(db.get(include=['embeddings', 'documents', 'metadatas']))
        llm = ChatOllama(model=model,  base_url = "http://llama:11434")
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
            db.as_retriever(), 
            llm,
            prompt=QUERY_PROMPT
        )
        
        #RAG prompt
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
        response = chain.invoke(user_prompt)
        return response

    try: 
        response = client.chat(model=model, messages=[
        {
            'role': 'user',
            'content': prompt,
        }, ])
    except ollama.ResponseError as e:
        return e.error
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
async def importPDF(modelName: str, file: UploadFile = File(...)):
    client = Client(host='llama')
    try: 
        client.show(modelName)
    except:
        return {"Message": f"Model {modelName} not found"}
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
    data[0].page_content = data[0].page_content.replace('\n', ' ')


    # Split and chunk 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    # Add to vector database
    doc_db = Chroma.from_documents(
        documents=chunks, 
        embedding=OllamaEmbeddings(model=modelName,  base_url = "http://llama:11434", show_progress=True),
        collection_name="local-rag",
        persist_directory= f"{modelName}/{file_name[:-4]}"
    )
    all_db = Chroma.from_documents(
        documents=chunks, 
        embedding=OllamaEmbeddings(model=modelName,  base_url = "http://llama:11434", show_progress=True),
        collection_name="local-rag",
        persist_directory= f"{modelName}/All_Documents"
    )
    doc_db.persist()
    all_db.persist()

    return {"Message": "Sucessfully added pdf"}





