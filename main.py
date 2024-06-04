from fastapi import FastAPI, UploadFile, File,  HTTPException, Query
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
from typing import List
import os
import shutil

app = FastAPI()

@app.post("/generate/{prompt}")
async def llama(user_prompt: str, modelName: str, documents: List[str] = Query(None)):
    client = Client(host='llama')
    try: 
        client.show(modelName)
    except:
        return {"Message": f"Model {modelName} not found"}
    if documents:
        #embedding function will convert personal text to embeddings for specific model
        embedding_function=OllamaEmbeddings(model=modelName,  base_url = "http://llama:11434", show_progress=True)
        directory_path = f"/code/vector_documents/{modelName}"
        if not os.path.exists(directory_path):
            return {"Mesaage": f"No pdfs saved for model {modelName}"}
        length = len(directory_path)
        for document in documents:
            directory_path += f"/{document}"
            if not os.path.exists(directory_path):
                directory_path = directory_path[:length]
                dir_list = os.listdir(directory_path) 
                return {"Message": f"Document {document} not imported", f"Available Documents For {modelName}": dir_list}
            directory_path = directory_path[:length]
        res = {}
        for document in documents:
            directory_path += f"/{document}"
            #connect to chroma db
            db = Chroma(persist_directory= directory_path, embedding_function=embedding_function, collection_name="local-rag")
            #connect to chose llm model
            llm = ChatOllama(model=modelName,  base_url = "http://llama:11434")
            #query prompt used for generating different perspectives of the given question
            QUERY_PROMPT = PromptTemplate(
                input_variables=["question"],
                template="""You are an AI language model assistant. Your task is to generate five
                different versions of the given user question to retrieve relevant documents from
                a vector database. By generating multiple perspectives on the user question, your
                goal is to help the user overcome some of the limitations of the distance-based
                similarity search. Provide these alternative questions separated by newlines.
                Original question: {question}""",
            )
            #retrieves proper vectors in the db that gives llm proper context
            retriever = MultiQueryRetriever.from_llm(
                db.as_retriever(), 
                llm,
                prompt=QUERY_PROMPT
            )
            #The template given to llm for final question. Gives the context and answer
            template = """Answer the question based ONLY on the following context:
            {context}
            Question: {question}
            """
            #converts template to appropriate prompt
            prompt = ChatPromptTemplate.from_template(template)
            #user_prompt given as question, and context is the retriever
            #then prompt utilized the question and context to generate proper prompt
            #feed prompt to llm and then output it
            chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
            )
            #call the chain
            response = chain.invoke(user_prompt)
            res[f"{document}.pdf"] = response
            directory_path = directory_path[:length]
        return res
    try: 
        response = client.chat(model=modelName, messages=[
        {
            'role': 'user',
            'content': prompt,
        }, ])
    except ollama.ResponseError as e:
        return e.error
    return {'message': response['message']['content']}

@app.post("/createModel/{modelName}")
async def createModel(baseModel: str, modelName: str, system: str = ""):
    client = Client(host='llama')
    try: 
        client.show(modelName)
        return {"Message": f"Model Name {modelName} already exists"}
    except:
        print("User picked valid name")
    modelfile = f'''FROM {baseModel}\nSYSTEM {system}'''
    try:
        response = client.create(model=modelName, modelfile=modelfile)
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
    # Save the uploaded file to the specified path inside docker container
    try:
        file_name = file.filename
        filepath = f"/code/app/{file_name}"
        with open(filepath, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})
    #load the file suitable for splitting and chunking
    loader = PyPDFLoader(file_path=filepath)
    data = loader.load()
    #remove excessive \n
    data[0].page_content = data[0].page_content.replace('\n', ' ')
    # Split and chunk 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    # Add to vector database and save to container
    doc_db = Chroma.from_documents(
        documents=chunks, 
        embedding=OllamaEmbeddings(model=modelName,  base_url = "http://llama:11434", show_progress=True),
        collection_name="local-rag",
        persist_directory= f"/code/vector_documents/{modelName}/{file_name[:-4]}"
    )
    os.remove(filepath)
    return {"Message": "Sucessfully added pdf"}


@app.post("/list_documents/")
async def list_documents(modelName: str):
    client = Client(host='llama')
    try: 
        client.show(modelName)
    except:
        return {"Message": f"Model {modelName} not found"}
    path = f"/code/vector_documents/{modelName}"
    if not os.path.exists(path):
        return {"Mesaage": f"No pdfs saved for model {modelName}"}
    dir_list = os.listdir(path) 
    return{f"Imported Documents for {modelName}": dir_list}


@app.post("/list_models")
async def list_models():
    client = Client(host='llama')
    all_models = []
    for model in client.list()["models"]:
        all_models.append(model["name"][0:-7])
    return{"All Created Models": all_models}



@app.delete("/delete_document")
async def delete_document(document: str, modelName: str):
    client = Client(host='llama')
    try: 
        client.show(modelName)
    except:
        return {"Message": f"Model {modelName} not found"}
    path = f"/code/vector_documents/{modelName}"
    modelpath = path
    if not os.path.exists(path): return {"Message": f"Model {modelName} does not have any imported PDFs"}
    path+= f"/{document}"
    available_docs = os.listdir(modelpath)
    if not os.path.exists(path): return {"Message": f"Document {document} does not exist", 
        f"Available Documents For {modelName}": available_docs}
    shutil.rmtree(path)
    if not os.listdir(modelpath): shutil.rmtree(modelpath)
    return {"Message": f"Document {document} sucessfully removed"}

@app.delete("/deleteAll_Documents")
async def deleteAll_Documents(modelName: str):
    client = Client(host='llama')
    try: 
        client.show(modelName)
    except:
        return {"Message": f"Model {modelName} not found"}
    path = f"/code/vector_documents/{modelName}"
    if not os.path.exists(path): return {"Message": f"Model {modelName} does not have any imported PDFs"}
    shutil.rmtree(path)
    return {"Message": f"Successfully removed all documents for model {modelName}"}

@app.delete("/deleteModel/{modelName}")
async def deleteModel(modelName: str):
    client = Client(host='llama')
    try:
        response = client.delete(model=modelName)
        path = f"/code/vector_documents/{modelName}"
        if os.path.exists(path):
            shutil.rmtree(path)
        return {"Message": f"Successfully deleted {modelName}"}
    except ollama.ResponseError as e:
        all_models = []
        for model in client.list()["models"]:
            all_models.append(model["name"][0:-7])
        return {"Message": f"No model named {modelName}", "Created Models": all_models}
