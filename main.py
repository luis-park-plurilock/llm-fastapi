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
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, BitsAndBytesConfig, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from typing import List
import os
import shutil
import torch
import pandas as pd
import ast
import chromadb
from chromadb import DEFAULT_TENANT, DEFAULT_DATABASE, Settings

app = FastAPI()



@app.post("/generate/{prompt}")
async def generate(user_prompt: str, modelName: str, documents: List[str] = Query(None)):
    client = Client(host='llama')
    try: 
        client.show(modelName)
    except:
        return {"Message": f"Model {modelName} not found"}
    if documents:
        embedding_function=OllamaEmbeddings(model=modelName,  base_url = "http://llama:11434", show_progress=True)
        client = chromadb.HttpClient(
            host = "chromaDB",
            port = 8000,
            settings = Settings(allow_reset=True, anonymized_telemetry=False),
            headers=None,
            tenant = DEFAULT_TENANT,
            database = DEFAULT_DATABASE,
        )
        for document in documents:
            try:
                client.get_collection(f"{document}{modelName}")
            except:
                return {"Message": f"Document {document} not imported for model {modelName}"}
        res = {}
        for document in documents:
            db = Chroma(client = client, embedding_function=embedding_function, collection_name= f"{document}{modelName}")            
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
                db.as_retriever(), 
                llm,
                prompt=QUERY_PROMPT
            )
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
            res[f"{document}.pdf"] = response
        return res
    try: 
        response = client.chat(model=modelName, messages=[
        {
            'role': 'user',
            'content': user_prompt,
        }, ])
    except ollama.ResponseError as e:
        return e.error
    return {'message': response['message']['content']}

@app.post("/createModel/{modelName}")
async def createModel(modelName: str, system: str = "", baseModel: str = "llama3",):
    client = Client(host='llama')
    try: 
        client.show(modelName)
        return {"Message": f"Model Name {modelName} already exists"}
    except:
        print("User picked valid name")
    modelfile = f'''FROM {baseModel}\nSYSTEM {system}'''
    try:
        client.create(model=modelName, modelfile=modelfile)
        return  {"Message": f"Successfully created {modelName}"}
    except ollama.ResponseError as e:
        return e.error

@app.post("/importDocument/")
async def importDocument(modelName: str, file: UploadFile = File(...)):
    client = Client(host='llama')
    try: 
        client.show(modelName)
    except:
        return {"Message": f"Model {modelName} not found"}
    if not file.filename.lower().endswith(".pdf"):
        return {"Message": f"Please import a PDF"}
    if file.content_type != "application/pdf":
        return {"Message": f"Please import a PDF"}
    try:
        file_name = file.filename
        filepath = f"/code/{file_name}"
        with open(filepath, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})
    loader = PyPDFLoader(file_path=filepath)
    data = loader.load()
    data[0].page_content = data[0].page_content.replace('\n', ' ')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    client = chromadb.HttpClient(
        host = "chromaDB",
        port = 8000,
        settings = Settings(allow_reset=True, anonymized_telemetry=False),
        headers=None,
        tenant = DEFAULT_TENANT,
        database = DEFAULT_DATABASE,
    )
    try:
        client.get_collection(f"{file_name[:-4]}{modelName}")
        return {"message": f"{modelName} has file {file_name[:-4]} already imported"}
    except:
        print("File is unique")
    embeddings = OllamaEmbeddings(model=modelName,  base_url = "http://llama:11434", show_progress=True)
    Chroma.from_documents(
        client= client,
        documents=chunks, 
        embedding=embeddings,
        collection_name= f"{file_name[:-4]}{modelName}",
        collection_metadata = {"model": modelName, "file_name": file_name[:-4]}
    )
    os.remove(filepath)
    return {"Message": f"Sucessfully added document {file_name}"}


@app.get("/list_documents/")
async def list_documents(modelName: str):
    client = Client(host='llama')
    try: 
        client.show(modelName)
    except:
        return {"Message": f"Model {modelName} not found"}
    client = chromadb.HttpClient(
        host = "chromaDB",
        port = 8000,
        settings = Settings(allow_reset=True, anonymized_telemetry=False),
        headers=None,
        tenant = DEFAULT_TENANT,
        database = DEFAULT_DATABASE,
    )
    cols = client.list_collections()
    all_docs = []
    for col in cols:
        if col.metadata["model"] == modelName:
            all_docs.append(col.metadata["file_name"])
  
    return{f"Imported Documents for {modelName}": all_docs}


@app.get("/list_models")
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
    client = chromadb.HttpClient(
        host = "chromaDB",
        port = 8000,
        settings = Settings(allow_reset=True, anonymized_telemetry=False),
        headers=None,
        tenant = DEFAULT_TENANT,
        database = DEFAULT_DATABASE,
    )
    try:
        client.get_collection(f"{document}{modelName}")
        client.delete_collection(f"{document}{modelName}")
    except:
        return {"message": f"{modelName} does not have document {document} imported"}
    return {"message": f"Successfully deleted document {document} from model {modelName}"}

@app.delete("/deleteAll_Documents")
async def deleteAll_Documents(modelName: str):
    client = Client(host='llama')
    try: 
        client.show(modelName)
    except:
        return {"Message": f"Model {modelName} not found"}
    client = chromadb.HttpClient(
        host = "chromaDB",
        port = 8000,
        settings = Settings(allow_reset=True, anonymized_telemetry=False),
        headers=None,
        tenant = DEFAULT_TENANT,
        database = DEFAULT_DATABASE,
    )
    cols = client.list_collections()
    for col in cols:
        if col.metadata["model"] == modelName:
            document = col.metadata["file_name"]
            client.delete_collection(f"{document}{modelName}")
    return {"Message": f"Successfully removed all documents for model {modelName}"}

@app.delete("/deleteModel/{modelName}")
async def deleteModel(modelName: str):
    client = Client(host='llama')
    try:
        client.delete(model=modelName)
        return {"Message": f"Successfully deleted {modelName}"}
    except ollama.ResponseError as e:
        all_models = []
        for model in client.list()["models"]:
            all_models.append(model["name"][0:-7])
        return {"Message": f"No model named {modelName}", "Created Models": all_models}


@app.post("/fine_tune_model/{modelName}")
async def fine_tune_model(modelName: str, train_dataset: UploadFile = File(...), eval_dataset: UploadFile = File(...),
            gradAcc: int = 50, lr: float = 2.0e-05, epochs: int = 1, packing: bool = True, batch_size: int = 1, 
            gc: bool = True, rank: int = 24, lora_alpha: int = 48, lora_dropout: float = 0.1):
    
    tokenizer = AutoTokenizer.from_pretrained("./app/mistral-7B-v0.1")
    client = Client(host='llama')
    try: 
        client.show(modelName)
        return {"Message": f"Model Name {modelName} already exists"}
    except:
        print("User picked valid name")
    if not train_dataset.filename.lower().endswith(".csv") or not eval_dataset.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only CSV files are allowed.")
    if train_dataset.content_type != "text/csv" or  eval_dataset.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Invalid content type. Only CSV files are allowed.")
    try:
        train_name = train_dataset.filename
        filepath = f"/code/app/{train_name}"
        with open(filepath, "wb") as f:
            f.write(await train_dataset.read())
        eval_name = eval_dataset.filename
        filepath = f"/code/app/{eval_name}"
        with open(filepath, "wb") as f:
            f.write(await eval_dataset.read())
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 1024
    tokenizer.padding_side = 'right'
    tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set \
    add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' \
    + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if \
    add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %})" 
    def apply_chat_template(example, tokenizer):
        messages = example["messages"]
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
        return example
    train_pd = pd.read_csv(f'./app/{train_dataset.filename}')
    eval_pd = pd.read_csv(f'./app/{eval_dataset.filename}')
    train_pd['messages'] = train_pd['messages'].apply(ast.literal_eval)
    eval_pd['messages'] = eval_pd['messages'].apply(ast.literal_eval)
    train_dataset = Dataset.from_pandas(train_pd)
    eval_dataset = Dataset.from_pandas(eval_pd)

    train_dataset = train_dataset.map(apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=['messages'],
        desc="Applying chat template",)
    eval_dataset = eval_dataset.map(apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=['messages'],
        desc="Applying chat template",)
    print(train_dataset["text"][0])
    print(eval_dataset["text"][0])
    training_args = SFTConfig(
        fp16=True, 
        do_eval=True,
        eval_strategy="epoch",
        gradient_accumulation_steps=gradAcc, 
        gradient_checkpointing=gc, 
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=lr,
        log_level="info",
        logging_steps=1, 
        logging_strategy="steps",
        lr_scheduler_type="cosine", 
        max_steps=-1, 
        num_train_epochs=epochs,
        output_dir="qlora_downloaded",
        overwrite_output_dir=True,
        per_device_eval_batch_size=batch_size, 
        per_device_train_batch_size=batch_size, 
        save_total_limit=None,
        seed=42,
        dataset_text_field = "text",
        max_seq_length=tokenizer.model_max_length,
        packing = packing
    )

    peft_config = LoraConfig(
        r=rank, 
        lora_alpha=lora_alpha, 
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
    )
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype= torch.bfloat16, 
    )
    device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

    model = AutoModelForCausalLM.from_pretrained("app/mistral-7b-v0.1", device_map = device_map, 
                    local_files_only = True, quantization_config = quantization_config, torch_dtype = 'auto')
    
    trainer = SFTTrainer(
            model=model, 
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config
        )
    trainer.train()
    trainer.save_model("qlora")

    with open("./app/Modelfile.txt", "r") as file:
        modelfile = file.read()

    os.system("python ./app/llama.cpp/convert-hf-to-gguf.py app/mistral-7b-v0.1 --outfile ./app/mistral-7b-v0.1.gguf --outtype f16")
    os.system("python ./app/llama.cpp/convert-lora-to-ggml.py qlora")
    os.system("./app/llama.cpp/export-lora -m ./app/mistral-7b-v0.1.gguf -o ./app/shared/finetunedModel.gguf -l ./qlora/ggml-adapter-model.bin")
    os.system("./app/llama.cpp/quantize ./app/shared/finetunedModel.gguf ./app/shared/finetunedModel-q4.gguf Q4_K_M")
    
    os.system(f"rm -rf ./app/{train_name}")
    os.system(f"rm -rf ./app/{eval_name}")
    os.system("rm -rf qlora")
    os.system("rm -rf qlora_downloaded")
    os.system("rm -rf ./app/mistral-7b-v0.1.gguf")
    os.system("rm -rf ./app/shared/finetunedModel.gguf")
    try:
        client.create(model=modelName, modelfile=modelfile)
        os.system("rm -rf ./app/shared/finetunedModel-q4.gguf")
        return {"Message": f"Sucessfully finetuned model {modelName}"}
    except ollama.ResponseError as e:
        os.system("rm -rf ./app/shared/finetunedModel-q4.gguf")
        return e.error
