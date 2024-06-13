#LLM-FastAPI
This project utilizes Docker to implement multiple API endpoints that allow for creating customizable LLMs.
Some of the key frameworks/modules/repositories that will be used to make this possible are: 
FastAPI - the API framework
Langchain - preprocessing imported documents for multi-query retrieval(RAG)
ChromaDB - storing imported documents in vector form
TRL - finetuning LLMs, Mistral-7b as the base model
Pytorch - dependancy for TRL 
Peft - using LORA Adapters for memory conservation 
Ollama - storage for customized LLMs
BitsandBytes - quantization of weights for memory conservation
Llama.cpp - converting finetuned models to gguf format for Ollama compatibility
##Hardware Requirements
As you would know, LLMs contain billions of tunable parameters, so components such as RAM and VRAM must be plentiful.
Furthermore, training billion parameter models are only comprehensible thorugh utilizing a GPU rather than a CPU. Therefore, a GPU will be needed.
First, to appropriately use this project, a Nvidia GPU and Driver is required. Also, the Nvidia Container Toolkit must be installed
for the ability to use the GPU in the docker containers. Nvidia Container Toolkit must be compatible with the OS and Nvidia Driver that is 
installed. Compatibility documentation can be found here: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.11.0/install-guide.html
It is highly recommended that the device running the Docker Containers to have a minimum of 24 gigabytes of RAM and VRAM, otherwise, the finetuning 
api call will be prone to an OOM error. 
##LLMs
The LLMs that will be used are Mistral-7b-v0.1 and Llama3. Mistal-7b will be used for finetuning as this model has shown great finetuning capabilities,
whereas Llama3 has shown great capabilites in generalized performance. Llama3 will be automatically be downloaded upon runnning the project(entrypoint.sh)
, but mistral-7b-v0.1 will need to be downloaded by running load_baseModel.py. To run this file, please install the module huggingfacehub, and
if need be, create a huggingface account and get an access token for the login(replace my login token if it is expired). This will be a one time download
for your local environment.
The reason for local installation of these models is because downloading these models when the API call is extremely slow (30 minues), so pulling these models
locally greatly reduces download time. 

##.sh Files
The clean_ollama_cache.sh file will delete all created models in your local environment. The clean_vectorDB will delete all imported documents from your local environment.
The entrypoint.sh file is used for the ollama container to download Llama3 when the docker container runs. 

#FastAPI Endpoints
##createModel
The create model api endpoint will simply take in the name of the model, the system template, and the base model, and utilize the ollama library to create the model. Note that ollama
is running on a seperate container, so we use the ollama client to connect to the ollama server running on the ollama container. 
The system template are commands that tell the model
how to act. For example, if the system template is "for each response, act as if you are Mario from the video game Mario Bros.", then for every response from the model,
the model will act as if they are Mario. 
The base model parameter is preset to Llama3, and it is strongly advised to keep the base model to Llama3 because loading in other LLMs will take extremely long to download.
##deleteModel
This simply connects to the ollama server, and uses ollama delete to the model. If ollama delete throws an errors, the api will return all the available models that can be deleted.
##list_Models
Uses ollama.list() function to list all created models
##importDocument
Only allows user to import PDFs. This API call will first copy the imported PDF onto the file system of the FastAPI docker container. This is done so that we can utilize PyPDFLoader
to extract the text from the pdf(PyPDFLoader only takes a parameter of the filepath of the pdf rather than the pdf itself). After doing so, the text is split into chunks by using
RecursiveCharacterTextSplitter. These chunks will then passed in vectorized form to the chromaDB container for storage and later use. Also note that chromaDB will have metadata specifying what model the document is associated with. Global document use for all models is not possible, because different models will have different tokenizers, so each model
will intepret documents differently.
##listDocuments
Finds all documents with the specified model in the metadata
each document
##deleteDocument
Connects to chromaDB container, and deletes the document/collection given the model.
##deleteAll_Documents
Deletes all documents for a given model. Does so by searching through all documents/collections in the chromaDB, and if the document's metadata shows that the document is for the
given model, will delete. 
##generate
This will generate a response from the given model, prompt, and optionally a list of documents. If not documents are listed, the the API will simply connect to the ollama container, and pass in a prompt given the modelName and return the response of the model. Otherwise, the API will use langchain's multi-query retriever to grab the most relevant vectors of each document, from chromaDb, to help answer the prompt. This is the Retreival Augmentation Generation system. 
##fine_tune_model
To fully understand what is about to be said, please read about these topics below: Lora, Quantization, Finetuning Params, Chat Templates, the Llama.cpp Repository, Dataset formatting
The fine_tune model takes in three required parameters: a training csv dataset, an evaluation csv dataset, and the modelName. There are 9 additional preset parameters that
specify how the model will train: gradient accumulation steps, gradient checkpointing, epochs, learning rate, lora rank, lora alpha, lora dropout, packing, and batch size. 
The first steps of the finetuning api is to format the dataset in chatML form. After doing so, we set up the training arguements, and this is where 6/9 parameters will be placed. Then, we will set up the peft configurations, and this is where the Lora parameters Lora rank, Lora Alpha, and Lora dropout will be placed. We then will load in mistral-7b-v0.1 as our base model in 4bit quantized form. Finally, we can train our model. After training is complete, the finetuned model will save in the folder named qlora(quantized LORA). Note that the qlora folder will only contain the trained Lora adapters. Ollama can only import models in the form of .gguf files, so this is where llama.cpp will come in handy. We first convert the base mistral-7b-v0.1 model into a 16bit weight .gguf file, then we convert the lora adpaters in a .ggml file. This will then enable us to merge the lora adapter in .ggml form, and the base model in .gguf form into a .gguf finetuned merged model. We then have to quanitize the finetuned model back to 4 bits because loading in 16 bits takes too long. Finally, create the model in the ollama container by using the merged, quantized finetuned model. Note that the gguf file needs to be in ollama container but we are doing this all in the FastAPI container. This is resolved by mounting each container into a share folder, so that the ollama container can access files in the share folder that are created from the FastApi container. 


