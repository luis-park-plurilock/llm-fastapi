# LLM-FastAPI
This project utilizes Docker to implement multiple API endpoints that allow for creating customizable LLMs.
Some of the key frameworks/modules/repositories that were used to make this possible were: 
FastAPI - the API framework
Langchain - preprocessing imported documents and multi-query retrieval(RAG)
ChromaDB - storing imported documents in vector form
TRL - finetuning mistral-7b-v0.1
Pytorch - dependancy for TRL 
Peft - using LORA Adapters for memory conservation 
Ollama - storage for customized LLMs
BitsandBytes - quantization of weights for memory conservation
Llama.cpp - converting finetuned models to gguf format for Ollama compatibility
## Hardware Requirements
As you would know, LLMs contain billions of tunable parameters, so components such as RAM and VRAM must be plentiful.
Furthermore, training billion parameter models are only comprehensible through utilizing a GPU rather than a CPU. Therefore, a GPU will be needed.
First, to appropriately use this project, a Nvidia GPU and Driver is required. Also, the Nvidia Container Toolkit must be installed
for the ability to use the GPU in the docker containers. Nvidia Container Toolkit must be compatible with the OS and Nvidia Driver that is 
installed. Compatibility documentation can be found here: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.11.0/install-guide.html
It is highly recommended that the device running the Docker Containers to have a minimum of 24 gigabytes RAM and and 24 gigabytes VRAM, otherwise, the finetuning 
api call will be prone to an OOM error. 
## LLMs
The LLMs that will be used are Mistral-7b-v0.1 and Llama3. Mistal-7b will be used for finetuning as this model has shown great finetuning capabilities,
whereas Llama3 seems to struggle with finetuning but is excellent for generalized intelligence. Llama3 will  automatically be downloaded upon runnning the 
docker compose file, but mistral-7b-v0.1 will need to be downloaded by running load_baseModel.py. To run this file, please install the module huggingfacehub, and
if need be, create a huggingface account and get an access token for the login(replace my login token if it is expired). This will be a one time download
for your local environment. The reason for local installation of these models is because downloading these models when the API is called is extremely slow (30 minutes), 
so pulling these models locally greatly reduces download time. 

## .sh Files
The clean_ollama_cache.sh file will delete all created models in your local environment. The clean_vectorDB will delete all imported documents from your local environment.
The entrypoint.sh file is used for the ollama container to download Llama3 when the docker container runs. 

# FastAPI Endpoints
## createModel
The create model api endpoint will simply take in the name of the model, the system template, and the base model, and utilize the ollama library to create the model. Note that ollama
is running on a seperate container, so we use the ollama client to connect to the ollama server running on the ollama container. 
The system template are commands that tell the model how to act. For example, if the system template is the following, "for each response, act as if you are Mario from the video game Mario Bros.", then for every response from the model, the model will act as if they are Mario. 
The base model parameter is preset to Llama3, and it is strongly advised to keep the base model to Llama3 because loading in other LLMs will take extremely long to download. Furthmore, if there are disk size constraints, then pulling in models other than Llama3 is not encouraged.
## deleteModel
This simply connects to the ollama server, and uses the ollama.delete function. If ollama delete throws an errors, the api will return all the available models that can be deleted.
## list_Models
Uses ollama.list() function to list all created models
## importDocument
Only allows user to import PDFs. This API call will first copy the imported PDF onto the file system of the FastAPI docker container. This is done because PyPDFLoader, a Langchain function that extracts the text from a pdf, only takes in a parameter of the filepath of the pdf rather than the pdf itsef. After doing so, the text is split into chunks by using
RecursiveCharacterTextSplitter. These chunks will then passed in a vectorized and tokenized form to the chromaDB container for storage and later use. Also note that chromaDB will have metadata specifying what model the document is associated with. Global document use for all models is not possible because different models will have different tokenizers, so each model will intepret documents differently.
## listDocuments
Connects to the chromaDB container and l=ists all documents with the specified model in the metadata.
## deleteDocument
Connects to the chromaDB container and deletes the document/collection given the model.
## deleteAll_Documents
Deletes all documents for a given model. Does so by deleting all documents that contain metadata referring to the given model.
## generate
This will generate a response from the given model, prompt, and optionally a list of documents. If not documents are listed, the API will simply connect to the ollama container, and pass in a prompt and the modelName, then return the response of the model. Otherwise, the API will use langchain's multi-query retriever to grab the most relevant vectors of each listed document, from chromaDb, to help answer the prompt. This is the Retreival Augmentation Generation system. 
## fine_tune_model
### To fully understand what is about to be said, please read about these topics below: Lora, Quantization, Finetuning Params, Chat Templates, the Llama.cpp Repository, Dataset formatting
The fine_tune model takes in three required parameters: a training csv dataset, an evaluation csv dataset, and the modelName. There are 9 additional preset parameters that
specify how the model will train: gradient accumulation steps, gradient checkpointing, epochs, learning rate, lora rank, lora alpha, lora dropout, packing, and batch size. 
The first steps of the finetuning api is to format the dataset in chatML form. After doing so, we set up the training arguements, and this is where 6/9 parameters will be placed. Then, we will set up the peft configurations, and this is where the Lora parameters Rank, Alpha, and Dropout will be placed. We then will load in mistral-7b-v0.1 as our base model in a 4bit quantized form. Finally, we can train our model. After training is complete, the finetuned model will be saved in the folder named qlora(quantized LORA). Note that the qlora folder will only contain the trained Lora adapters rather than the full finetuned model. 
Ollama can only import models in the form of .gguf files, so this is where llama.cpp will come in handy. We first convert the base mistral-7b-v0.1 model into a 16bit weight .gguf file, then we convert the lora adpaters in a .ggml file. This will then enable us to merge the lora adapter (.ggml file), and the base model (.gguf file) into a .gguf finetuned merged model. We then have to quanitize the finetuned model back to 4 bits for memory and inference optimization. Finally, import the merged, quantized, finetuned model inside the ollama container. Note that the final .gguf file needs to be in ollama container for it to be able to create the model. This is can be done easily by mounting both the FastAPI and the Ollama container into a share folder, so that the ollama container can access files in the share folder that are created from the FastApi container. 
# Topics to Understand Finetuning Components
## Quantization
Due to GPU and RAM constraints, loading billion parameter models in 16 bit precision is not feasible. To mitigate space constraints, we utilize quantization, which refers to the process of reducing the precision of number used to represent the model's parameters. In the current implementation of the finetuning API call, 4 bit quantization is used. If one is able to afford a finer precision in the bit representation of the parameters, they are able to change the quantization configurations for when pulling mistral-7b-v0.1.
For example: quantization_config = BitsAndBytesConfig(load_in_4bit=True) ->  quantization_config = BitsAndBytesConfig(load_in_8bit=True). This will allow training to be more precise. Furthmore, when quantizing the final finetuned model, specify a lower quantization setting. For example:
./app/llama.cpp/quantize ./app/shared/finetunedModel.gguf ./app/shared/finetunedModel-q4.gguf Q4_K_M" <- change Q4_K_M to your preference, options available in llama.cpp repository 
This will allow for the final finetuned model to have better performance but at the cost of memory.
## Lora
Low Rank Adaptation (LoRA) is a lightweight method for fine-tuning large language models. In essence, all trainable weights of a model can be represented as a matrix. For instance, consider a model with 10 billion parameters, which would correspond to a 100,000 x 100,000 matrix. During backpropagation, the gradients also need to be stored in a 100,000 x 100,000 matrix. This poses a significant memory challenge.

LoRA addresses this issue by representing the gradients with two smaller matrices instead of one large matrix. Specifically, instead of using a 100,000 x 100,000 matrix, we use two matrices: A (100,000 x R) and B (R x 100,000). The product of matrices A and B reconstructs the original 100,000 x 100,000 matrix. This approach reduces memory requirements because if R is relatively small, both matrices A and B together will have fewer total elements than the original matrix. This way, LoRA effectively reduces the memory footprint while preserving the model's ability to learn and adapt.

In the finetuning API call, 




