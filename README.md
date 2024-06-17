# LLM-FastAPI
This project utilizes Docker to implement multiple API endpoints that allow for creating customizable LLMs.
Some of the key frameworks/modules/repositories that were used to make this possible were:  
  
**FastAPI** - the API framework  
**Langchain** - preprocessing imported documents and multi-query retrieval(RAG)  
**ChromaDB** - storing imported documents in vector form  
**TRL** - finetuning mistral-7b-v0.1  
**Pytorch** - dependancy for TRL   
**Peft** - using LORA Adapters for memory conservation  
**Ollama** - storage for customized LLMs  
**BitsandBytes** - quantization of weights for memory conservation  
**Llama.cpp** - converting finetuned models to gguf format for Ollama compatibility  
### Hardware Requirements
To appropriately use this project, a Nvidia GPU and Driver is required. Also, the Nvidia Container Toolkit must be installed
for the ability to use the GPU in the docker containers. Nvidia Container Toolkit must be compatible with the OS and Nvidia Driver that is 
installed. Compatibility documentation can be found here: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.11.0/install-guide.html.
It is highly recommended that the device running the Docker Containers to have a minimum of 24 gigabytes RAM and and 24 gigabytes VRAM, otherwise, the finetuning 
api call will be prone to an OOM error. 
### LLMs
The LLMs that will be used are Mistral-7b-v0.1 and Llama3. Mistal-7b will be used for finetuning as this model has shown great finetuning capabilities,
whereas Llama3 seems to struggle with finetuning but is excellent for generalized intelligence. Llama3 will  automatically be downloaded upon runnning the 
docker compose file, but mistral-7b-v0.1 will need to be downloaded by running load_baseModel.py. To run this file, please install the module huggingfacehub, and
if need be, create a huggingface account and get an access token for the login(replace my login token if it is expired). This will be a one time download
for your local environment. The reason for local installation of these models is because downloading these models when the API is called is extremely slow (30 minutes), 
so pulling these models locally greatly reduces download time. 

### Initial Load Up
When initially running the docker compose file, please wait for the Ollama container to fully download and pull Llama3. The first time running the docker compose file will be slow due to the size of the llama3 model(around 7-8 minutes). After it pulls llama3 for the first time, the model will be placed in the ollama_cache folder. This in result will make future load times much faster. 

# FastAPI Endpoints
### createModel
The create model api endpoint will simply take in the name of the model, the system template, and the base model, and utilize the ollama library to create the model. Note that ollama
is running on a seperate container, so we use the ollama client to connect to the ollama server running on the ollama container.  
The system template are commands that tell the model how to act. For example, if the system template is the following, "for each response, act as if you are Mario from the video game Mario Bros.", then for every response from the model, the model will act as if they are Mario.   
The base model parameter is preset to Llama3, and it is strongly advised to keep the base model to Llama3 because loading in other LLMs will take extremely long to download. Furthmore, if there are disk size constraints, then pulling in models other than Llama3 is not encouraged.  
### deleteModel
This simply connects to the ollama server, and uses the ollama.delete function. If ollama delete throws an errors, the api will return all the available models that can be deleted.
### list_Models
Uses ollama.list() function to list all created models
### importDocument
Only allows user to import PDFs. This API call will first copy the imported PDF onto the file system of the FastAPI docker container. This is done because PyPDFLoader, a Langchain function that extracts the text from a pdf, only takes in a parameter of the filepath of the pdf rather than the pdf itsef. After doing so, the text is split into chunks by using
RecursiveCharacterTextSplitter. These chunks will then passed in a vectorized and tokenized form to the chromaDB container for storage and later use. Also note that chromaDB will have metadata specifying what model the document is associated with. Global document use for all models is not possible because different models will have different tokenizers, so each model will intepret documents differently.
### listDocuments
Connects to the chromaDB container and l=ists all documents with the specified model in the metadata.
### deleteDocument
Connects to the chromaDB container and deletes the document/collection given the model.
### deleteAll_Documents
Deletes all documents for a given model. Does so by deleting all documents that contain metadata referring to the given model.
### generate
This will generate a response from the given model, prompt, and optionally a list of documents. If not documents are listed, the API will simply connect to the ollama container, and pass in a prompt and the modelName, then return the response of the model. Otherwise, the API will use langchain's multi-query retriever to grab the most relevant vectors of each listed document, from chromaDb, to help answer the prompt. This is the Retreival Augmentation Generation system. 
### fine_tune_model
**To fully understand what is about to be said, please read about these topics below: Lora, Quantization, Finetuning Params, the Llama.cpp Repository, Dataset Requirements**  
The fine_tune model takes in three required parameters: a training csv dataset, an evaluation csv dataset, and the modelName. There are 9 additional preset parameters that
specify how the model will train: gradient accumulation steps, gradient checkpointing, epochs, learning rate, lora rank, lora alpha, lora dropout, packing, and batch size. 
The first steps of the finetuning api is to format the dataset in chatML form. After doing so, we set up the training arguements, and this is where 6/9 parameters will be placed. Then, we will set up the peft configurations, and this is where the Lora parameters Rank, Alpha, and Dropout will be placed. We then will load in mistral-7b-v0.1 as our base model in a 4bit quantized form. Finally, we can train our model. After training is complete, the finetuned model will be saved in the folder named qlora(quantized LORA). Note that the qlora folder will only contain the trained Lora adapters rather than the full finetuned model.  
Ollama can only import models in the form of .gguf files, so this is where llama.cpp will come in handy. We first convert the base mistral-7b-v0.1 model into a 16bit weight .gguf file, then we convert the lora adpaters in a .ggml file. This will then enable us to merge the lora adapter (.ggml file), and the base model (.gguf file) into a .gguf finetuned merged model. We then have to quanitize the finetuned model back to 4 bits for memory and inference optimization. Finally, import the merged, quantized, finetuned model inside the ollama container. Note that the final .gguf file needs to be in ollama container for it to be able to create the model. This is can be done easily by mounting both the FastAPI and the Ollama container into a share folder, so that the ollama container can access files in the share folder that are created from the FastApi container. 
# Topics to Understand Finetuning Components
### Quantization
Due to GPU and RAM constraints, loading billion parameter models in 16 bit precision is not feasible. To mitigate space constraints, we utilize quantization, which refers to the process of reducing the precision of number used to represent the model's parameters. In the current implementation of the finetuning API call, 4 bit quantization is used. If one is able to afford a finer precision in the bit representation of the parameters, they are able to change the quantization configurations for when pulling mistral-7b-v0.1.  
  
For example:   
quantization_config = BitsAndBytesConfig(load_in_4bit=True) ->  quantization_config = BitsAndBytesConfig(load_in_8bit=True).   
  
This will allow training to be more precise. Furthmore, when quantizing the final finetuned model, specify a lower quantization setting.   
  
For example:  
./app/llama.cpp/quantize ./app/shared/finetunedModel.gguf ./app/shared/finetunedModel-q4.gguf Q4_K_M"   
Change Q4_K_M to your preference, options available in llama.cpp repository.   
  
This will allow for the final finetuned model to have better performance but at the cost of memory.  
### Lora
Low Rank Adaptation (LoRA) is a lightweight method for fine-tuning large language models. In essence, all trainable weights of a model can be represented as a matrix. For instance, consider a model with 10 billion parameters, which would correspond to a 100,000 x 100,000 matrix. During backpropagation, the gradients also need to be stored in a 100,000 x 100,000 matrix. This poses a significant memory challenge.  
  
LoRA addresses this issue by representing the gradients with two smaller matrices instead of one large matrix. Specifically, instead of using a 100,000 x 100,000 matrix, we use two matrices: A (100,000 x R) and B (R x 100,000). The product of matrices A and B reconstructs the original 100,000 x 100,000 matrix. This approach reduces memory requirements because if R is relatively small, both matrices A and B together will have fewer total elements than the original matrix. This way, LoRA effectively reduces the memory footprint while preserving the model's ability to learn and adapt. R is refered to as "rank", and this is found as one of the hyperparameters for finetuning. As one increases rank, memory use will increase for storing gradients, but precision of representing gradients will also increase. Lora Alpha is another hyperparameter, and this indicates the scaling factor for the weight matrices.   

### Finetuning Parameters
There are hundreds of finetuning parameters for the trl library. Please refer to this page in the trainingArguements section to learn what each parameter does: https://huggingface.co/docs/transformers/main_classes/trainer. Note that the finetuning API call takes in 9 finetuning parameters, and it is likely more will need to be added.

### Llama.cpp Repository
The Llama.cpp repository is a versatile tool that facilitates several key tasks for working with models. It allows users to:  
  
1. Convert LoRA adapters to .ggml files  
2. Convert base Huggingface models to .gguf files   
3. Merge .ggml files into .gguf files, resulting in a final fine-tuned model in .gguf format    
3. Quantize .gguf files  
  
These conversions are essential because Ollama servers require models to be in .gguf format and do not support the typical Huggingface model structure, which includes .safetensors and .json files.  
  
The Llama.cpp folder in this repository was cloned from an older version of the repository (https://github.com/ggerganov/llama.cpp/tree/04a5ac211ef40936295980b7cdf0ba6e97093146) to retain the convert-lora-to-ggml.py script, which newer versions have removed. This script converts LoRA adapters to .ggml files.  
  
Alternatively, instead of using convert-lora-to-ggml.py, you can use the merge_and_unload function from the Peft library to merge the base model with LoRA adapters, producing a Huggingface-formatted model. This merged model can then be converted to a .gguf file using convert-hf-to-gguf.py.  
  
By following these steps, you can ensure compatibility with Ollama servers and leverage the full capabilities of your fine-tuned models.  

### Dataset Requirements
The datasets passed into the finetuning API **must** be in this format:   
  
    "[{""content"": ""prompt1"", ""role"": ""user""}, {""content"": ""response1"", ""role"": ""assistant""} , ....]"  
    "[{""content"": ""prompt2"", ""role"": ""user""}, {""content"": ""response2"", ""role"": ""assistant""}, ....]"  
  
As you can see, the API supports a continuous stream of prompt and responses from the user and assistant. Make sure that there are no other keys other than "content" and "role". Furthermore, the value for "role" should only be "assistant" or "user". Please refer to the dummy dataset in the repository to clear any confusions.   
  
Also, note that if the training process is unusually fast and the dataset is relatively small (1000-5000 examples) with short examples, set **packing** to False. Packing combines smaller examples, significantly reducing the dataset size. For instance, I once set packing to True for a dataset with 1000 examples. Although the model trained quickly, it resulted in minimal learning. The finetuned model showed no improvement over the pretrained model. After extensive debugging, I discovered that packing had reduced my dataset from 1000 examples to just 26, leaving insufficient data for effective training. While packing is beneficial for large datasets, it should be avoided for smaller ones.


# Future Progression
### Fresh Llama.cpp Repo
If, the current commit of the Llama.cpp repository is out of date and one wishes to pull a fresh commit, please do the following:  
    
1. Delete the current Llama.cpp repository 
2. Git clone a newer version of the repository: https://github.com/ggerganov/llama.cpp.git
3. Compile the repository by either using 'make' or 'cmake'
4. Create another folder and move all content in the Llama.cpp repository in the newly created folder
5. Delete the empty Llama.cpp repository
6. Rename the newly created folder to Llama.cpp
7. If the new Llama.cpp is over 100MB, delete a few unused files (GitHub doesn't allow users to push folders over 100MB)  
  
The reason for copying all contents to a newly created folder is to avoid issues with submodules in GitHub. When cloning a project that includes the Llama.cpp repository as a submodule, the Llama.cpp folder may not be pulled correctly. By copying all files to a new folder and pushing that folder to GitHub, this issue is circumvented.  If there is a more efficient method, please feel free to adjust the steps accordingly, as my limited knowledge of Git has led me to this approach.  
   
If the latest commit of the freshly cloned Llama.cpp repository does not include the convert-lora-to-ggml.py file, you can still proceed after training the LoRA adapters. Simply use the merge_and_unload function from the Peft library to merge the LoRA adapters with the base model Mistral-7b-v0.1. After merging, convert the resulting model directly to a .gguf file, bypassing the need for a .ggml file.  
For example:   
  
trainer.train()  
trainer.save_model("qlora") #saving trained model to qlora folder  
base = AutoModelForCausalLM.from_pretrained("app/mistral-7b-v0.1", .....)  
model = PeftModel.from_pretrained(base, "qlora")  
merged_model = model.merge_and_unload() #merge adapters with base  
merged_model.save_pretrained("path")  
os.system("python ./app/llama.cpp/convert-hf-to-gguf.py path --outfile ./app/finetuned.gguf --outtype f16")  
  

### Changing Finetuning Model
If one wants to change the finetuning model, please do the following:  
  
1. In the load_baseModel.py, change the repo id to the repository of the preferred finetuned model
2. In the fintuned FastAPI call, change the both the tokenizer and AutoModelForCausalLM path to where the saved model is located
3. **You must change the chat template to the one it was trained on**, you can find the template on the Huggingface website
4. In the finetuned FastAPI call, change the os.system calls to the appropriate path as well


# Addtional Notes
### Persistence of Data
In this repository, there are two folders that allow the persistence of data: vectorDB and ollama_cache. vectorDB will contain the data of imported documents, and ollama_cache will contain created models. There are .gitignore files placed in these folders because GitHub does not allow pushing folders of 100MB or bigger: these folders scale extremely fast.
### entrypoint.sh
When the Ollama container initially runs, it must first start the Ollama server and then pull Llama3. The Ollama image is designed to launch the Ollama server upon container startup. Initially, a RUN [pull llama3] command was added to the Dockerfile, but this caused a race condition. If the Ollama server wasn't fully initialized before the pull command executed, the command would fail and break the container.  
  
To resolve this, an entrypoint script (.sh file) was implemented for the Ollama container. This script starts the Ollama server and enters a while loop, continuously checking until the server is fully initialized. Once the server is ready, it pulls Llama3 and then enters an infinite loop. This ensures the entrypoint process remains active, keeping the container running until manually stopped.
### Unused Llama.cpp Files
The Llama.cpp directory is quite large, and many of its files are not essential for this project. To conserve disk space when running the containers, you can delete unused files. However, exercise caution because the necessary files have many dependencies within the Llama.cpp folder. I have opted not to delete a significant number of files due to uncertainty about their associations and to avoid potential issues.
### ChromaDB Setup
ChromaDB serves as a powerful vector database tailored for efficiently handling and querying extensive collections of vector embeddings. Its architecture revolves around organizing these embeddings into distinct units termed "collections." In our project's implementation, each imported document is treated as an individual collection within ChromaDB. However, it's worth noting that the system supports the aggregation of multiple documents into a single collection if desired. Should this be the preferred approach, adjustments to the importDocument API implementation can be made accordingly.