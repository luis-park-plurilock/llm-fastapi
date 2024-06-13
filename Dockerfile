#install python for image
FROM python:3.9
#working directory
WORKDIR /code
RUN mkdir -p /code/app/saved_files
#copy over the requirements
COPY ./requirements_backend.txt /code/requirements_backend.txt
#copy all code to /code/app
COPY . /code/app
#install requirements
RUN pip3 install --no-cache-dir --upgrade -r ./requirements_backend.txt
RUN pip3 install --no-cache-dir --upgrade -r ./app/llama.cpp/requirements.txt

#run fast api, main.py is the main process
CMD ["fastapi", "run", "app/main.py", "--port", "80"]