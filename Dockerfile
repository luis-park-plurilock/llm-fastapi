# Stage 1: Build dependencies
FROM python:3.9 AS builder

# Set the working directory
WORKDIR /code

# Create a directory for saved files
RUN mkdir -p /code/app/saved_files

# Copy over the requirements
COPY ./requirements_backend.txt /code/requirements_backend.txt
# Copy all code to /code/app
COPY . /code/app

# Install backend requirements
RUN pip3 install --no-cache-dir --upgrade -r /code/requirements_backend.txt
RUN pip3 install --no-cache-dir --upgrade -r /code/app/llama.cpp/requirements.txt

# Stage 2: Create the final image
FROM python:3.9-slim

# Set the working directory
WORKDIR /code

# Copy the code and installed dependencies from the builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /code /code

# Set the entry point for FastAPI using Uvicorn
CMD ["fastapi", "run", "app/main.py", "--port", "80"]