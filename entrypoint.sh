#!/bin/bash
ollama serve &
echo "Starting Ollama server..."
echo "Waiting for Ollama server to be active..."
while [ "$(ollama list | grep 'NAME')" == "" ]; do
  sleep 1
done
ollama pull llama3
#keep in infinite loop for process to stay alive
while :; do
  sleep 300
done