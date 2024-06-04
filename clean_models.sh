#!/bin/bash

file_to_keep="llama3"
folder_path="ollama_cache/models/manifests/registry.ollama.ai/library"

# Change to the target directory
cd "$folder_path" || exit

# Loop through all files in the directory
for folder in *; do
  # Check if the file is not the file to keep
  if [ "$folder" != "$file_to_keep" ]; then
    # Delete the file
    rm -rf "$folder"
  fi
done

