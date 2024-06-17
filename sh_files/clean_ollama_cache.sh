#!/bin/bash

# Specify the directory to clean up (change this path to your desired directory)
DIRECTORY="./ollama_cache"

# Loop through each file/directory in the specified directory
for file in "$DIRECTORY"/*; do
  # Exclude the .dockerignore file
  if [ "$(basename "$file")" != ".gitignore" ]; then
    # Delete the file or directory
    echo "Deleting $(basename "$file")"
    rm -rf "$file"
  fi
done

echo "Cleanup completed."