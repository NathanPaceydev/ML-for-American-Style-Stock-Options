#!/bin/bash

# Path to your requirements.txt file
REQUIREMENTS_FILE="./requirements.txt"

# Read each line from the requirements.txt file
while IFS= read -r line
do
  # Extract package name (ignore version and file/URL paths)
  PACKAGE_NAME=$(echo "$line" | sed -E 's/@.*//' | sed -E 's/==.*//')
  
  # Check if the line is not empty and does not start with a comment
  if [[ ! -z "$PACKAGE_NAME" && ! "$PACKAGE_NAME" =~ ^#.* ]]; then
    echo "Upgrading $PACKAGE_NAME..."
    pip install --upgrade "$PACKAGE_NAME"
  fi
done < "$REQUIREMENTS_FILE"
