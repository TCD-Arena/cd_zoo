#!/bin/bash

# This script runs execute_all_methods on all folder names found in a given path.
# Usage: ./execute_method_on_all_folders.sh <path_to_datasets> [method_name] [skip_n]
# Example: ./execute_method_on_all_folders.sh ../rename_after_generation
# Example: ./execute_method_on_all_folders.sh ../rename_after_generation var
# Example: ./execute_method_on_all_folders.sh ../rename_after_generation var 5

# Check if correct number of arguments provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <path_to_datasets> [method_name] [skip_n]"
    echo "Example: $0 ../rename_after_generation"
    echo "Example: $0 ../rename_after_generation var"
    echo "Example: $0 ../rename_after_generation var 5"
    exit 1
fi

DATASET_PATH="$1"
METHOD="$2"
SKIP_N="${3:-0}"  # Default to 0 if not provided

# Check if the provided path exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Directory '$DATASET_PATH' does not exist."
    exit 1
fi


if [ -n "$METHOD" ]; then
    if [ "$SKIP_N" -gt 0 ]; then
        echo "Running execute_all_methods with method '$METHOD' on all folders in '$DATASET_PATH', skipping first $SKIP_N folders"
    else
        echo "Running execute_all_methods with method '$METHOD' on all folders in '$DATASET_PATH'"
    fi
else
    if [ "$SKIP_N" -gt 0 ]; then
        echo "Running execute_all_methods on all folders in '$DATASET_PATH', skipping first $SKIP_N folders"
    else
        echo "Running execute_all_methods on all folders in '$DATASET_PATH'"
    fi
fi
echo "----------------------------------------"

# Get all folder names in the specified path
folders=($(find "$DATASET_PATH" -maxdepth 1 -type d -printf '%f\n' | grep -v '^$' | sort))

# Remove the base directory name if it appears in the list
folders=($(printf '%s\n' "${folders[@]}" | grep -v "$(basename "$DATASET_PATH")"))

# Skip the first SKIP_N folders if specified
if [ "$SKIP_N" -gt 0 ]; then
    echo "Skipping first $SKIP_N folders..."
    folders=("${folders[@]:$SKIP_N}")
fi

echo "Found ${#folders[@]} folders to process:"
printf '%s\n' "${folders[@]}"
echo "----------------------------------------"


# Process each folder
for folder in "${folders[@]}"; do
    echo "Processing folder: $folder"
    if [ -n "$METHOD" ]; then
        echo "Executing: ./execute_all_methods.sh $folder --method $METHOD"
         ./execute_all_methods.sh "$folder" --method "$METHOD" &
    else
        echo "Executing: ./execute_all_methods.sh $folder"
         ./execute_all_methods.sh "$folder" 
    fi
    echo "Completed: $folder"
    echo "----------------------------------------"
done

wait
echo "Done processing all folders in '$DATASET_PATH'"
echo "Total folders processed: ${#folders[@]}"
