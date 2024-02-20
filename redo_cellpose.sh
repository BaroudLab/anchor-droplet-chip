#!/bin/bash

# Define the base path
BASE_PATH="/Volumes/Multicell/Madison/2024-01-19_MLY003_Cas9_sync/pos/Included"
# Get today's date in "YYYY-MM-DD" format
today=$(date +%Y-%m-%d)

# Loop through each folder matching the pattern
for folder in ${BASE_PATH}/pos*/input/; do
    echo "Checking folder: $folder"
    # Check if 'cellpose.tif' exists
    if [[ -f "${folder}cellpose.tif" ]]; then
        # Get the modification date of 'cellpose.tif' in "YYYY-MM-DD" format
        modDate=$(stat -f "%Sm" -t "%Y-%m-%d" "${folder}cellpose.tif")
        if [[ "$modDate" != "$today" ]]; then
            echo "'cellpose.tif' in $folder is not from today. segmenting ..."
            # Run your script here. Replace 'your_script.sh' with the actual script you want to run.
            python src/adc/yeast/cellpose/segment.py ${folder}/stack.tif
            echo "making table"
            python /Users/aaristov/Documents/adc-nn/table.py $folder --redo
            echo "making contours"
            python /Users/aaristov/Documents/adc-nn/src/adc_nn/tools/contours.py ${folder}/cellpose.tif
        else
            echo "'cellpose.tif' in $folder is from today. Skipping..."
        fi
    else
        echo "'cellpose.tif' in $folder is from today or does not exist. Skipping..."
    fi
done