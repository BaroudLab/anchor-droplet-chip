#!/bin/bash
set -e
# The URL from which to fetch the JSON data
API_URL="https://nocodb01.pasteur.fr/api/get_cellpose_tasks"

# Use curl to fetch the JSON data and jq to parse the 'task' value

# Check if the task path is not empty
# Loop indefinitely
while true; do
    # Fetch the JSON from the API
    response=$(curl -s $API_URL)

    # Extract the "task" value from the JSON response
    folder=$(echo $response | jq -r '.task')
    # folder="${task/home/Users}"

    # Check if the "task" is not null
    if [ "$folder" != "null" ]; then
        echo "Processing task: $folder"

        python src/adc/yeast/cellpose/segment.py ${folder}stack.tif
        echo "making table"
        python ~/Documents/adc-nn/table.py $folder
        echo "making contours"
        python ~/Documents/adc-nn/src/adc_nn/tools/contours.py ${folder}cellpose.tif
    else
        echo "No task available, waiting 10 seconds..."
        sleep 10
    fi
done
