#!/bin/bash

# Name of the Redis queue
queue_name="cellpose"

# Loop indefinitely
while true
do
    # Use BLPOP to pop an item from the queue. Adjust -h for Redis host and -p for port if needed.
    # The "0" argument specifies no timeout, so BLPOP will block until an item is available.
    read -r queue folder <<< $(redis-cli BLPOP $queue_name 0)

    echo "Segmenting ${folder}/stack.tif ..."
    # Run your script here. Replace 'your_script.sh' with the actual script you want to run.
    python src/adc/yeast/cellpose/segment.py ${folder}/stack.tif
    echo "making table"
    python /Users/aaristov/Documents/adc-nn/table.py $folder --redo
    echo "making contours"
    python /Users/aaristov/Documents/adc-nn/src/adc_nn/tools/contours.py ${folder}/cellpose.tif
done