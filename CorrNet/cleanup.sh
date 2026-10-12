#!/bin/bash
# Removes the copies of the source code that main.py saves into a work directory.
# Everything produced by training and active learning is kept: _best_model.pt, logs,
# evaluation outputs, config.yaml, labeled.corpus.csv, unlabeled.corpus.csv,
# ALParams.json and videoRank.json.

if [ -z "$1" ]; then
    echo "Error: No directory provided. Usage: ./cleanup.sh <directory>"
    exit 1
fi

DIR="$1"

if [ ! -d "$DIR" ]; then
    echo "Error: Directory '$DIR' does not exist."
    exit 1
fi

echo "Cleaning up ${DIR}!"

# main.py copies itself, the network, the temporal convolution, the ResNet backbone,
# the dataloader and the config file passed with --config. The full configuration is
# also saved as config.yaml, so the copied config is redundant and can go.
find "$DIR" -maxdepth 1 -type f \( \
  -name "main.py" -o \
  -name "slr_network.py" -o \
  -name "tconv.py" -o \
  -name "resnet.py" -o \
  -name "dataloader_video.py" -o \
  -name "baseline*.yaml" \
\) -exec rm -v {} +