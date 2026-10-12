#!/bin/bash
# Removes the copies of the source code that main.py saves into a work directory.
# Everything produced by training and active learning is kept: _best_model.pt, logs,
# evaluation outputs, labeled.corpus.csv, unlabeled.corpus.csv, ALParams.json and
# videoRank.json.

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

# main.py copies itself and the network file. The copied config file
# (baseline-replicationX*.yaml) is kept on purpose: the Swin-MSTP main.py does not
# write config.yaml, so this copy is the only record of the training configuration.
find "$DIR" -maxdepth 1 -type f \( \
  -name "main.py" -o \
  -name "slr_network_Swin_MSTP.py" \
\) -exec rm -v {} +