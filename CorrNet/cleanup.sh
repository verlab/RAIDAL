if [ -z "$1" ]; then
    echo "Error: No directory provided. Usage: ./cleanup.sh <directory>"
    exit 1
fi

DIR="$1"

if [ ! -d "$DIR" ]; then
    echo "Error: Directory '$DIR' does not exist."
    exit 1
fi

cd "$DIR" || exit 1

echo "Cleaning up ${DIR}!"

find ./ -depth \( \
  -name "baseline.yaml" -o \
  -name "main.py" -o \
  -name "dataloader_video.py" -o \
  -name "slr_network_multi.py" -o \
  -name "modules" -o \
  -name "slowfast_modules" \
\) -exec rm -rfv {} \;