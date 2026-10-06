# Environment setup

All our experiments were run on Linux with an NVIDIA GPU. The repository includes a Dockerfile because `ctcdecode` is an old C++ extension and is sensitive to the Python, PyTorch, CUDA, gcc and glibc compiler versions around it.

The included Dockerfile offers a way to easily set up the proper environment for running the experiments. I have tested this Dockerfile on different computers running Arch Linux and different versions of Ubuntu Server (18.04, 20.04 and 22.04), and it worked on all of them.

## Creating Docker Container

Move to the Docker folder and run the appropritate create_container script for your current CUDA version.

```bash
nvidia-smi # You can see your CUDA version on the top right

cd CorrNet/docker\ assets/ # Enter the scripts folder and run the appropriate script from within the folder.

./create_container_cuda13.sh # if CUDA == 13.0
./create_container_cuda12.sh # if CUDA == 12.0
```


Enter the container and install the remaining dependencies.

```bash
docker exec -it corrnet bash
cd /model
./setup.sh
```

Activate the Conda environment again whenever a new shell is opened.

```bash
source /miniconda/etc/profile.d/conda.sh
conda activate modelvenv
```

## Sanity checks

Check PyTorch and CUDA.

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Check the decoder.

```bash
python -c "import ctcdecode; print('ctcdecode ok')"
```

Check the active learning parser.

```bash
python activelearning.py --help
```