# Environment setup

All our experiments were run on Linux with an NVIDIA GPU. The repository includes a Dockerfile because `ctcdecode` is an old C++ extension and is sensitive to the Python, PyTorch, CUDA, gcc and glibc versions around it.

The included Dockerfile offers a way to easily set up the proper environment for running the experiments. I have tested this Dockerfile on different computers running Arch Linux and different versions of Ubuntu Server (18.04, 20.04 and 22.04), and it worked on all of them.

## Creating Docker Container

Move to the Docker folder and run one of the two `create_container` scripts from inside it. They are identical except for how the GPU is passed to the container. `create_container_cuda12.sh` uses `--gpus all`, and `create_container_cuda13.sh` uses `--device=nvidia.com/gpu=all` (the CDI syntax). The names come from the CUDA 12.x and 13.x machines they were tested on, but which one works depends on how Docker and the NVIDIA Container Toolkit are configured on your machine. If the container can't see your GPU, try the other script.

```bash
nvidia-smi # You can see your CUDA version on the top right

# Enter the scripts folder and run the appropriate script from within the folder.
cd CorrNet/docker_assets/  # or: cd Swin-MSTP/docker_assets/

./create_container_cuda13.sh # if CUDA == 13.X
./create_container_cuda12.sh # if CUDA == 12.X
```


Enter the container and install the remaining dependencies.

```bash
docker exec -it corrnet bash # or: docker exec -it swin_mstp bash
cd /model
./setup.sh
```

`setup.sh` activates the environment only inside its own process, so activate it in your shell right after the script finishes, and again whenever you open a new shell.

```bash
source /miniconda/etc/profile.d/conda.sh
conda activate modelvenv
```

## Sanity checks

Check PyTorch and CUDA.

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Check the decoder.

```bash
python3 -c "import ctcdecode; print('ctcdecode ok')"
```

Check the active learning parser.

```bash
python3 activelearning.py --help
```