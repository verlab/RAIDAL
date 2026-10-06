# To create the docker container, run:

MODEL_ROOTDIR="$(cd .. && pwd)" # The model is located in the parent folder of 'docker assets/'

docker build -t ubuntu2004-custom:latest .

docker run -d -it --shm-size=48g --device=nvidia.com/gpu=all -v "${MODEL_ROOTDIR}":/model --name=swin_mstp ubuntu2004-custom bash
