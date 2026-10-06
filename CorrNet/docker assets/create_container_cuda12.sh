# To create the docker container, run:

MODEL_ROOTDIR="$(cd .. && pwd)" # The model is located in the parent folder of docker_assets/

docker build -t ubuntu2004-custom:latest .

docker run -d -it --shm-size=48g --gpus all -v "${MODEL_ROOTDIR}":/model --name=corrnet ubuntu2004-custom bash
