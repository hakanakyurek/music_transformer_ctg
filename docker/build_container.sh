CONTAINER_NAME="deepl"
IMAGE_NAME="hakan/base:latest"


docker run --privileged -it \
           --gpus all \
           --env=TERM=xterm-256color \
           --name $CONTAINER_NAME \
           $IMAGE_NAME \
           bash \
