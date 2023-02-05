CONTAINER_NAME="deepl"
IMAGE_NAME="hakan/base:latest"


docker run --privileged -it \
           --volume=/dev:/dev:rw \
           --shm-size=1gb \
           --env=TERM=xterm-256color \
           --name $CONTAINER_NAME \
           $IMAGE_NAME \
           bash \
