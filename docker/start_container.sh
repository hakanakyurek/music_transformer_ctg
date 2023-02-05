#!/bin/sh

CONTAINER_NAME="deepl"

docker start $CONTAINER_NAME
docker exec -it $CONTAINER_NAME bash
