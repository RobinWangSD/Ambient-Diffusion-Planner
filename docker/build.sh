#!/bin/bash
PUSH_PLANNING_INTERN=1
PLANNING_INTERN_DOCKER_PATH=docker-registry.qualcomm.com/prediction/prediction-lab
TAG=unitraj

docker build -t $PLANNING_INTERN_DOCKER_PATH:$TAG -f ./Dockerfile .

if [ "$PUSH_PLANNING_INTERN" -eq 1 ]; then
    echo "Pushing to planning intern docker registry"
    docker push $PLANNING_INTERN_DOCKER_PATH:$TAG
fi

