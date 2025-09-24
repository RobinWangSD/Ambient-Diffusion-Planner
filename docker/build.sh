#!/bin/bash
PUSH_PLANNING_INTERN=1
PLANNING_INTERN_DOCKER_PATH=robinwangucsd/ambient_diffusion_planner
TAG=latest

docker build -t $PLANNING_INTERN_DOCKER_PATH:$TAG -f ./Dockerfile .

if [ "$PUSH_PLANNING_INTERN" -eq 1 ]; then
    echo "Pushing to docker registry"
    docker push $PLANNING_INTERN_DOCKER_PATH:$TAG
fi

