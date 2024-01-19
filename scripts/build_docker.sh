#!/bin/bash


if test -z "$CI_REGISTRY_IMAGE"
then
	export CI_REGISTRY_IMAGE=ghcr.io/airtai/infobip-service
fi

if test -z "$CI_COMMIT_REF_NAME"
then
	export CI_COMMIT_REF_NAME=$(git branch --show-current)
fi

if [ "$CI_COMMIT_REF_NAME" == "main" ]
then
    export TAG=latest
	export CACHE_FROM="latest"
else
    export TAG=dev
	export CACHE_FROM="dev"
fi

echo Building $CI_REGISTRY_IMAGE


docker build --build-arg ACCESS_REP_TOKEN -t $CI_REGISTRY_IMAGE:$TAG .


if [ "$CI_COMMIT_REF_NAME" == "main" ]
then
	docker tag $CI_REGISTRY_IMAGE:$TAG $CI_REGISTRY_IMAGE:$CI_COMMIT_REF_NAME
fi

docker system prune --force
