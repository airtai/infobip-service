#!/bin/bash

set -a # automatically export all variables
source .env
set +a

if test -z "$CI_REGISTRY_IMAGE"
then
	echo "INFO: CI_REGISTRY_IMAGE variable not set, setting it to 'ghcr.io/airtai/infobip-kafka-service'"
	export CI_REGISTRY_IMAGE="ghcr.io/airtai/infobip-kafka-service"
fi

if test -z "$TAG"
then
	echo "ERROR: TAG variable must be defined, exiting"
	exit -1
fi

if test -z "$CI_REGISTRY"
then
	echo "INFO: CI_REGISTRY variable not set, setting it to 'ghcr.io'"
	export CI_REGISTRY="ghcr.io"
fi

if test -z "$GITHUB_USERNAME"
then
	echo "ERROR: GITHUB_USERNAME variable must be defined, exiting"
	exit -1
fi

if test -z "$GITHUB_PASSWORD"
then
	echo "ERROR: GITHUB_PASSWORD variable must be defined, exiting"
	exit -1
fi

if test -z "$GPU_DEVICE"
then
	echo "ERROR: GPU_DEVICE variable must be defined, exiting"
	exit -1
fi

if test -z "$DOMAIN"
then
	echo "ERROR: DOMAIN variable must be defined, exiting"
	exit -1
fi

if test -z "$ROOT_PATH"
then
	echo "ERROR: ROOT_PATH variable must be defined inside action secrets '{MODE}_CONFIG' variable, exiting"
	exit -1
fi

echo "INFO: stopping already running docker container"
sshpass -p "$SSH_PASSWORD" ssh -p 13402 -o StrictHostKeyChecking=no "$SSH_USER"@"$DOMAIN" "docker stop $SSH_USER-iks || echo 'No containers available to stop'" # nosemgrep
sshpass -p "$SSH_PASSWORD" ssh -p 13402 -o StrictHostKeyChecking=no "$SSH_USER"@"$DOMAIN" "docker container prune -f || echo 'No stopped containers to delete'" # nosemgrep

echo "INFO: copying .env file to server"
sshpass -p "$SSH_PASSWORD" ssh -p 13402 -o StrictHostKeyChecking=no "$SSH_USER"@"$DOMAIN" "rm -rf /home/$SSH_USER/.env" # nosemgrep
sshpass -p "$SSH_PASSWORD" scp -P 13402 -o StrictHostKeyChecking=no .env "$SSH_USER"@"$DOMAIN":/home/$SSH_USER/.env # nosemgrep

echo "INFO: pulling docker images"
sshpass -p "$SSH_PASSWORD" ssh -p 13402 -o StrictHostKeyChecking=no "$SSH_USER"@"$DOMAIN" "echo $GITHUB_PASSWORD | docker login -u '$GITHUB_USERNAME' --password-stdin '$CI_REGISTRY'" # nosemgrep
sshpass -p "$SSH_PASSWORD" ssh -p 13402 -o StrictHostKeyChecking=no "$SSH_USER"@"$DOMAIN" "docker pull '$CI_REGISTRY_IMAGE':'$TAG'" # nosemgrep
sleep 10

echo "Deleting old images"
sshpass -p "$SSH_PASSWORD" ssh -p 13402 -o StrictHostKeyChecking=no "$SSH_USER"@"$DOMAIN" "docker system prune -f || echo 'No images to delete'" # nosemgrep
sshpass -p "$SSH_PASSWORD" ssh -p 13402 -o StrictHostKeyChecking=no "$SSH_USER"@"$DOMAIN" "docker images" # nosemgrep

echo "INFO: starting docker container"
sshpass -p "$SSH_PASSWORD" ssh -p 13402 -o StrictHostKeyChecking=no "$SSH_USER"@"$DOMAIN" "docker run --name $SSH_USER-iks --env-file /home/$SSH_USER/.env --gpus 'device=$GPU_DEVICE' -v $ROOT_PATH:$ROOT_PATH -d '$CI_REGISTRY_IMAGE':'$TAG'" # nosemgrep
