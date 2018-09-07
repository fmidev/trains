#!/bin/bash

FILE=$1
NAME=$2
CONT_NAME=${NAME//[_:]/-}

docker build -t eu.gcr.io/trains-197305/$NAME -f $FILE .
gcloud docker -- push eu.gcr.io/trains-197305/$NAME
gcloud beta compute instances create-with-container $CONT_NAME --container-image eu.gcr.io/trains-197305/$NAME --custom-cpu 16 --custom-memory 32 --boot-disk-size 200 --preemptible --container-restart-policy never --tags http-server
