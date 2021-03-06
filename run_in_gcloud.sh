#!/bin/bash

MEM="32"
CPU="16"
GPU="0"
PRE="0"

for i in "$@"
do
case $i in
    -m=*|--mem=*)
    MEM="${i#*=}"
    shift # past argument=value
    ;;
    -c=*|--cpu=*)
    CPU="${i#*=}"
    shift # past argument=value
    ;;
    -g=*|--gpu=*)
    GPU="${i#*=}"
    shift # past argument=value
    ;;
    -f=*|--file=*)
    FILE="${i#*=}"
    shift # past argument=value
    ;;
    -n=*|--name=*)
    NAME="${i#*=}"
    shift # past argument=value
    ;;
    -p=*|--preemptible=*)
    PRE="${i#*=}"
    shift # past argument=value
    ;;
    *)
          # unknown option
    ;;
esac
done

RUN_NAME=${NAME//[_:]/-}

set -x
set -e

PREEMPTIBLE=""
if [[ $PRE -eq 1 ]]; then
  PREEMPTIBLE="--preemptible"
fi

docker build -t gcr.io/trains-197305/$NAME -f $FILE .
gcloud docker -- push gcr.io/trains-197305/$NAME
#gcloud beta compute instances create-with-container $RUN_NAME --container-image eu.gcr.io/trains-197305/$NAME --custom-cpu $CPU --custom-memory $MEM --boot-disk-size 200 --preemptible --container-restart-policy never --tags http-server
gcloud beta compute instances create-with-container $RUN_NAME --container-image gcr.io/trains-197305/$NAME --custom-cpu $CPU --custom-memory $MEM --boot-disk-size 200 --container-restart-policy never --tags http-server $PREEMTIBLE
