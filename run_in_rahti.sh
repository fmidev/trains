#!/bin/bash
MEM="64Gi"
CPU="8"
GPU="0"

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
    *)
          # unknown option
    ;;
esac
done

RUN_NAME=${NAME//[_:]/-}-$RANDOM
#RUN_NAME=${NAME//[_:]/-}
set -x
docker build -t tervo/$NAME -f $FILE .
docker push tervo/$NAME

if [ "$GPU" -gt 0 ]; then
  oc run $RUN_NAME --image tervo/$NAME --expose=true --port 8888 --replicas=1 --restart=Never --image-pull-policy='Always' --overrides='
  {
    "apiVersion": "v1",
    "kind": "Pod",
    "spec": {
      "containers": [
      {
        "env": [
        {
          "name": "NVIDIA_VISIBLE_DEVICES",
          "value": "all"
        },
        {
          "name": "NVIDIA_DRIVER_CAPABILITIES",
          "value": "compute,utility"
        },
        {
          "name": "NVIDIA_REQUIRE_CUDA",
          "value": "cuda>=8.0"
        }
        ],
        "image": "tervo/'$NAME'",
        "name": "'$RUN_NAME'",
        "resources": {
          "limits": {
            "cpu": "'$CPU'",
            "memory": "'$MEM'",
            "nvidia.com/gpu": '$GPU'
          },
          "requests": {
            "cpu": "'$CPU'",
            "memory": "'$MEM'",
            "nvidia.com/gpu": '$GPU'
          }
        },
        "volumeMounts": [{
          "mountPath": "/board",
          "name": "volboard"
        }]
      }
      ],
      "volumes": [{
        "name":"volboard",
        "persistentVolumeClaim": {
          "claimName": "vol-board"
        }
      }]
    }
  }'
else
  oc run $RUN_NAME --image tervo/$NAME --expose=true --port 8888 --replicas=1 --restart=Never --image-pull-policy='Always' --overrides='
  {
    "apiVersion": "v1",
    "kind": "Pod",
    "spec": {
      "containers": [
      {
        "env": [
        {
          "name": "NVIDIA_VISIBLE_DEVICES",
          "value": "all"
        },
        {
          "name": "NVIDIA_DRIVER_CAPABILITIES",
          "value": "compute,utility"
        },
        {
          "name": "NVIDIA_REQUIRE_CUDA",
          "value": "cuda>=8.0"
        }
        ],
        "image": "tervo/'$NAME'",
        "name": "'$RUN_NAME'",
        "resources": {
          "limits": {
            "cpu": "'$CPU'",
            "memory": "'$MEM'"
          },
          "requests": {
            "cpu": "'$CPU'",
            "memory": "'$MEM'"
          }
        },
        "volumeMounts": [{
          "mountPath": "/board",
          "name": "volboard"
        }]
      }
      ],
      "volumes": [{
        "name":"volboard",
        "persistentVolumeClaim": {
          "claimName": "vol-board"
        }
      }]
    }
  }'
fi
sleep 15
oc logs --follow $RUN_NAME
#oc expose service $RUN_NAME
