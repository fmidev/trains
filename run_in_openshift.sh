#!/bin/bash
FILE=$1
NAME=$2
TYPE=1
RUN_NAME=${NAME//[_:]/-}-$RANDOM
set -x
docker build -t tervo/$NAME -f $FILE .
docker push tervo/$NAME

CPU="15"
MEM="32Gi"
if [ -n "$3" ]; then
  TYPE=$3
  if [ "$TYPE" -eq 2 ]; then
    CPU="1"
    MEM="4Gi"
  fi
fi

if [ "$TYPE" -eq 3 ]; then
  oc run $RUN_NAME --image tervo/$NAME --expose=true --port 8888 --replicas=1 --restart=Never --image-pull-policy='Always' --overrides='
  {
    "apiVersion": "v1",
    "kind": "Pod",
    "spec": {
      "containers": [
      {
        "env": [
        {
          "name": "https_proxy",
          "value": "wwwproxy.fmi.fi:8080"
        },
        {
          "name": "http_proxy",
          "value": "wwwproxy.fmi.fi:8080"
        },
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
            "cpu": "2",
            "memory": "'$MEM'",
            "nvidia.com/gpu": 1
          },
          "requests": {
            "cpu": "2",
            "memory": "'$MEM'",
            "nvidia.com/gpu": 1
          }
        }
      }
      ],
      "volumes": [{
        "name":"volboard"
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
          "name": "https_proxy",
          "value": "wwwproxy.fmi.fi:8080"
        },
        {
          "name": "http_proxy",
          "value": "wwwproxy.fmi.fi:8080"
        },
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
        }
      }
      ],
      "volumes": [{
        "name":"volboard"
      }]
    }
  }'
fi


# oc run $RUN_NAME --image tervo/$NAME --expose=true --port 80 --replicas=1 --restart=Never --image-pull-policy='Always' --overrides='
# {
#   "apiVersion": "v1",
#   "kind": "Pod",
#   "spec": {
#     "containers": [
#       {
#         "env": [
#           {
#             "name": "https_proxy",
#             "value": "wwwproxy.fmi.fi:8080"
#           },
#           {
#             "name": "http_proxy",
#             "value": "wwwproxy.fmi.fi:8080"
#           },
#           {
#             "name": "NVIDIA_VISIBLE_DEVICES",
#             "value": "all"
#           },
#           {
#             "name": "NVIDIA_DRIVER_CAPABILITIES",
#             "value": "compute,utility"
#           },
#           {
#             "name": "NVIDIA_REQUIRE_CUDA",
#             "value": "cuda>=8.0"
#           }
#         ],
#         "image": "tervo/'$NAME'",
#         "name": "'$RUN_NAME'",
#         "resources": {
#           "limits": {
#             "cpu": "'$CPU'",
#             "memory": "'$MEM'",
#             "nvidia.com/gpu": 1
#           },
#           "requests": {
#             "cpu": "'$CPU'",
#             "memory": "'$MEM'",
#             "nvidia.com/gpu": 1
#           },
#         "volumeMounts": [{
#               "mountPath": "/board",
#               "name": "volboard"
#             }]
#         }
#       }
#     ],
#     "volumes": [{
#           "name":"volboard"
#     }]
#   }
# }'
oc expose service $RUN_NAME

#  "requests": {
#     "cpu": "'$CPU'",
#     "memory": "'$MEM'",
#     "nvidia.com/gpu": 1
#
# ,
#          "persistentVolumeClaim": {
#               "claimName": "vol-board"
#           }
