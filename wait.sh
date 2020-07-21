#!/bin/bash
gpu=$1
while [ $(gpustat | grep $gpu | awk '{print $10}') -gt 0  ]
do
    echo 'waiting at' $(date) #>> wait.log
    sleep 5
done
echo 'finished waiting at' $(date) >> wait.log
