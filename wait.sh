#!/bin/bash
while [ $(nvidia-smi | grep python3 | wc -l) -gt 0  ]
do
    echo 'waiting at' $(date) >> wait.log
    sleep 5
done
echo 'finished waiting at' $(date) >> wait.log
