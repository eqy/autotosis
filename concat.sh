#!/bin/bash
ffmpeg -f concat -i $1 -c copy $2
