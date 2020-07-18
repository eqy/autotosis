#!/bin/bash
tunelist=$1
filelist=$2
output=$3
ffmpeg -f concat -i $tunelist -c copy tunetemp.mp3
ffmpeg -f concat -i $filelist -c copy vidtemp.mp4
ffmpeg -i vidtemp.mp4 -i tunetemp.mp3 -filter_complex "[0:a][1:a]amerge=inputs=2[a]" -map 0:v -map "[a]" -c:v copy -c:a libvorbis -ac 2 -shortest $output
