#from  autotosis
Training a standard imagenet architecture like ResNet-18 for a different classification task isn't exactly cutting edge, but it's a fun way to wait for other (more important/serious/"real") models to train.

These days I find myself ~~watching a lot of twitch streamers (e.g., [Artosis](twitch.tv/artosis))~~ with a lot of extra time while waiting for models to train.
Artosis happens to have an extensive selection of high quality community-curated clips.
Compilations of these clips are entertaining (EXPLICIT language) [1](https://www.youtube.com/watch?v=ykvlpUbGy6w) [2](https://www.youtube.com/watch?v=bBevrkgI5uc), but building them requires human eyeballs and effort.

From here, a fun experiment arises out of a few questions:
- What's the minimal amount of data I need to get a model like ResNet-18 off-the-ground for a binary prediction task with clips?
- What's the minimal amount of hardware I need to prototype such a model?
- Can this be done fully automatically with no manual video editing?

In this case, I chose the binary task to be the one used to generate the Artosis videos (e.g., Rage or Not Rage), relying on his face cam as the only input data.
It turns out that you need relatively few labels to get a model like ResNet-18 off the ground.
With only about three dozen "labeled" clips, the model was already producing usable predictions.

Still, one issue that I didn't anticipate is that the "distribution" of frames is different when considering clips vs. full stream sessions or VODs.
Community-sourced clips have a high likelihood of having rage or at least something interesting---they've been selected for a reason.
However, VODs may have long periods of little to no interesting reactions from the streamer, and getting the false-positive rate of the model down can be tricky.
I'm currently doing a long "student-teacher" manual loop: I run the model on a longer VOD (~a few hours of video), the model gives back some segments it thinks are rage/interesting, and I add the segments that are clearly predicted incorrectly back into the training data.
Another hack is to use precision as the validation criteria for selecting the best model instead of outright accuracy when the data is imbalanced.


Here, "labeling" is just simply roughly annotating which segments of each clip corresponded to Rage Moments, and the corresponding frames for the model are extracted via `ffmpeg`.
Note that this kind of labeling basically also causes the model to label smiling as rage, becuase there are some good rage moments where Artosis is laughing.


All my other machines (that have GPUs) are currently busy at the moment doing "real" work, so I tried training this ResNet-18 on my ancient i5-3570K (it doesn't even have AVX2 extensions!) machine.
However, it turns out that this is still fast enough to train a 128x128 input resolution model in a few days.
For longer video clips, inference can be "sped up" by subsampling the video (e.g., predictions at << 60fps).


Finally, `ffmpeg` remains clunky but powerful and fast (for most tasks), and it was wholly able to accomplish all of the tasks needed to train the model, ingest video for predictions, and re-slice videos to automatically generate highlight compilations.
In more detail, this includes:
- Extracting frames from video (.mp4s to many .jpgs)
- Overlaying dynamic text at a fine granuarlity to video clips
- Trimming long videos to regions of interest according to model predictions
- Concatenating clips of interest into longer highlight reels.

## Dependencies
- python 3.6
- pytorch + torchvision
- pillow
- numpy
- progress
- ffmpeg
- ffmpeg-python

## other technical fluff
### input processing
I'm not very familiar with torchvision's [video offerings](https://pytorch.org/docs/stable/torchvision/io.html), so autotosis is built off the back of ffmpeg and PIL.
For training, each video is processed into 128x128 frames by first extracting each frame (at 60fps) and cropping to the artoFace region.
We can speed up the inference time by decreasing the framerate at which frames are dumped (e.g., from 60fps down to 4fps).

### training time/hardware setup
ArtosisNet was prototyped on a destitute and ancient 7-year old PC running Windows + WSL.
If I had the resources, I would use another machine, but all of my lab boxes are busy with "real" work.
One major drawback of WSL is that there is no GPU/CUDA support, so ArtosisNet was trained entirely on *CPU*, an aging i5-3570K slightly overclocked to 4 GHz.
At a batch size of 32 and frugal input resolution of 128x128, ArtosisNet (with its ResNet-18-based architecture) takes approximately 2 seconds per batch when the machine is not busy with other tasks.
Training for ~90 epochs takes several days on this machine. 


Somewhat surprisingly, the current bottleneck seems to be the ffmepg encoding pipeline when extracting multiple highlight clips from an input video.
My ffmpeg-fu is nonexistent, so I'm relying on an example posted in an issue thread on the ffmpeg-python repo [here](https://github.com/kkroening/ffmpeg-python/issues/184).
While it works, there seems to be a lot of wasted processing on frames irrelevant to the final output, resulting to an end-to-end processing time of multiple hours for a ~5 hour stream session VOD on my slow machine.


### on splitting training/validation data
One classical issue of classifying time-series data is managing the likelihood of high temporal redundancy of the data when attempting to estimate the performance of a model.
This issue is a well-known problem in [neuroscience settings that collect EEG data](https://arxiv.org/abs/1812.07697).
In other words, even with Artosis's extremely high mouth actions per minute, at a framerate of 30 or 60fps, adjacent frames are likely to be highly correlated, so if we naively split the data into training and validation sets without accounting for temporal redundancy, the classifier can fool us into thinking it has high performance by simply memorizing the training data.
To mitigate this effect, we split training and validation data at the granularity of individual _clips_ or video segments, rather than frames.
