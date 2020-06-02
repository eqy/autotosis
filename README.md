# autotosis
Training a standard imagenet architecture like ResNet-18 for a different classification task isn't exactly cutting edge, but it's a fun way to wait for other (more important/serious/"real") models to train.

These days I find myself ~~watching a lot of twitch streamers~~ (e.g., [Artosis](twitch.tv/artosis)) with a lot of extra time while waiting for models to train.
Artosis happens to have an extensive selection of high quality community-curated clips.
Compilations of these clips are entertaining (EXPLICIT language) [1](https://www.youtube.com/watch?v=ykvlpUbGy6w) [2](https://www.youtube.com/watch?v=bBevrkgI5uc), but building them requires human eyeballs and effort.

From here, a fun experiment arises out of a few questions:
- What's the minimal amount of data I need to get a model like ResNet-18 off-the-ground for a binary prediction task with clips?
- What's the minimal amount of hardware I need to prototype such a model?
- Can we use this to generate highlight reels fully automatically with no manual video editing?

In this case, I chose the binary task to be the one used to generate the Artosis videos (e.g., Rage or Not Rage), relying on his face cam as the only input data.
It turns out that you need relatively few labels to get a model like ResNet-18 off the ground.
With only about three dozen "labeled" clips, the model was already producing usable predictions.

One issue that I didn't anticipate is that the "distribution" of frames is different when considering clips vs. full stream sessions or VODs.
Community-sourced clips have a high likelihood of having rage or at least something interesting---they've been selected for a reason.
However, VODs may have long periods of little to no interesting reactions from the streamer, and getting the false-positive rate of the model down can be tricky.
I'm currently doing a long ["student-teacher"](https://arxiv.org/abs/1911.04252) manual loop: I run the model on a longer VOD (~a few hours of video), the model gives back some segments it thinks are rage/interesting, and I add the segments that are clearly predicted incorrectly back into the training data.
Another hack is to use precision as the validation criteria for selecting the best model instead of outright accuracy when the data is imbalanced, but this strategy is brittle as it can lead to models with very low recall (also useless).


Here, "labeling" is just simply roughly annotating which segments of each clip corresponded to Rage Moments, and the corresponding frames for the model are extracted via `ffmpeg`.
Note that this kind of labeling basically also causes the model to label smiling as rage, becuase there are some good rage moments where Artosis is laughing.


All my other machines (that have GPUs) are currently busy at the moment doing "real" work, so I tried training this ResNet-18 on my ancient i5-3570K (it doesn't even have AVX2 extensions!) machine.
However, it turns out that this is still fast enough to train a 128x128 input resolution model.
For longer video clips, inference can be "sped up" by subsampling the video (e.g., predictions at << 60 fps).
(Update: I've started training a 256x256 version of ResNet-18 on a decomissioned P106-100 mining GPU I got from ebay for $80)


Finally, `ffmpeg` remains clunky but powerful and fast (for most tasks), and it was wholly able to accomplish all of the tasks needed to train the model, ingest video for predictions, and re-slice videos to automatically generate highlight compilations.
In more detail, this includes:
- Extracting frames from video (.mp4s to many .jpgs)
- Overlaying dynamic text at a fine granuarlity to video clips
- Trimming long videos to regions of interest according to model predictions
- Concatenating clips of interest into longer highlight reels.


One remaining issue is that of choosing when to cut/trim clips.
Currently, autotosis just uses hard 5-second boundaries for binning and trimming video segments, but this can create jarring cuts when things don't align well with 5-second intervals.
It may be possible to use some handcrafted heuristics to smooth things out (e.g., by tracking audio levels or monitoring the predictions of the model in a fine-grained way), but for now the focus is mainly on improving the prediction quality to generate usable highlight reels.


## Dependencies
- python 3.6
- pytorch + torchvision
- pillow
- numpy
- progress
- ffmpeg
- ffmpeg-python

## Workflow
- Add data to data.csv
- `python3 clip.py` to generate training data
- `python3 artosisnet.py data -b 32` example to train (seems like only a few epochs are needed before overfitting)
- `python3 inference -p input.mp4 -n output.mp4 --percentile 0.996 --threshold 0.70` example to generate highlights

## Testing Results
- 2020/05/31 ResNet-50 256x256 best val acc ~87.3% AP ?.??? (256x256 source, with more neg/pos examples)
- 2020/05/23 ResNet-50 256x256 best val acc ~90.4% AP 0.930 (256x256 source, with more neg/pos examples)
- 2020/05/12 ResNet-18 256x256 best val acc ~92.0% (256x256 source, with more neg/pos examples)
- 2020/05/09 ResNet-18 256x256 best val acc ~94.8% (256x256 source, with mostly more neg examples)
- 2020/05/07 ResNet-18 256x256 best val acc ~94.0% (256x256 source, with more neg examples)
- 2020/05/06 ResNet-18 256x256 best val acc ~91.9% (using 512x512 source images)
- 2020/05/04 ResNet-18 256x256 best val acc ~92.7% (using 256x256 source images)
- 2020/05/04 ResNet-18 128x128 best val acc ~91%
- 2020/05/04 ResNet-50 128x128 best val acc ~90.9%

## Testing TODOs (hyperparameters to tune)
- Removing normalization (usually for natural images, but might negatively affect spectrogram)
- Spectrogram cutoff frequency (currently 8KHz, but can we go down to 4/3KHz?)
- Resolution ?? 384px wide, 512px wide? (384+ is the limit of 6GB GPUs basically)
- Removing full screen view?

## other technical fluff
### input processing
I'm not very familiar with torchvision's [video offerings](https://pytorch.org/docs/stable/torchvision/io.html), so autotosis is built off the back of ffmpeg and PIL.
For training, each video is processed into 128x128 frames by first extracting each frame (at 60 fps) and cropping to the artoFace region.
We can speed up the inference time by decreasing the framerate at which frames are dumped (e.g., from 60 fps down to 4 fps).

### training time/hardware setup
ArtosisNet was prototyped on a destitute and ancient 7-year old PC running Windows + WSL.
If I had the resources, I would use another machine, but all of my lab boxes are busy with "real" work.
One major drawback of WSL is that there is no GPU/CUDA support, so ArtosisNet was trained entirely on *CPU*, an aging i5-3570K slightly overclocked to 4 GHz.
At a batch size of 32 and frugal input resolution of 128x128, ArtosisNet (with its ResNet-18-based architecture) takes approximately 2 seconds per batch when the machine is not busy with other tasks.
Training for ~90 epochs takes several days on this machine.
Update: it seems that with the current amount of data, only two epochs of training are needed before the model starts overfitting, so prototyping with CPU training remains reasonably quick.
Update 2: Having *slow* model training tying up my main desktop PC has gotten annoying enough that I invested in a "budget" training setup, R5 3600 + 2xP106-100 decommissioned mining GPUs from ebay.
Total cost of the build shipped was < $1100.


Somewhat surprisingly, the current bottleneck seems to be the ffmepg encoding pipeline when extracting multiple highlight clips from an input video.
My ffmpeg-fu is nonexistent, so I'm relying on an example posted in an issue thread on the ffmpeg-python repo [here](https://github.com/kkroening/ffmpeg-python/issues/184).
While it works, there seems to be a lot of wasted processing on frames irrelevant to the final output, resulting to an end-to-end processing time of multiple hours for a ~5 hour stream session VOD on my slow machine.


### on splitting training/validation data
One classical issue of classifying time-series data is managing the likelihood of high temporal redundancy of the data when attempting to estimate the performance of a model.
This issue is a well-known problem in [neuroscience settings that collect EEG data](https://arxiv.org/abs/1812.07697).
In other words, even with Artosis's extremely high mouth actions per minute, at a framerate of 30 or 60fps, adjacent frames are likely to be highly correlated, so if we naively split the data into training and validation sets without accounting for temporal redundancy, the classifier can fool us into thinking it has high performance by simply memorizing the training data.
To mitigate this effect, we split training and validation data at the granularity of individual _clips_ or video segments, rather than frames.
