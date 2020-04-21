# autotosis
Training a standard imagenet architecture like ResNet-18 for a different classification task isn't exactly cutting edge, but it's a fun way to wait for other (more important/serious/"real") models to train.

These days I find myself ~~watching a lot of twitch streamers (e.g., [Artosis](twitch.tv/artosis))~~ with a lot of extra time while waiting for models to train.
Many twitch streamers have extensive community-curated clips/highlights of their broadcasts.
Compilations of these clips are entertaining (NSFW) [1](https://www.youtube.com/watch?v=ykvlpUbGy6w) [2](https://www.youtube.com/watch?v=bBevrkgI5uc), but building them requires human eyeballs and effort.

From here, a fun experiment from a few questions:
+ What's the minimal amount of data I need to get a model like ResNet-18 off-the-ground for a binary prediction task with clips?
+ What's the minimal amount of hardware I need to prototype such a model?
+ Can this be done fully automatically with no manual video editing?

In this case, I chose the binary task to be the one used to generate the Artosis videos (e.g., Rage or Not Rage), relying on his face cam as the only input data.
It turns out that you need relatively few labels to get a model like ResNet-18 off the ground.
With only about three dozen "labeled" clips, the model was already producing usable predictions.

Here, "labeling" is just simply roughly annotating which segments of each clip corresponded to Rage Moments, and the corresponding frames for the model are extracted via `ffmpeg`.


All my other machines (that have GPUs) are currently busy at the moment doing "real" work, so I tried training this ResNet-18 on my ancient i5-3570K (it doesn't even have AVX2 extensions!) machine.
However, it turns out that this is still fast enough to train a 128x128 input resolution model in a few days.
For longer video clips, inference can be "sped up" by subsampling the video (e.g., predictions at << 60fps).


Finally, `ffmpeg` remains clunky but powerful and fast, and it was wholly able to accomplish all of the tasks needed to train the model, ingest video for predictions, and re-slice videos to automatically generate highlight compilations.
In more detail, this includes:
+ Extracting frames from video (.mp4s to many .jpgs)
+ Overlaying dynamic text at a fine granuarlity to video clips
+ Trimming long videos to regions of interest according to model predictions
+ Concatenating clips of interest into longer highlight reels.

## Dependencies
+ python 3.6
+ pytorch + torchvision
+ pillow
+ numpy
+ progress
+ ffmpeg
+ ffmpeg-python
