## timing data
"End to end" pipeline:
- ~2 hour VOD (in 1 hr chunks), inference on CPU, batch size 1, with training running in background: ~625mins/10+ hours at FRAMESKIP 15
- 396 min at FRAMESKIP 30

Frame extraction:
- frameskip=4, one-by-one: >> 15 minutes for single video
- frameskip=1, postprocessing: ~ <= 1 minute for single video/clip

## dataset size
- 2020/4/28 101 videos, train:  85482 individual images (56327 neg, 29155 pos), val: 14143 (11220 neg, 2923 pos)
- 2020/4/21 41 videos, train: 49956 individual images (26526 neg, 23631 pos), val: 2743 (2108 neg, 635 pos)
- 2020/4/19 36 videos, 48381 individual images (25414 neg, 22967 pos)
