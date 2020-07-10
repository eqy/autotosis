## timing data
"End to end" pipeline:
- ~5 hour Vod (in 1 hr chunks), inference on single GPU, batch size ~64, ~95 minutes/1+ hours at FRAMESKIP 15
- ~2 hour VOD (in 1 hr chunks), inference on CPU, batch size 1, with training running in background: ~625mins/10+ hours at FRAMESKIP 15
- ~4.5 hour VOD, inference on CPU, batch size 1, without training in background: ~259m/4+ hours at FRAMESKIP 30
- 396 min at FRAMESKIP 30

Frame extraction:
- frameskip=4, one-by-one: >> 15 minutes for single video
- frameskip=1, postprocessing: ~ <= 1 minute for single video/clip

## training time
- Throttling 1060: 512x256 : 0.562s per batch at batch size 32
- Throttling 1060: 768x256 : 1.060s per batch at batch size 32

## dataset size
- 2020/7/9 507 videos, train: 451199 individual images (229264 neg, 221935 pos), val: 135314 (70955 neg, 64359 pos)
- 2020/6/29 479 videos, train: 419339 individual images (215765 neg, 203754 pos), val: 118694 (64215 neg, 54479 pos)
- 2020/6/08 340 videos, train: 267899 individual images (159814 neg, 108085 pos), val: 70934 (44935 neg, 25999 pos)
- 2020/5/08 168 videos, train: 137257 individual images (92982 neg, 44275 pos), val: 30692 (22625 neg, 8067 pos)
- 2020/4/30 144 videos, train: 121657 individual images (77705 neg, 43952 pos), val: 26227 (18160 neg, 8067 pos)
- 2020/4/28 101 videos, train: 85482 individual images (56327 neg, 29155 pos), val: 14143 (11220 neg, 2923 pos)
- 2020/4/21 41 videos, train: 49956 individual images (26526 neg, 23631 pos), val: 2743 (2108 neg, 635 pos)
- 2020/4/19 36 videos, 48381 individual images (25414 neg, 22967 pos)
