import argparse
import os
import sys
from clip import Clip

import ffmpeg

sys.setrecursionlimit(10**6)

def _join_videos(listpath, outputpath):
    (
        ffmpeg
        .input(listpath, format='concat', safe=0)
        .output(outputpath, c='copy')
        .overwrite_output()
        .run()
    )


def single_inference(args):
    clip = Clip(args.single_inference)
    clip.inference_frameskip = 2
    clip.inference('model_best.pth.tar')
    clip.generate_annotated(args.name)


def highlights(args):
    paths = list()
    idx = 0
    while True:
        path = f'{args.prefix}{idx}.mp4'
        if os.path.exists(f'{args.prefix}{idx}.mp4'):
            paths.append(path)
            idx += 1
        else:
            break

    paths = sorted(paths)

    if not len(paths):
        assert os.path.exists(args.prefix)

    if len(paths) >= 1:
        print("joining videos...")
        tempvideolist = 'tempvideolist'
        basename = os.path.splitext(args.name)[0]
        tempconcatvideo = f'temp{basename}.mp4'
        with open(tempvideolist, 'w') as f:
            for path in paths:
                f.write(f'file \'{path}\'\n')
        _join_videos(tempvideolist, tempconcatvideo)
        clip = Clip(tempconcatvideo)
        clip.inference('model_best.pth.tar')
        os.unlink(tempvideolist)
        clip.bin()
        print(clip.bins)
        clip.generate_highlights(output_path=args.name, percentile=args.percentile, threshold=args.threshold, delete_temp=args.delete_temp)
        os.unlink(tempconcatvideo)
    else:
        path = args.prefix
        clip = Clip(path)
        clip.inference('model_best.pth.tar')
        clip.bin()
        print(clip.bins)
        clip.generate_highlights(output_path=args.name, percentile=args.percentile, threshold=args.threshold, delete_temp=args.delete_temp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--single-inference", help="file name for single full inference")
    parser.add_argument("-p", "--prefix", help="prefix of file name to parse")
    parser.add_argument("-n", "--name", help="name of output", required=True)
    parser.add_argument("-d", "--delete-temp", help="delete temporary clips", action='store_true')
    parser.add_argument("--percentile", default=0.990, type=float)
    parser.add_argument("--threshold", default=0.500, type=float)
    args = parser.parse_args()


    assert args.single_inference is not None or args.prefix is not None
    if args.single_inference is not None:
        single_inference(args)
    else:
        highlights(args) 


if __name__ == '__main__':
    main()
