import argparse
import os
from clip import Clip

import ffmpeg


def _join_videos(listpath, outputpath):
    (
        ffmpeg
        .input(listpath, format='concat', safe=0)
        .output(outputpath, c='copy')
        .overwrite_output()
        .run()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prefix", help="prefix of file name to parse", required=True)
    parser.add_argument("-n", "--name", help="name of output", required=True)
    args = parser.parse_args()

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

    if len(paths) > 1:
        print("joining videos...")
        tempvideolist = 'tempvideolist'
        basename = os.path.splitext(args.name)[0]
        tempconcatvideo = f'temp{basename}.mp4'
        with open(tempvideolist, 'w') as f:
            for path in paths:
                f.write(f'file \'{path}\'\n')
        _join_videos(tempvideolist, tempconcatvideo)
        clip = Clip(tempconcatvideo)
        clip.inference('checkpoint.pth.tar') 
        os.unlink(tempvideolist)
        clip.bin()
        print(clip.bins)
        clip.generate_highlights()
        os.unlink(tempconcatvideo)
    else:
        os.unlink(tempconcatvideo)
    
    #clip.inference('model_best.pth.tar')
    #clip.generate_annotated('test' + os.path.basename(test))
    
    #test = 'rawdata/whatthefrickvultures.mp4'
    #test = 'rawdata/whatswrongwithyou.mp4'
    #test = 'rawdata/amigonnamiss.mp4'
    #test = 'rawdata/winningtolosing.mp4'
    #test = 'testvods/artosis-works-on-fun-damentals0.mp4'



if __name__ == '__main__':
    main()
