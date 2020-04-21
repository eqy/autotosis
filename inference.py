import args
import os
from clip import Clip

# TODO: add prefix support, concat -> process -> concat

def main():
    #test = 'rawdata/whatthefrickvultures.mp4'
    #test = 'rawdata/whatswrongwithyou.mp4'
    #test = 'rawdata/amigonnamiss.mp4'
    #test = 'rawdata/winningtolosing.mp4'
    test = 'testvods/artosis-works-on-fun-damentals0.mp4'

    clip = Clip(test)
    clip.inference('checkpoint.pth.tar') 
    #clip.inference('model_best.pth.tar')
    #clip.generate_annotated('test' + os.path.basename(test))
    clip.bin()
    print(clip.bins)
    clip.generate_highlights()


if __name__ == '__main__':
    main()
