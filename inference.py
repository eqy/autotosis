from clip import Clip


def main():
    test = 'rawdata/whatthefrickvultures.mp4'
    clip = Clip(test)
    #clip.inference('checkpoint.pth.tar') 
    clip.inference('model_best.pth.tar')
    clip.generate_annotated('annotest.mp4')


if __name__ == '__main__':
    main()
