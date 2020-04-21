from clip import Clip


def main():
    test = 'rawdata/whatswrongwithyou.mp4'
    clip = Clip(test)
    clip.inference('checkpoint.pth.tar') 


if __name__ == '__main__':
    main()
