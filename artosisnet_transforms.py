from PIL import Image
import torchvision.transforms as transforms


class PartialRandomResizedCrop(transforms.RandomResizedCrop):
    """Crop only the top segment(s) of a stacked image"""
    
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR, segments=2, erase_scale=(0.02, 0.33), erase_ratio=(0.1, 10.0)):
        self.segments = segments
        self.erase_scale = erase_scale
        self.erase_ratio = erase_ratio
        self.erase = transforms.RandomErasing(p=1.0, scale=self.erase_scale, ratio=self.erase_ratio)
        self.totensor = transforms.ToTensor()
        self.topil = transforms.ToPILImage()
        super(PartialRandomResizedCrop, self).__init__(size, scale, ratio, interpolation)

    def __call__(self, img):
        width, height = img.size
        assert height % width == 0
        assert self.size[0] == width
        square_dim = self.size[0]
        img = img.copy()
        for idx in range(0, self.segments):
            assert (idx+1)*square_dim <= height
            tempimg = img.crop((0, idx*square_dim, square_dim, (idx+1)*square_dim))
            tempimgrandcrop = super(PartialRandomResizedCrop, self).__call__(tempimg)
            img.paste(tempimgrandcrop, (0, idx*square_dim))
        if self.segments*square_dim < height:
            tempimg = img.crop((0, self.segments*square_dim, square_dim, (self.segments+1)*square_dim))
            tempimgranderase = self.topil(self.erase(self.totensor(tempimg)))
            img.paste(tempimgranderase, (0, self.segments*square_dim)) 
        return img
    
    def __repr__(self):
        fmt_str = super(PartialRandomResizedCrop, self).__repr__()
        fmt_str += ', segments={0}'.format(self.segments)
        return fmt_str


def main():
    # quick test
    pth = 'data/train/1/youkiddingme_1187.jpg'
    pth2 = 'data_nosound/train/1/youkiddingme_1187.jpg'
    tr = PartialRandomResizedCrop(256, scale=(0.5, 1.0), segments=2)
    for i in range(0, 10):
        img = Image.open(pth)
        aug = tr(img)
        aug.save('temptest{0:d}.jpg'.format(i))
        img = Image.open(pth2)
        aug = tr(img)
        aug.save('temptest2{0:d}.jpg'.format(i))


if __name__ == '__main__':
    main()
