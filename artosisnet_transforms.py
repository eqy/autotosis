import math
import numbers
import random
from PIL import Image
import torchvision.transforms as transforms


class RandomErasing2(transforms.RandomErasing):
    """Random Erasing with More Retries"""

    @staticmethod
    def get_params(img, scale, ratio, value=0):
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (Tensor): Tensor image of size (C, H, W) to be erased.
            scale: range of proportion of erased area against input image.
            ratio: range of aspect ratio of erased area.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_c, img_h, img_w = img.shape
        area = img_h * img_w

        for _ in range(1000):
            erase_area = random.uniform(scale[0], scale[1]) * area
            aspect_ratio = random.uniform(ratio[0], ratio[1])

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))

            if h < img_h and w < img_w:
                i = random.randint(0, img_h - h)
                j = random.randint(0, img_w - w)
                if isinstance(value, numbers.Number):
                    v = value
                elif isinstance(value, torch._six.string_classes):
                    v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
                elif isinstance(value, (list, tuple)):
                    v = torch.tensor(value, dtype=torch.float32).view(-1, 1, 1).expand(-1, h, w)
                return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img


class PartialRandomResizedCrop(transforms.RandomResizedCrop):
    """Crop only the top segment(s) of a stacked image"""
    
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR, segments=2, erase_scale=(0.1, 1.0), erase_ratio=(0.001, 100.0)):
        self.segments = segments
        self.erase_scale = erase_scale
        self.erase_ratio = erase_ratio
        self.erase = RandomErasing2(p=1.0, scale=self.erase_scale, ratio=self.erase_ratio)
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
