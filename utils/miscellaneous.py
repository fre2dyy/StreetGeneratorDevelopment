import numpy as np
from PIL import Image
import PIL



def imagetonumpy(picture):
    np_img = np.asarray(picture)
    np_img = np_img.astype(float)
    np_img_noa = np_img[:, :, 0:3]
    np_img_noa = np_img_noa / 255
    return np_img_noa


def numpytoimage(numpy):
    numpy = numpy * 255
    n_img_noa_original = Image.fromarray(numpy.astype(np.uint8))
   # n_img_noa_original_inv = PIL.ImageOps.invert(n_img_noa_original)
    return n_img_noa_original


def change_contrast(img, level):
    # img = Image.open(img)
    img.load()

    factor = (259 * (level + 255)) / (255 * (259 - level))

    def contrast(c):
        return 128 + factor * (c - 128)

    return img.point(contrast)