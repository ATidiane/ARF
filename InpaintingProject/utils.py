# -*- coding: utf-8 -*-

import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from tqdm import tqdm, trange


def read_im(fn):
    """ Read an image and transform it to HSV

    :param fn: path to file
    :returns: a tensor
    :rtype: numpy.ndarray

    """

    return plt.imread(fn)


def normalize(im, a, b):
    """ Normalize the matrix between a and b

    :param im: tensor or vectors
    :param a: lower interval value for the normalization
    :param b: upper interval value for the normalization
    :returns: tensor or vectors
    :rtype: numpy.array

    """

    lmin = float(im.min())
    lmax = float(im.max())

    return a + ((im - lmin) * (b - a) / (lmax - lmin))


def transform_to_2D(im):
    """ Convert image 3D to 2D

    :param im: a tensor
    :returns: an array of vectors
    :rtype: numpy.ndarray

    """

    return np.array([np.ravel(im[i]) for i in range(im.shape[0])])


def transform_to_3D(im):
    """ Convert image(patch) 2D to 3D

    :param im: an array of vectors
    :returns: a tensor
    :rtype: numpy.ndarray

    """

    l = int(im[i].shape[0] / 3)
    return np.array([im[i].reshape(l, 3) for i in range(im.shape[0])])


def get_patch(i, j, h, im):
    """ Return a patch given i and j coordonate and the height h

    :param i: line coordonate of pij (the pixel in the center of the patch
    :param j: column coordonate of pij
    :param h: height of the patch
    :param im: tensor or array of vectors
    :returns: a patch (tensor)
    :rtype: numpy.ndarray

    """

    k = 0 if int(i - (h / 2)) < 0 else int(i - (h / 2))
    l = im.shape[0] if int(i + (h / 2)) > im.shape[0] else int(i + (h / 2))
    m = 0 if int(j - (h / 2)) < 0 else int(j - (h / 2))
    n = im.shape[1] if int(j + (h / 2)) > im.shape[1] else int(j + (h / 2))

    return im[k:l, m:n]


def noise(im, prc):
    """ Noise prc percent of the given image

    :param im: a tensor
    :param prc: the percentage of pixels to noise
    :returns: a tensor
    :rtype: numpy.ndarray

    """

    im_noise = im.copy()
    nb_lines, nb_cols = im_noise.shape[0], im_noise.shape[1]
    nb_noise_pixels = int(nb_lines * nb_cols * prc / 100)

    with tqdm(total=nb_noise_pixels) as pbar:
        while nb_noise_pixels:
            x, y = random.randint(
                0, nb_lines - 1), random.randint(0, nb_cols - 1)
            if -100 not in im_noise[x, y]:
                im_noise[x, y] = -100
                nb_noise_pixels -= 1
                pbar.update(1)
            else:
                continue
    return im_noise


def delete_rect(im, i, j, h, w):
    """ Return the image with the part of given coordonates deleted

    :param im: a tensor
    :param i: line coordonate of pij
    :param j: column coordonate of pij
    :param h: height of the rectangle to delete
    :param w: width of the rectangle to delete
    :returns: a patch (tensor)
    :rtype: numpy.ndarray

    """

    img = im.copy()
    img[int(i - (h / 2)): int(i + (h / 2)),
        int(j - (w / 2)): int(j + (w / 2))] = -100
    return img


def image_patches(im, step=100):
    """ Returns two dictionaries, one containing patches of the image with no
        missing pixels and the other with only patches with missing pixels

    :param im: a tensor or an array of vectors
    :param step: add a patch to the correspondant dictionary each step pixels
    :returns: 2 dictionaries of patches
    :rtype: tuple of dict()

    """

    s = int(step / 2)
    nb_lines, nb_cols, nb_canaux = im.shape
    live_pixels, missing_pixels = dict(), dict()
    for i in range(s, nb_lines, step):
        for j in range(s, nb_cols, step):
            patch = get_patch(i, j, step, im)
            if -100 in patch:
                missing_pixels[i, j] = patch
            else:
                live_pixels[i, j] = patch

    return live_pixels, missing_pixels
