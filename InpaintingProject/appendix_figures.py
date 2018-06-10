# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import offsetbox


def show_usps(data):
    """ Fonction nous permettant de visualiser les doonées USPS

    :param data: Given USPS data

    """

    plt.imshow(data.reshape((16, 16)), interpolation="nearest", cmap="YlGnBu")
    plt.colorbar()
    plt.show()


def show_im(im, title="", ax=plt):
    """ Plot the image given im matrix

    :param im: matrix or vector
    :param title: Title of the image
    :param ax: The ax where to plot

    """

    ax.imshow(im)
    try:
        ax.title(title)
    except BaseException:
        ax.set_title(title)


def plot_components(data, model, images=None, ax=None,
                    thumb_frac=0.05, cmap='gray'):
    """ Beautiful plot of writtings for each number

    :param data: The given data, 256 pixels for each line
    :param model:
    :param images: reshaped data to (16, 16)
    :param ax:
    :param thumb_frac:
    :param cmap: colormap
    :returns: Plots
    :rtype: void

    """

    ax = ax or plt.gca()

    proj = model.fit_transform(data)
    ax.plot(proj[:, 0], proj[:, 1], '.k')

    if images is not None:
        min_dist_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
        shown_images = np.array([2 * proj.max(0)])
        for i in range(data.shape[0]):
            dist = np.sum((proj[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, proj[i]])
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i], cmap=cmap),
                proj[i])
            ax.add_artist(imagebox)


def plot_dico_patches(dico, title="", ncols=5):
    """ PLot all patches of the given dictionnary

    :param dico: The given dictionnary
    :param ncols: Number of columns of the plot

    """

    if len(dico) == 0:
        raise Exception("Empty dictionnary problem !")

    nrows = int(len(dico) / ncols)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True)

    imgs, index = list(dico.values()), 0
    for i in range(nrows):
        for j in range(ncols):
            ax[i, j].axis('off')
            show_im(imgs[index], ax=ax[i, j])
            index += 1

    fig.subplots_adjust(hspace=0, wspace=0)
    fig.suptitle(title)
