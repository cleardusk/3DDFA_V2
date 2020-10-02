# coding: utf-8

__author__ = 'cleardusk'

import sys

sys.path.append('..')

import cv2
import numpy as np

from Sim3DR import rasterize
from utils.functions import plot_image
from .tddfa_util import _to_ctype


def depth(img, ver_lst, tri, show_flag=False, wfp=None, with_bg_flag=True):
    if with_bg_flag:
        overlap = img.copy()
    else:
        overlap = np.zeros_like(img)

    for ver_ in ver_lst:
        ver = _to_ctype(ver_.T)  # transpose

        z = ver[:, 2]
        z_min, z_max = min(z), max(z)

        z = (z - z_min) / (z_max - z_min)

        # expand
        z = np.repeat(z[:, np.newaxis], 3, axis=1)

        overlap = rasterize(ver, tri, z, bg=overlap)

    if wfp is not None:
        cv2.imwrite(wfp, overlap)
        print(f'Save visualization result to {wfp}')

    if show_flag:
        plot_image(overlap)

    return overlap
