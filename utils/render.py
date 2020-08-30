# coding: utf-8

__author__ = 'cleardusk'

import sys

sys.path.append('..')

import cv2
import matplotlib.pyplot as plt

from Sim3DR import RenderPipeline
from .tddfa_util import _to_ctype, tri

cfg = {
    'intensity_ambient': 0.3,
    'color_ambient': (1, 1, 1),
    'intensity_directional': 0.6,
    'color_directional': (1, 1, 1),
    'intensity_specular': 0.1,
    'specular_exp': 5,
    'light_pos': (0, 0, 5),
    'view_pos': (0, 0, 5)
}

render_app = RenderPipeline(**cfg)


def render(img, ver_lst, alpha=0.6, show_flag=False, wfp=None):
    overlap = img.copy()
    for ver_ in ver_lst:
        ver = _to_ctype(ver_.T)  # transpose
        overlap = render_app(ver, tri, overlap)

    res = cv2.addWeighted(img, 1 - alpha, overlap, alpha, 0)

    if show_flag:
        height, width = img.shape[:2]
        plt.figure(figsize=(12, height / width * 12))

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.axis('off')

        plt.imshow(res[..., ::-1])
        plt.show()

    if wfp is not None:
        cv2.imwrite(wfp, res)
        print(f'Save visualization result to {wfp}')

    return res
