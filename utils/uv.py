# coding: utf-8

__author__ = 'cleardusk'

import sys

sys.path.append('..')

import cv2
import numpy as np
import os.path as osp
import scipy.io as sio

from Sim3DR import rasterize
from utils.functions import plot_image
from utils.io import _load
from utils.tddfa_util import _to_ctype

make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)


def load_uv_coords(fp):
    C = sio.loadmat(fp)
    uv_coords = C['UV'].copy(order='C').astype(np.float32)
    return uv_coords


def process_uv(uv_coords, uv_h=256, uv_w=256):
    uv_coords[:, 0] = uv_coords[:, 0] * (uv_w - 1)
    uv_coords[:, 1] = uv_coords[:, 1] * (uv_h - 1)
    uv_coords[:, 1] = uv_h - uv_coords[:, 1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1), dtype=np.float32)))  # add z
    return uv_coords


g_uv_coords = load_uv_coords(make_abs_path('../configs/BFM_UV.mat'))
indices = _load(make_abs_path('../configs/indices.npy'))  # todo: handle bfm_slim
g_uv_coords = g_uv_coords[indices, :]


def get_colors(img, ver):
    # nearest-neighbor sampling
    [h, w, _] = img.shape
    ver[0, :] = np.minimum(np.maximum(ver[0, :], 0), w - 1)  # x
    ver[1, :] = np.minimum(np.maximum(ver[1, :], 0), h - 1)  # y
    ind = np.round(ver).astype(np.int32)
    colors = img[ind[1, :], ind[0, :], :]  # n x 3

    return colors


def bilinear_interpolate(img, x, y):
    """
    https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python
    """
    x0 = np.floor(x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int32)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, img.shape[1] - 1)
    x1 = np.clip(x1, 0, img.shape[1] - 1)
    y0 = np.clip(y0, 0, img.shape[0] - 1)
    y1 = np.clip(y1, 0, img.shape[0] - 1)

    i_a = img[y0, x0]
    i_b = img[y1, x0]
    i_c = img[y0, x1]
    i_d = img[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa[..., np.newaxis] * i_a + wb[..., np.newaxis] * i_b + wc[..., np.newaxis] * i_c + wd[..., np.newaxis] * i_d


def uv_tex(img, ver_lst, tri, uv_h=256, uv_w=256, uv_c=3, show_flag=False, wfp=None):
    uv_coords = process_uv(g_uv_coords.copy(), uv_h=uv_h, uv_w=uv_w)

    res_lst = []
    for ver_ in ver_lst:
        ver = _to_ctype(ver_.T)  # transpose to m x 3
        colors = bilinear_interpolate(img, ver[:, 0], ver[:, 1]) / 255.
        # `rasterize` here serves as texture sampling, may need to optimization
        res = rasterize(uv_coords, tri, colors, height=uv_h, width=uv_w, channel=uv_c)
        res_lst.append(res)

    # concat if there more than one image
    res = np.concatenate(res_lst, axis=1) if len(res_lst) > 1 else res_lst[0]

    if wfp is not None:
        cv2.imwrite(wfp, res)
        print(f'Save visualization result to {wfp}')

    if show_flag:
        plot_image(res)

    return res
