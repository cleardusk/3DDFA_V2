# coding: utf-8

from . import _init_paths
import numpy as np
import Sim3DR_Cython


def get_normal(vertices, triangles):
    normal = np.zeros_like(vertices, dtype=np.float32)
    Sim3DR_Cython.get_normal(normal, vertices, triangles, vertices.shape[0], triangles.shape[0])
    return normal


def rasterize(vertices, triangles, colors, bg=None,
              height=None, width=None, channel=None,
              reverse=False):
    if bg is not None:
        height, width, channel = bg.shape
    else:
        assert height is not None and width is not None and channel is not None
        bg = np.zeros((height, width, channel), dtype=np.uint8)

    buffer = np.zeros((height, width), dtype=np.float32) - 1e8

    if colors.dtype != np.float32:
        colors = colors.astype(np.float32)
    Sim3DR_Cython.rasterize(bg, vertices, triangles, colors, buffer, triangles.shape[0], height, width, channel,
                            reverse=reverse)
    return bg
