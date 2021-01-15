# coding: utf-8

"""
Borrowed from https://github.com/1996scarlet/Dense-Head-Pose-Estimation/blob/main/service/CtypesMeshRender.py

To use this render, you should build the clib first:
```
cd utils/asset
gcc -shared -Wall -O3 render.c -o render.so -fPIC
cd ../..
```
"""

import sys

sys.path.append('..')

import os.path as osp
import cv2
import numpy as np
import ctypes
from utils.functions import plot_image

make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)


class TrianglesMeshRender(object):
    def __init__(
            self,
            clibs,
            light=(0, 0, 5),
            direction=(0.6, 0.6, 0.6),
            ambient=(0.3, 0.3, 0.3)
    ):
        if not osp.exists(clibs):
            raise Exception(f'{clibs} not found, please build it first, by run '
                            f'"gcc -shared -Wall -O3 render.c -o render.so -fPIC" in utils/asset directory')

        self._clibs = ctypes.CDLL(clibs)

        self._light = np.array(light, dtype=np.float32)
        self._light = np.ctypeslib.as_ctypes(self._light)

        self._direction = np.array(direction, dtype=np.float32)
        self._direction = np.ctypeslib.as_ctypes(self._direction)

        self._ambient = np.array(ambient, dtype=np.float32)
        self._ambient = np.ctypeslib.as_ctypes(self._ambient)

    def __call__(self, vertices, triangles, bg):
        self.triangles = np.ctypeslib.as_ctypes(3 * triangles)  # Attention
        self.tri_nums = triangles.shape[0]

        self._clibs._render(
            self.triangles, self.tri_nums,
            self._light, self._direction, self._ambient,
            np.ctypeslib.as_ctypes(vertices),
            vertices.shape[0],
            np.ctypeslib.as_ctypes(bg),
            bg.shape[0], bg.shape[1]
        )


render_app = TrianglesMeshRender(clibs=make_abs_path('asset/render.so'))


def render(img, ver_lst, tri, alpha=0.6, show_flag=False, wfp=None, with_bg_flag=True):
    if with_bg_flag:
        overlap = img.copy()
    else:
        overlap = np.zeros_like(img)

    for ver_ in ver_lst:
        ver = np.ascontiguousarray(ver_.T)  # transpose
        render_app(ver, tri, bg=overlap)

    if with_bg_flag:
        res = cv2.addWeighted(img, 1 - alpha, overlap, alpha, 0)
    else:
        res = overlap

    if wfp is not None:
        cv2.imwrite(wfp, res)
        print(f'Save visualization result to {wfp}')

    if show_flag:
        plot_image(res)

    return res
