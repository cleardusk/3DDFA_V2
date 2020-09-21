# coding: utf-8

__author__ = 'cleardusk'

import sys

sys.path.append('..')

import os.path as osp
import numpy as np
from utils.io import _load

make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)


class BFMModel(object):
    def __init__(self, bfm_fp):
        bfm = _load(bfm_fp)
        self.u = bfm.get('u').astype(np.float32)  # fix bug
        self.w_shp = bfm.get('w_shp').astype(np.float32)
        self.w_exp = bfm.get('w_exp').astype(np.float32)
        self.tri = bfm.get('tri')
        self.keypoints = bfm.get('keypoints').astype(np.long)  # fix bug
        w = np.concatenate((self.w_shp, self.w_exp), axis=1)
        self.w_norm = np.linalg.norm(w, axis=0)

        self.u_base = self.u[self.keypoints].reshape(-1, 1)
        self.w_shp_base = self.w_shp[self.keypoints]
        self.w_exp_base = self.w_exp[self.keypoints]


cfg_path = make_abs_path('../configs')
bfm = BFMModel(osp.join(cfg_path, 'bfm_noneck_v3.pkl'))  # you can change the bfm pkl path
tri = _load(osp.join(cfg_path, 'tri.pkl'))  # this tri is re-built for bfm_noneck_v3
