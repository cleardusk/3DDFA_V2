# coding: utf-8

__author__ = 'cleardusk'

import sys

sys.path.append('..')

import argparse
import numpy as np
import torch


# use global for accelerating
# u_base, w_shp_base, w_exp_base = bfm.u_base, bfm.w_shp_base, bfm.w_exp_base
# u, w_shp, w_exp = bfm.u, bfm.w_shp, bfm.w_exp
# tri = _to_ctype(tri.T).astype(np.int32)

def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def load_model(model, checkpoint_fp):
    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        kc = k.replace('module.', '')
        if kc in model_dict.keys():
            model_dict[kc] = checkpoint[k]
        if kc in ['fc_param.bias', 'fc_param.weight']:
            model_dict[kc.replace('_param', '')] = checkpoint[k]

    model.load_state_dict(model_dict)
    return model


class ToTensorGjz(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NormalizeGjz(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor.sub_(self.mean).div_(self.std)
        return tensor


def similar_transform(pts3d, roi_box, size):
    pts3d[0, :] -= 1  # for Python compatibility
    pts3d[2, :] -= 1
    pts3d[1, :] = size - pts3d[1, :]

    sx, sy, ex, ey = roi_box
    scale_x = (ex - sx) / size
    scale_y = (ey - sy) / size
    pts3d[0, :] = pts3d[0, :] * scale_x + sx
    pts3d[1, :] = pts3d[1, :] * scale_y + sy
    s = (scale_x + scale_y) / 2
    pts3d[2, :] *= s
    pts3d[2, :] -= np.min(pts3d[2, :])
    return np.array(pts3d, dtype=np.float32)


def _parse_param(param):
    """matrix pose form
    param: shape=(62,), 62 = 12 + 40 + 10
    scale may lie in R or alpha_shp + alpha_exp?
    """
    R_ = param[:12].reshape(3, -1)
    R = R_[:, :3]
    offset = R_[:, -1].reshape(3, 1)
    alpha_shp = param[12:52].reshape(-1, 1)
    alpha_exp = param[52:].reshape(-1, 1)

    return R, offset, alpha_shp, alpha_exp

# def recon_sparse(param, roi_box, size):
#     """68 3d landmarks reconstruction from 62: matrix pose form"""
#     R, offset, alpha_shp, alpha_exp = _parse_param(param)
#     pts3d = R @ (u_base + w_shp_base @ alpha_shp + w_exp_base @ alpha_exp).reshape(3, -1, order='F') + offset
#     pts3d = similar_transform(pts3d, roi_box, size)
#     return pts3d
#
#
# def recon_dense(param, roi_box, size):
#     """Dense points reconstruction: 53215 points"""
#     R, offset, alpha_shp, alpha_exp = _parse_param(param)
#     pts3d = R @ (u + w_shp @ alpha_shp + w_exp @ alpha_exp).reshape(3, -1, order='F') + offset
#     pts3d = similar_transform(pts3d, roi_box, size)
#     return pts3d
