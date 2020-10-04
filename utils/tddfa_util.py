# coding: utf-8

__author__ = 'cleardusk'

import sys

sys.path.append('..')

import argparse
import numpy as np
import torch


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
    param: shape=(trans_dim+shape_dim+exp_dim,), i.e., 62 = 12 + 40 + 10
    """

    # pre-defined templates for parameter
    n = param.shape[0]
    if n == 62:
        trans_dim, shape_dim, exp_dim = 12, 40, 10
    elif n == 72:
        trans_dim, shape_dim, exp_dim = 12, 40, 20
    elif n == 141:
        trans_dim, shape_dim, exp_dim = 12, 100, 29
    else:
        raise Exception(f'Undefined templated param parsing rule')

    R_ = param[:trans_dim].reshape(3, -1)
    R = R_[:, :3]
    offset = R_[:, -1].reshape(3, 1)
    alpha_shp = param[trans_dim:trans_dim + shape_dim].reshape(-1, 1)
    alpha_exp = param[trans_dim + shape_dim:].reshape(-1, 1)

    return R, offset, alpha_shp, alpha_exp
