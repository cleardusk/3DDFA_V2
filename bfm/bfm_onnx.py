# coding: utf-8

__author__ = 'cleardusk'

import sys

sys.path.append('..')

import os.path as osp
import numpy as np
import torch
import torch.nn as nn

from utils.io import _load, _numpy_to_cuda, _numpy_to_tensor

make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)


def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr


def _load_tri(bfm_fp):
    if osp.split(bfm_fp)[-1] == 'bfm_noneck_v3.pkl':
        tri = _load(make_abs_path('../configs/tri.pkl'))  # this tri/face is re-built for bfm_noneck_v3
    else:
        tri = _load(bfm_fp).get('tri')

    tri = _to_ctype(tri.T).astype(np.int32)
    return tri


class BFMModel_ONNX(nn.Module):
    """BFM serves as a decoder"""

    def __init__(self, bfm_fp, shape_dim=40, exp_dim=10):
        super(BFMModel_ONNX, self).__init__()

        _to_tensor = _numpy_to_tensor

        # load bfm
        bfm = _load(bfm_fp)

        u = _to_tensor(bfm.get('u').astype(np.float32))
        self.u = u.view(-1, 3).transpose(1, 0)
        w_shp = _to_tensor(bfm.get('w_shp').astype(np.float32)[..., :shape_dim])
        w_exp = _to_tensor(bfm.get('w_exp').astype(np.float32)[..., :exp_dim])
        w = torch.cat((w_shp, w_exp), dim=1)
        self.w = w.view(-1, 3, w.shape[-1]).contiguous().permute(1, 0, 2)

        # self.u = _to_tensor(bfm.get('u').astype(np.float32))  # fix bug
        # w_shp = _to_tensor(bfm.get('w_shp').astype(np.float32)[..., :shape_dim])
        # w_exp = _to_tensor(bfm.get('w_exp').astype(np.float32)[..., :exp_dim])
        # self.w = torch.cat((w_shp, w_exp), dim=1)

        # self.keypoints = bfm.get('keypoints').astype(np.long)  # fix bug
        # self.u_base = self.u[self.keypoints].reshape(-1, 1)
        # self.w_shp_base = self.w_shp[self.keypoints]
        # self.w_exp_base = self.w_exp[self.keypoints]

    def forward(self, *inps):
        R, offset, alpha_shp, alpha_exp = inps
        alpha = torch.cat((alpha_shp, alpha_exp))
        # pts3d = R @ (self.u + self.w_shp.matmul(alpha_shp) + self.w_exp.matmul(alpha_exp)). \
        #     view(-1, 3).transpose(1, 0) + offset
        # pts3d = R @ (self.u + self.w.matmul(alpha)).view(-1, 3).transpose(1, 0) + offset
        pts3d = R @ (self.u + self.w.matmul(alpha).squeeze()) + offset
        return pts3d


def convert_bfm_to_onnx(bfm_onnx_fp, shape_dim=40, exp_dim=10):
    # print(shape_dim, exp_dim)
    bfm_fp = bfm_onnx_fp.replace('.onnx', '.pkl')
    bfm_decoder = BFMModel_ONNX(bfm_fp=bfm_fp, shape_dim=shape_dim, exp_dim=exp_dim)
    bfm_decoder.eval()

    # dummy_input = torch.randn(12 + shape_dim + exp_dim)
    dummy_input = torch.randn(3, 3), torch.randn(3, 1), torch.randn(shape_dim, 1), torch.randn(exp_dim, 1)
    R, offset, alpha_shp, alpha_exp = dummy_input
    torch.onnx.export(
        bfm_decoder,
        (R, offset, alpha_shp, alpha_exp),
        bfm_onnx_fp,
        input_names=['R', 'offset', 'alpha_shp', 'alpha_exp'],
        output_names=['output'],
        dynamic_axes={
            'alpha_shp': [0],
            'alpha_exp': [0],
        },
        do_constant_folding=True
    )
    print(f'Convert {bfm_fp} to {bfm_onnx_fp} done.')


if __name__ == '__main__':
    convert_bfm_to_onnx('../configs/bfm_noneck_v3.onnx')
