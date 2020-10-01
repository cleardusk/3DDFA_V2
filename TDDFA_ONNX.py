# coding: utf-8

__author__ = 'cleardusk'

import os
import os.path as osp
import time
import numpy as np
import cv2
import onnxruntime

from utils.onnx import convert_to_onnx
from utils.io import _load
from utils.functions import (
    crop_img, parse_roi_box_from_bbox, parse_roi_box_from_landmark,
)
from utils.tddfa_util import recon_dense, recon_sparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)


class TDDFA_ONNX(object):
    """TDDFA_ONNX: the ONNX version of Three-D Dense Face Alignment (TDDFA)"""

    def __init__(self, **kvs):
        # torch.set_grad_enabled(False)

        # config
        self.gpu_mode = kvs.get('gpu_mode', False)
        self.gpu_id = kvs.get('gpu_id', 0)
        self.size = kvs.get('size', 120)

        param_mean_std_fp = kvs.get(
            'param_mean_std_fp', make_abs_path(f'configs/param_mean_std_62d_{self.size}x{self.size}.pkl')
        )

        onnx_fp = kvs.get('onnx_fp', kvs.get('checkpoint_fp').replace('.pth', '.onnx'))

        # convert to onnx online if not existed
        if onnx_fp is None or not osp.exists(onnx_fp):
            print(f'{onnx_fp} does not exist, try to convert the `.pth` version to `.onnx` online')
            onnx_fp = convert_to_onnx(**kvs)

        self.session = onnxruntime.InferenceSession(onnx_fp, None)

        # params normalization config
        r = _load(param_mean_std_fp)
        self.param_mean = r.get('mean')
        self.param_std = r.get('std')

    def __call__(self, img_ori, objs, **kvs):
        # Crop image, forward to get the param
        param_lst = []
        roi_box_lst = []

        crop_policy = kvs.get('crop_policy', 'box')
        for obj in objs:
            if crop_policy == 'box':
                # by face box
                roi_box = parse_roi_box_from_bbox(obj)
            elif crop_policy == 'landmark':
                # by landmarks
                roi_box = parse_roi_box_from_landmark(obj)
            else:
                raise ValueError(f'Unknown crop policy {crop_policy}')

            roi_box_lst.append(roi_box)
            img = crop_img(img_ori, roi_box)
            img = cv2.resize(img, dsize=(self.size, self.size), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
            img = (img - 127.5) / 128.

            inp_dct = {'input': img}

            if kvs.get('timer_flag', False):
                end = time.time()
                param = self.session.run(None, inp_dct)[0]
                elapse = f'Inference time: {(time.time() - end) * 1000:.1f}ms'
                print(elapse)
            else:
                param = self.session.run(None, inp_dct)[0]

            param = param.flatten().astype(np.float32)
            param = param * self.param_std + self.param_mean  # re-scale
            param_lst.append(param)

        return param_lst, roi_box_lst

    def recon_vers(self, param_lst, roi_box_lst, **kvs):
        dense_flag = kvs.get('dense_flag', False)
        size = self.size

        ver_lst = []
        for param, roi_box in zip(param_lst, roi_box_lst):
            if kvs.get('timer_flag', False):
                end = time.time()

            if dense_flag:
                pts3d = recon_dense(param, roi_box, size)  # 38365 points
            else:
                pts3d = recon_sparse(param, roi_box, size)  # 68 points

            if kvs.get('timer_flag', False):
                elapse = f'Reconstruction time: {(time.time() - end) * 1000:.1f}ms'
                print(elapse)

            ver_lst.append(pts3d)

        return ver_lst
