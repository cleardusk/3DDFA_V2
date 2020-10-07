# coding: utf-8

__author__ = 'cleardusk'

import sys
import argparse
import cv2
import yaml

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.tddfa_util import str2bool
from FaceBoxes.utils.timer import Timer


def main(args):
    _t = {
        'det': Timer(),
        'reg': Timer(),
        'recon': Timer()
    }

    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA
    if args.onnx:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        tddfa = TDDFA(**cfg)
        face_boxes = FaceBoxes()

    # Given a still image path and load to BGR channel
    img = cv2.imread(args.img_fp)
    print(f'Input image: {args.img_fp}')

    # Detect faces, get 3DMM params and roi boxes
    print(f'Input shape: {img.shape}')
    if args.warmup:
        print('Warmup by once')
        boxes = face_boxes(img)
        param_lst, roi_box_lst = tddfa(img, boxes)
        ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=args.dense_flag)

    for _ in range(args.repeated):
        img = cv2.imread(args.img_fp)

        _t['det'].tic()
        boxes = face_boxes(img)
        _t['det'].toc()

        n = len(boxes)
        if n == 0:
            print(f'No face detected, exit')
            sys.exit(-1)

        _t['reg'].tic()
        param_lst, roi_box_lst = tddfa(img, boxes)
        _t['reg'].toc()

        _t['recon'].tic()
        ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=args.dense_flag)
        _t['recon'].toc()

    mode = 'Dense' if args.dense_flag else 'Sparse'
    print(f"Face detection: {_t['det'].average_time * 1000:.2f}ms, "
          f"3DMM regression: {_t['reg'].average_time * 1000:.2f}ms, "
          f"{mode} reconstruction: {_t['recon'].average_time * 1000:.2f}ms")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The latency testing of still image of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-f', '--img_fp', type=str, default='examples/inputs/JianzhuGuo.jpg')
    parser.add_argument('--onnx', action='store_true', default=False)
    parser.add_argument('--warmup', type=str2bool, default='true')
    parser.add_argument('--dense_flag', type=str2bool, default='true')
    parser.add_argument('--repeated', type=int, default=32)

    args = parser.parse_args()
    main(args)
