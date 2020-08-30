# coding: utf-8

__author__ = 'cleardusk'

import sys
import argparse
import cv2
import yaml

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
from utils.functions import draw_landmarks, get_suffix
from utils.tddfa_util import str2bool


def main(args):
    # Init TDDFA
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
    gpu_mode = args.mode == 'gpu'
    tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)

    # Init FaceBoxes
    face_boxes = FaceBoxes()

    # Given a still image path and load to BGR channel
    img = cv2.imread(args.img_fp)

    # Detect faces, get 3DMM params and roi boxes
    boxes = face_boxes(img)
    n = len(boxes)
    print(f'Detect {n} faces')
    if n == 0:
        print(f'No face detected, exit')
        sys.exit(-1)

    param_lst, roi_box_lst = tddfa(img, boxes)

    # Visualization and serialization
    dense_flag = args.opt in ('2d_dense', '3d')  # if opt is 2d_dense or 3d, reconstruct dense vertices
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)

    wfp = f'examples/results/{args.img_fp.split("/")[-1].replace(get_suffix(args.img_fp), "")}_{args.opt}.jpg'

    if args.opt == '2d_sparse':
        draw_landmarks(img, ver_lst, show_flag=args.show_flag, dense_flag=dense_flag, wfp=wfp)
    elif args.opt == '2d_dense':
        draw_landmarks(img, ver_lst, show_flag=args.show_flag, dense_flag=dense_flag, wfp=wfp)
    elif args.opt == '3d':
        render(img, ver_lst, alpha=0.6, show_flag=args.show_flag, wfp=wfp)
    else:
        raise Exception(f'Unknown opt {args.opt}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The demo of still image of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-f', '--img_fp', type=str, default='examples/inputs/trump_hillary.jpg')
    parser.add_argument('-m', '--mode', type=str, default='cpu', help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='2d_sparse', choices=['2d_sparse', '2d_dense', '3d'])
    parser.add_argument('--show_flag', type=str2bool, default='true', help='whether to show the visualization result')

    args = parser.parse_args()
    main(args)
