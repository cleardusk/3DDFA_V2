# coding: utf-8

__author__ = 'cleardusk'

import argparse
import cv2
import yaml

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.functions import draw_landmarks, get_suffix
from utils.tddfa_util import str2bool


def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.FullLoader)
    gpu_mode = args.mode == 'gpu'
    tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)

    # Initialize FaceBoxes
    face_boxes = FaceBoxes()

    # Given a still image path and load to BGR channel
    img = cv2.imread(args.img_fp)

    # Detect faces, get 3DMM params and roi boxes
    boxes = face_boxes(img)
    print(f'Detect {len(boxes)} faces')
    param_lst, roi_box_lst = tddfa(img, boxes)

    # Visualization
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=args.dense_flag)
    wfp = f'examples/results/{args.img_fp.split("/")[-1].replace(get_suffix(args.img_fp), "")}_dense{args.dense_flag}.jpg'

    draw_landmarks(img, ver_lst, show_flag=args.show_flag, dense_flag=args.dense_flag, wfp=wfp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The demo of still image of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-f', '--img_fp', type=str)
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--show_flag', default='true', type=str2bool, help='whether show the visualization result')
    parser.add_argument('--dense_flag', default='true', type=str2bool, help='whether reconstructing dense')

    args = parser.parse_args()
    main(args)
