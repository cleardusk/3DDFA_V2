# coding: utf-8

__author__ = 'cleardusk'

import argparse
import imageio
import numpy as np
from tqdm import tqdm
import yaml
from collections import deque

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
# from utils.render_ctypes import render
from utils.functions import cv_draw_landmark, get_suffix


def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()

    # Given a video path
    fn = args.video_fp.split('/')[-1]
    reader = imageio.get_reader(args.video_fp)

    fps = reader.get_meta_data()['fps']
    suffix = get_suffix(args.video_fp)
    video_wfp = f'examples/results/videos/{fn.replace(suffix, "")}_{args.opt}_smooth.mp4'
    writer = imageio.get_writer(video_wfp, fps=fps)

    # the simple implementation of average smoothing by looking ahead by n_next frames
    # assert the frames of the video >= n
    n_pre, n_next = args.n_pre, args.n_next
    n = n_pre + n_next + 1
    queue_ver = deque()
    queue_frame = deque()

    # run
    dense_flag = args.opt in ('2d_dense', '3d',)
    pre_ver = None
    for i, frame in tqdm(enumerate(reader)):
        if args.start > 0 and i < args.start:
            continue
        if args.end > 0 and i > args.end:
            break

        frame_bgr = frame[..., ::-1]  # RGB->BGR

        if i == 0:
            # detect
            boxes = face_boxes(frame_bgr)
            boxes = [boxes[0]]
            param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            # refine
            param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy='landmark')
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            # padding queue
            for _ in range(n_pre):
                queue_ver.append(ver.copy())
            queue_ver.append(ver.copy())

            for _ in range(n_pre):
                queue_frame.append(frame_bgr.copy())
            queue_frame.append(frame_bgr.copy())

        else:
            param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver], crop_policy='landmark')

            roi_box = roi_box_lst[0]
            # todo: add confidence threshold to judge the tracking is failed
            if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                boxes = face_boxes(frame_bgr)
                boxes = [boxes[0]]
                param_lst, roi_box_lst = tddfa(frame_bgr, boxes)

            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            queue_ver.append(ver.copy())
            queue_frame.append(frame_bgr.copy())

        pre_ver = ver  # for tracking

        # smoothing: enqueue and dequeue ops
        if len(queue_ver) >= n:
            ver_ave = np.mean(queue_ver, axis=0)

            if args.opt == '2d_sparse':
                img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave)  # since we use padding
            elif args.opt == '2d_dense':
                img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave, size=1)
            elif args.opt == '3d':
                img_draw = render(queue_frame[n_pre], [ver_ave], tddfa.tri, alpha=0.7)
            else:
                raise ValueError(f'Unknown opt {args.opt}')

            writer.append_data(img_draw[:, :, ::-1])  # BGR->RGB

            queue_ver.popleft()
            queue_frame.popleft()

    # we will lost the last n_next frames, still padding
    for _ in range(n_next):
        queue_ver.append(ver.copy())
        queue_frame.append(frame_bgr.copy())  # the last frame

        ver_ave = np.mean(queue_ver, axis=0)

        if args.opt == '2d_sparse':
            img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave)  # since we use padding
        elif args.opt == '2d_dense':
            img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave, size=1)
        elif args.opt == '3d':
            img_draw = render(queue_frame[n_pre], [ver_ave], tddfa.tri, alpha=0.7)
        else:
            raise ValueError(f'Unknown opt {args.opt}')

        writer.append_data(img_draw[..., ::-1])  # BGR->RGB

        queue_ver.popleft()
        queue_frame.popleft()

    writer.close()
    print(f'Dump to {video_wfp}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The smooth demo of video of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-f', '--video_fp', type=str)
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('-n_pre', default=1, type=int, help='the pre frames of smoothing')
    parser.add_argument('-n_next', default=1, type=int, help='the next frames of smoothing')
    parser.add_argument('-o', '--opt', type=str, default='2d_sparse', choices=['2d_sparse', '2d_dense', '3d'])
    parser.add_argument('-s', '--start', default=-1, type=int, help='the started frames')
    parser.add_argument('-e', '--end', default=-1, type=int, help='the end frame')
    parser.add_argument('--onnx', action='store_true', default=False)

    args = parser.parse_args()
    main(args)
