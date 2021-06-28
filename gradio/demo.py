# before import, make sure FaceBoxes and Sim3DR are built successfully, e.g.,
import sys
from subprocess import call

def run_cmd(command):
    try:
        print(command)
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)
  

run_cmd("sh ./build.sh")

import cv2
import yaml

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
from utils.depth import depth
from utils.pncc import pncc
from utils.uv import uv_tex
from utils.pose import viz_pose
from utils.serialization import ser_to_ply, ser_to_obj
from utils.functions import draw_landmarks, get_suffix

import matplotlib.pyplot as plt
from skimage import io
import gradio as gr

# load config
cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)

# Init FaceBoxes and TDDFA, recommend using onnx flag
onnx_flag = True  # or True to use ONNX to speed up
if onnx_flag:    
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '4'
    from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
    from TDDFA_ONNX import TDDFA_ONNX

    face_boxes = FaceBoxes_ONNX()
    tddfa = TDDFA_ONNX(**cfg)
else:
    face_boxes = FaceBoxes()
    tddfa = TDDFA(gpu_mode=False, **cfg)
    


def inference (img):
    # face detection
    boxes = face_boxes(img)
    # regress 3DMM params
    param_lst, roi_box_lst = tddfa(img, boxes)
    # reconstruct vertices and render
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
    return render(img, ver_lst, tddfa.tri, alpha=0.6, show_flag=False);    


title = "3DDFA_V2"
description = "demo for 3DDFA_V2. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2009.09960'>Towards Fast, Accurate and Stable 3D Dense Face Alignment</a> | <a href='https://github.com/cleardusk/3DDFA_V2'>Github Repo</a></p>"

gr.Interface(
    inference, 
    [gr.inputs.Image(type="numpy", label="Input")], 
    gr.outputs.Image(type="numpy", label="Output"),
    title=title,
    description=description,
    article=article
    ).launch(debug=True)
