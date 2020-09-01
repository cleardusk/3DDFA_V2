# coding: utf-8

import os.path as osp

import torch
import numpy as np
import cv2

from .utils.prior_box import PriorBox
from .utils.nms_wrapper import nms
from .utils.box_utils import decode
from .utils.timer import Timer
from .utils.functions import check_keys, remove_prefix, load_model
from .utils.config import cfg
from .models.faceboxes import FaceBoxesNet

# some global configs
confidence_threshold = 0.05
top_k = 5000
keep_top_k = 750
nms_threshold = 0.3
vis_thres = 0.5
resize = 1

scale_flag = True
HEIGHT, WIDTH = 720, 1080

make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)
pretrained_path = make_abs_path('weights/FaceBoxesProd.pth')


def viz_bbox(img, dets, wfp='out.jpg'):
    # show
    for b in dets:
        if b[4] < vis_thres:
            continue
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    cv2.imwrite(wfp, img)
    print(f'Viz bbox to {wfp}')


class FaceBoxes:
    def __init__(self, timer_flag=False):
        torch.set_grad_enabled(False)

        net = FaceBoxesNet(phase='test', size=None, num_classes=2)  # initialize detector
        self.net = load_model(net, pretrained_path=pretrained_path, load_to_cpu=True)
        self.net.eval()
        # print('Finished loading model!')

        self.timer_flag = timer_flag

    def __call__(self, img_):
        img_raw = img_.copy()

        # scaling to speed up
        scale = 1
        if scale_flag:
            h, w = img_raw.shape[:2]
            if h > HEIGHT:
                scale = HEIGHT / h
            if w * scale > WIDTH:
                scale *= WIDTH / (w * scale)
            # print(scale)
            if scale == 1:
                img_raw_scale = img_raw
            else:
                h_s = int(scale * h)
                w_s = int(scale * w)
                # print(h_s, w_s)
                img_raw_scale = cv2.resize(img_raw, dsize=(w_s, h_s))
                # print(img_raw_scale.shape)

            img = np.float32(img_raw_scale)
        else:
            img = np.float32(img_raw)

        # forward
        _t = {'forward_pass': Timer(), 'misc': Timer()}
        im_height, im_width, _ = img.shape
        scale_bbox = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)

        _t['forward_pass'].tic()
        loc, conf = self.net(img)  # forward pass
        _t['forward_pass'].toc()
        _t['misc'].tic()
        priorbox = PriorBox(image_size=(im_height, im_width))
        priors = priorbox.forward()
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        if scale_flag:
            boxes = boxes * scale_bbox / scale / resize
        else:
            boxes = boxes * scale_bbox / resize

        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        # keep = py_cpu_nms(dets, args.nms_threshold)
        keep = nms(dets, nms_threshold)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        _t['misc'].toc()

        if self.timer_flag:
            print('Detection: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(1, 1, _t[
                'forward_pass'].average_time, _t['misc'].average_time))

        # filter using vis_thres
        det_bboxes = []
        for b in dets:
            if b[4] > vis_thres:
                xmin, ymin, xmax, ymax, score = b[0], b[1], b[2], b[3], b[4]
                w = xmax - xmin + 1
                h = ymax - ymin + 1
                bbox = [xmin, ymin, xmax, ymax, score]
                det_bboxes.append(bbox)

        return det_bboxes


def main():
    face_boxes = FaceBoxes(timer_flag=True)

    fn = 'trump_hillary.jpg'
    img_fp = f'../examples/inputs/{fn}'
    img = cv2.imread(img_fp)
    dets = face_boxes(img)  # xmin, ymin, w, h
    # print(dets)

    wfn = fn.replace('.jpg', '_det.jpg')
    wfp = osp.join('../examples/results', wfn)
    viz_bbox(img, dets, wfp)


if __name__ == '__main__':
    main()
