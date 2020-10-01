# coding: utf-8

__author__ = 'cleardusk'

import timeit
import numpy as np

SETUP_CODE = '''
import os
os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import onnxruntime

onnx_fp = "weights/mb1_120x120.onnx" # if not existed, convert it, see "convert_to_onnx function in utils/onnx.py"
session = onnxruntime.InferenceSession(onnx_fp, None)

img = np.random.randn(1, 3, 120, 120).astype(np.float32)
'''

TEST_CODE = '''
session.run(None, {"input": img})
'''


def main():
    repeat, number = 5, 100
    res = timeit.repeat(setup=SETUP_CODE,
                        stmt=TEST_CODE,
                        repeat=repeat,
                        number=number)
    res = np.array(res, dtype=np.float32)
    res /= number
    mean, var = np.mean(res), np.std(res)
    print('Inference speed: {:.2f}±{:.2f} ms'.format(mean * 1000, var * 1000))


if __name__ == '__main__':
    main()
