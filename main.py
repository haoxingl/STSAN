from __future__ import absolute_import, division, print_function, unicode_literals

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
gpu_id = "0, 1, 2, 3, 4, 5, 6, 7"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

from train_stream_t import main as main_t
from train_ST_SAN import main as main_stsan

""" train Stream-T first and then ST-SAN """

if __name__ == "__main__":
    main_t()
    main_stsan()
