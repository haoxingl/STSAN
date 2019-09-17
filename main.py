from __future__ import absolute_import, division, print_function, unicode_literals

from train_stream_t import main as main_t
from train_ST_SAN import main as main_stsan

""" train Stream-T first and then ST-SAN """

if __name__ == "__main__":
    for model_index in range(1, 2):
        main_t(model_index)
        main_stsan(model_index)
