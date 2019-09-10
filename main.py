from train_stream_t import main as main_t
from train_ST_SAN import main as main_stsan

""" train Stream-T first and then ST-SAN """

if __name__ == "__main__":
    main_t()
    main_stsan()
