import sys
import multiprocessing
from nnunetv2.inference.predict_from_raw_data import predict_entry_point_upl_sfda

if __name__ == '__main__':
    multiprocessing.freeze_support()
    predict_entry_point_upl_sfda()