# predict_script.py
import multiprocessing
from nnunetv2.inference.predict_from_raw_data import predict_entry_point

if __name__ == '__main__':
    multiprocessing.freeze_support()
    predict_entry_point()