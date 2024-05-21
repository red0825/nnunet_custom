# train_script.py
import multiprocessing
from nnunetv2.run.run_training import run_training_entry

if __name__ == '__main__':
    multiprocessing.freeze_support()
    run_training_entry()