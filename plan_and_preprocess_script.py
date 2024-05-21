# plan_and_preprocess_script.py
import multiprocessing
from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import plan_and_preprocess_entry

if __name__ == '__main__':
    multiprocessing.freeze_support()
    plan_and_preprocess_entry()