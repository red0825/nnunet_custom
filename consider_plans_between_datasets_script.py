import sys
import multiprocessing
from nnunetv2.experiment_planning.plans_for_pretraining.consider_plans_between_datasets import entry_point_consider_plans_between_datasets

if __name__ == '__main__':
    multiprocessing.freeze_support()    
    entry_point_consider_plans_between_datasets()