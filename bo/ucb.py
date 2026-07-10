# import os
# import sys
# import shutil
# import numpy as np
# import subprocess
# import h5py
# from PIL import Image
# import time
# import random

# import torch
# from botorch.sampling.samplers import SobolQMCNormalSampler
# from botorch.models import SingleTaskGP
# from botorch.fit import fit_gpytorch_model
# from gpytorch.mlls import ExactMarginalLogLikelihood
# from botorch.optim import optimize_acqf
# from botorch.acquisition import UpperConfidenceBound
# from botorch.models.transforms.outcome import Standardize
# from botorch.acquisition.monte_carlo import qUpperConfidenceBound
# from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
# from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
# from botorch.utils.transforms import unnormalize

# from bo import bo_random_config


def initialize(config):
    pass
    
#     n_initial_jobs = config['bo']['parallel_jobs']
#     n_var_params = len([p for p in config['bo']['params'].values() if p is not None])
    
#     train_x = np.zeros([n_initial_jobs, n_var_params])

#     for i in range(n_initial_jobs):
#         config_i = bo_random_config(config)
#         train_x[i] = [p for p in config_i['bo']['params'].values() if p is not None]


def ask(config, bo_state):

    next_config = config.copy()

    return next_config


def tell(job_config, bo_state, y_value):
    pass