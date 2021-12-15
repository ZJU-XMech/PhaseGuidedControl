from PPO2 import PPO2
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle

fname = '2020-04-13-21-22-55_Iteration_9950.pkl'
model_file = os.path.join('data', 'anymal', '2020-04-13-21-22-55', fname)
param_file = os.path.join('data', 'anymal', '2020-04-13-21-22-55', fname[:-4]+'_params.pkl')

def dump_params(model_file):
    model = PPO2.load(model_file)
    params = model.get_parameters()
    folder = os.path.split(model_file)[0]
    filename = os.path.split(model_file)[1][:-4] + '_params.pkl'
    with open(os.path.join(folder, filename), 'wb') as f:
        pickle.dump(params, f)
    return True
