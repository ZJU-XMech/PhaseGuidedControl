import os
import pickle

import numpy as np


param_file = os.path.join('model', '2020-10-14-14-18-45', 'Model_Iteration_10000_params.pkl')
with open(param_file, 'rb') as f:
    params = pickle.load(f)
wx0 = params['model/lstm_pi0/wx:0']
wh0 = params['model/lstm_pi0/wh:0']
b0 = params['model/lstm_pi0/b:0'].reshape([-1, 1])
wx1 = params['model/lstm_pi1/wx:0']
wh1 = params['model/lstm_pi1/wh:0']
b1 = params['model/lstm_pi1/b:0'].reshape([-1, 1])
pi_w = params['model/pi/w:0']
pi_b = params['model/pi/b:0'].reshape([-1, 1])

np.savetxt(os.path.join('..', 'bp_software_v3_deepermimic', 'config', 'cpg_mimic_wx0.csv'), wx0, delimiter=',')
np.savetxt(os.path.join('..', 'bp_software_v3_deepermimic', 'config', 'cpg_mimic_wh0.csv'), wh0, delimiter=',')
np.savetxt(os.path.join('..', 'bp_software_v3_deepermimic', 'config', 'cpg_mimic_b0.csv'), b0, delimiter=',')
np.savetxt(os.path.join('..', 'bp_software_v3_deepermimic', 'config', 'cpg_mimic_wx1.csv'), wx1, delimiter=',')
np.savetxt(os.path.join('..', 'bp_software_v3_deepermimic', 'config', 'cpg_mimic_wh1.csv'), wh1, delimiter=',')
np.savetxt(os.path.join('..', 'bp_software_v3_deepermimic', 'config', 'cpg_mimic_b1.csv'), b1, delimiter=',')
np.savetxt(os.path.join('..', 'bp_software_v3_deepermimic', 'config', 'cpg_mimic_fc_w.csv'), pi_w, delimiter=',')
np.savetxt(os.path.join('..', 'bp_software_v3_deepermimic', 'config', 'cpg_mimic_fc_b.csv'), pi_b, delimiter=',')