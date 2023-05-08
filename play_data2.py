import glob
import numpy as np
import os

fs = glob.glob('demo_data_deltajoint/*.npz')

for f in fs:
    new_data = {}
    data = np.load(f, allow_pickle=False)
    for k, v in data.items():
        new_data[k] = v
    new_data['action'][:, :-1] *= 10
    fname = f.split('/')[-1]
    with open(os.path.join(f'demo_data_deltaaction_scaled/{fname}'), 'wb') as f:
        np.savez(f, **new_data)   
