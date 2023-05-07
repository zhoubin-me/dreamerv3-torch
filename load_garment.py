import glob
import numpy as np
import pickle
from tqdm import tqdm
import cv2

NUM_EPS = 60
NUM_STEPS = 600

for eps in tqdm(range(1, NUM_EPS+1)):
    fs = glob.glob(f'../garment_demo_save_rl_60/demo_actor_0_eps_{eps}_step*.npz')
    steps = [int(x.split('_')[-1][:-4]) for x in fs]
    MAX_STEP = max(steps)
    transitions = []
    keys = None
    for i_step in range(1, MAX_STEP+1):
        step_data = np.load(f'../garment_demo_save_rl_60/demo_actor_0_eps_{eps}_step_{i_step}.npz', allow_pickle=True)
        obs = [v for _, v in step_data['observation'].item().items()]
        if keys is None:
            keys = [k for k in step_data['observation'].item().keys()]
            keys += ['action', 'reward', 'discount', 'is_terminal', 'is_first']
        discount = step_data['discount']
        step_type = step_data['step_type']
        is_first = step_type == 0
        is_terminal = step_type == 2
        transition = (*obs, step_data['action'], step_data['reward'], step_data['discount'], is_terminal, is_first)
        transitions.append(transition)
    eps_data = list(zip(*transitions))
    eps_data_dict = {}
    for k, v in zip(keys, eps_data):
        if len(v[0].shape) == 3:
            v = [cv2.resize(x.transpose(1, 2, 0), (64, 64)) for x in v]
        v = np.stack(v, axis=0).squeeze()
        if k == 'vecGripperOpenAmount':
            v = np.expand_dims(v, axis=-1)
        eps_data_dict[k] = v

    for k, v in eps_data_dict.items():
        print(k, v.shape)

    eps_data_dict['image'] = eps_data_dict['rgbWrist']
    # action = eps_data_dict['action']
    # action = action[:, :-1]
    # action[:, -1] = 1.0 / (1.0 + np.exp(-action[:, -1]))
    # eps_data_dict['action'] = eps_data_dict['action'][:, :-1]
    del eps_data_dict['rgbWrist']
    with open(f'demo_data_eps/eps_{eps:03d}-{MAX_STEP}.npz', 'wb') as f:
        np.savez(f, **eps_data_dict)
