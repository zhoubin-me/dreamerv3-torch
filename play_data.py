import glob
import numpy as np
from envs.homebench_env import HomeBenchEnv
from collections import defaultdict
datas = glob.glob('../garment_demo_save_rl_60/*.npz')
import cv2
import os
from tqdm import tqdm
NUM_EPS = 60

env_jp = HomeBenchEnv(task='RLLGarment.GarmentV1', action_mode='joint_position')
env_djp = HomeBenchEnv(task='RLLGarment.GarmentV1', action_mode='delta_joint_with_gripper')
env_djp2 = HomeBenchEnv(task='RLLGarment.GarmentV1', action_mode='delta_joint_with_gripper')

for eps in tqdm(range(44, NUM_EPS+1)):
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
    
    with open(f'demo_data_original/eps_{eps:03d}-{MAX_STEP}.npz', 'wb') as f:
        np.savez(f, **eps_data_dict)
    
    keys = ['vecEEPose', 'vecGripperOpenAmount', 'vecJointPositions', 'action', 'reward', 'discount', 'is_terminal', 'is_first', 'image']
    eps_data_joint = defaultdict(list)
    obs = env_jp.reset()
    for i, action in enumerate(eps_data_dict['action']):
        next_obs, reward, done, info = env_jp.step(action)
        for k in keys:
            if k in ['vecEEPose', 'vecGripperOpenAmount', 'vecJointPositions', 'image', 'is_terminal', 'is_first']:
                eps_data_joint[k].append(obs[k])
            elif k == 'action':
                eps_data_joint[k].append(action)
            elif k == 'reward':
                eps_data_joint[k].append(reward)
            elif k == 'discount':
                eps_data_joint[k].append(info['discount'])
            else:
                raise ValueError(k)
        print('*' * 10, i, reward, done)
        obs = next_obs
        if done:
            break

    if len(eps_data_dict['action']) < 640 and not done:
        action = eps_data_dict['action'][-1]
        for i in range(640 - len(eps_data_dict['action'])):
            next_obs, reward, done, _ = env_jp.step(action)
            for k in keys:
                if k in ['vecEEPose', 'vecGripperOpenAmount', 'vecJointPositions', 'image', 'is_terminal', 'is_first']:
                    eps_data_joint[k].append(obs[k])
                elif k == 'action':
                    eps_data_joint[k].append(action)
                elif k == 'reward':
                    eps_data_joint[k].append(reward)
                elif k == 'discount':
                    eps_data_joint[k].append(info['discount'])
                else:
                    raise ValueError(k)
            print('*' * 10, reward, done)
            obs = next_obs
            if done:
                break

    for k, v in eps_data_joint.items():
        if isinstance(v, list):
            eps_data_joint[k] = np.stack(eps_data_joint[k], axis=0)
    
    steps = len(eps_data_joint['image'])
    fname = f'eps_{eps:03d}-{steps}.npz'
    with open(os.path.join(f'demo_data_joint/{fname}'), 'wb') as f:
        np.savez(f, **eps_data_joint)    
    


    obs = env_djp.reset()
    delta_actions = []
    for i, action in enumerate(eps_data_joint['action']):
        delta_action = action - obs['vecJointPositions']
        open_amount = action[-2] * 20
        delta_action = delta_action[:-1]
        delta_action[-1] = open_amount
        delta_actions.append(delta_action)
        next_obs, reward, done, _ = env_djp.step(delta_action)
        obs = next_obs
        print('=' * 10, i, reward, done)
        if done:
            break

    if len(eps_data_joint['action']) < 640 and not done:
        delta_action = np.zeros(8)
        delta_action[-1] = eps_data_dict['action'][-1][-2] * 20
        for i in range(640 - len(eps_data_dict['action'])):
            next_obs, reward, done, _ = env_djp.step(delta_action)
            delta_actions.append(delta_action)
            print('=' * 10, reward, done)
            obs = next_obs
            if done:
                break
    
    eps_data_delta = defaultdict(list)
    obs = env_djp2.reset()
    for i, delta_action in enumerate(delta_actions):
        next_obs, reward, done, _ = env_djp2.step(delta_action)
        for k in keys:
            if k in ['vecEEPose', 'vecGripperOpenAmount', 'vecJointPositions', 'image', 'is_terminal', 'is_first']:
                eps_data_delta[k].append(obs[k])
            elif k == 'action':
                eps_data_delta[k].append(delta_action)
            elif k == 'reward':
                eps_data_delta[k].append(reward)
            elif k == 'discount':
                eps_data_delta[k].append(info['discount'])
            else:
                raise ValueError(k)
        print('-' * 10, i, reward, done)
        obs = next_obs
        if done:
            break
    
    steps = len(eps_data_delta['image'])
    fname = f'eps_{eps:03d}-{steps}.npz'
    with open(os.path.join(f'demo_data_deltajoint/{fname}'), 'wb') as f:
        np.savez(f, **eps_data_delta)   
