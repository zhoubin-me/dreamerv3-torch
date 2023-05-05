import glob
import numpy as np
from envs.homebench_env import HomeBenchEnv
from collections import defaultdict
datas = glob.glob('demo_data_eps2/*.npz')
env = HomeBenchEnv(task='RLLGarment.GarmentV1')
import cv2
import os

# all keys
keys = ['vecEEPose', 'vecGripperOpenAmount', 'vecJointPositions', 'action', 'reward', 'discount', 'is_terminal', 'is_first', 'image']

for data in datas:
    x = np.load(data, allow_pickle=True)
    obs = env.reset()
    actions = x['action']
    new_data = defaultdict(list)
    for i, action in enumerate(actions):
        delta_action = action[:-1] - obs['vecJointPositions'][:-1]
        open_amount = action[-2] * 2 / 0.1
        delta_action = delta_action * 10
        delta_action = np.clip(delta_action, -1, 1)
        delta_action[-1] = open_amount
        next_obs, reward, done, _ = env.step(delta_action)
        obs = next_obs
        for k in keys:
            if k in ['vecEEPose', 'vecGripperOpenAmount', 'vecJointPositions', 'image']:
                new_data[k].append(obs[k])
            elif k == 'action':
                new_data[k].append(delta_action)

    for k in ['is_terminal', 'is_first',  'reward', 'discount']:
        new_data[k] = x[k]

    for k, v in new_data.items():
        if isinstance(v, list):
            new_data[k] = np.stack(new_data[k], axis=0)

    fname = data.split('/')[-1]
    with open(os.path.join(f'demo_data_deltajoint/{fname}'), 'wb') as f:
        np.savez(f, **new_data)

    # print(delta_action)
    # import pdb; pdb.set_trace()
    # new_data = {}


    
    # for _ in range(100):
    #     delta_action = np.zeros(8)
    #     next_obs, reward, done, _ = env.step(delta_action)
    #     print(reward)
    # imgs = np.stack(imgs, axis=0)
    # out = cv2.VideoWriter('project1.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (64, 64))
    # imgs = x['image']
    # for img in imgs:
    #     out.write(img)
    # out.release()

    # out = cv2.VideoWriter('project2.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (64, 64))
    # for img in imgs_:
    #     out.write(img)
    # out.release()
    # import pdb; pdb.set_trace()