from homebench import HomeBench
from homebench.action_modes.joint_position import DeltaJointPosition
import gym
import numpy as np
import functools
import cv2

TASK_LIST = [
    'RLLGarment.GarmentV1',
    'HomeBenchExample.ReachTarget'
]

class HomeBenchEnv:
    def __init__(self, task, action_repeat=1, size=(64, 64), camera=None):
        assert task in TASK_LIST, task
        hb_env = HomeBench(task, DeltaJointPosition(), episode_steps=200)
        self._env = hb_env
        self._action_repeat = action_repeat

    @functools.cached_property
    def observation_space(self):
        spec = self._env.environment_spec.observations

        obs_spec = {}
        for k, v in spec.items():
            if len(v.shape) == 3:
                k = 'image'
                v = gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)
            else:
                v = gym.spaces.Box(-np.inf, np.inf, v.shape, dtype=np.float32)
            obs_spec[k] = v
        return gym.spaces.Dict(obs_spec)


    @functools.cached_property
    def action_space(self):
        spec = self._env.environment_spec.actions
        return gym.spaces.Box(spec.low, spec.high, spec.shape, dtype=np.float32)


    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0
        for _ in range(self._action_repeat):
            time_step = self._env.step([action])[0]
            reward += time_step.reward or 0
            if time_step.last():
                break
        obs = dict(time_step.observation)
        obs["image"] = self._preprocess_obs(obs['rgbWrist'])
        del obs['rgbWrist']
        # There is no terminal state in DMC
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        done = time_step.last()
        info = {"discount": np.array(time_step.discount, np.float32)}
        return obs, reward, done, info
    
    def reset(self):
        time_step = self._env.reset()[0]
        obs = dict(time_step.observation)
        obs["image"] = self._preprocess_obs(obs['rgbWrist'])
        del obs['rgbWrist']
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        return obs


    def _preprocess_obs(self, image):
        image = image.transpose(1, 2, 0)
        return cv2.resize(image, (64, 64))


    def render(self, *args, **kwargs):
        return None