from raisim_gym.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import numpy as np

import time
# import os


class RaisimGymVecEnv(VecEnv):
    def __init__(self, impl, cfg):
        super(RaisimGymVecEnv, self).__init__(impl)
        self._ob = []
        self._act = []
        self._ref = []
        self._reference = np.zeros([self.num_envs, 52], dtype=np.float32)
    
    def step(self, action, visualize=False):
        if not visualize:
            self.wrapper.step(action, self._observation, self._reward, self._done, self._extraInfo)
        else:
            self.wrapper.testStep(action, self._observation, self._reward, self._done, self._extraInfo)
            self.wrapper.getExtraDynamicsInfo(self._reference)
            self._ob.append(self._observation[0, :].copy())
            self._act.append(action[0, :])
            self._ref.append(self._reference[0, :].copy())

        if len(self._extraInfoNames) is not 0:
            info = [{'extra_info': {
                self._extraInfoNames[j]: self._extraInfo[i, j] for j in range(0, len(self._extraInfoNames))
            }} for i in range(self.num_envs)]
        else:
            info = [{} for i in range(self.num_envs)]

        for i in range(self.num_envs):
            self.rewards[i].append(self._reward[i])

            if self._done[i]:
                eprew = sum(self.rewards[i])
                eplen = len(self.rewards[i])
                epinfo = {"r": eprew, "l": eplen}
                info[i]['episode'] = epinfo
                self.rewards[i].clear()

        return self._observation.copy(), self._reward.copy(), self._done.copy(), info.copy()
    
    def reset(self):
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self.wrapper.reset(self._observation)
        return self._observation.copy()
    
    def save_action(self, file_name):
        fname = file_name.split('.')[-2]
        np.save('.' + fname + '_ob.npy', np.vstack(self._ob))
        np.save('.' + fname + '_act.npy', np.vstack(self._act))
        np.save('.' + fname + '_ref.npy', np.vstack(self._ref))
        self._ob = []
        self._act = []
        self._ref = []
    
    def set_gait(self, gait_idx):
        self.wrapper.setGait(gait_idx)
    
    def get_contact(self):
        contact = np.zeros([self.num_envs, 4], dtype=np.float32)
        self.wrapper.getContactInfo(contact)
        return contact.copy()
    
    def curriculumUpdate(self):
        self.wrapper.curriculumUpdate()