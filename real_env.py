from pymycobot import MyArmM
from pymycobot import MyArmC
import cv2
import os
from PIL import Image
import numpy as np
import time
import collections
import matplotlib.pyplot as plt
plt.ion
import dm_env
import h5py
import keyboard


class RealEnv:
    def __init__(self, arm_m: MyArmM, arm_c: MyArmC, camera_index):
        self.arm_m = arm_m
        self.arm_c = arm_c
        anglesC = arm_c.get_joints_angle()
        anglesC[1]=anglesC[1]*-1
        arm_m.set_joints_angle(anglesC, 3)
        #arm_c.set_joints_angle([0,0,0,0,0,0,0], 3)
        self.camera = cv2.VideoCapture(camera_index)
        

    def get_qpos(self):
        return self.arm_m.get_joints_angle()
    
    # def get_qvel(self):
    #     return self.arm_m.get_servos_speed()
    
    def get_images(self):
        ret, frame = self.camera.read()
        if ret is False:
            print("Failed to capture frame.")
            return None
        return frame
    
    def _reset_joints(self):
        anglesC = self.arm_c.get_joints_angle()
        anglesC[1]=anglesC[1]*-1
        self.arm_m.set_joints_angle(anglesC, 3)
        
    def get_observations(self):
        obs=collections.OrderedDict()
        obs['qpos']=self.get_qpos()
        # obs['qvel']=self.get_qvel()
        obs['image']=self.get_images()
        return obs
    
    def get_reward(self):
        return 0
    
    def reset(self):
        self._reset_joints()
        return dm_env.TimeStep(
            reward=self.get_reward(),
            observation=self.get_observations(),
            discount=None,
            step_type=dm_env.StepType.FIRST,
        )
    
    def step(self,action):
        action[1]=action[1]*-1
        self.arm_m.set_joints_angle(action,100)
        return dm_env.TimeStep(
            reward=self.get_reward(),
            observation=self.get_observations(),
            discount=None,
            step_type=dm_env.StepType.MID)
    
def get_action(arm_c: MyArmC):
    angle=arm_c.get_joints_angle()
    gripper_angle = angle.pop(-1)
    angle.append((gripper_angle - 0.08) / (-95.27 - 0.08) * (-123.13 + 1.23) - 1.23)
    return angle

def test_real_teleop():
    myarmm = MyArmM('/dev/ttyACM0')
    myarmc = MyArmC('/dev/ttyACM1')
    camera_index=0
    env = RealEnv(myarmm, myarmc, camera_index)
    ts=env.reset()
    episode = [ts]
    actions=[]
    acual_dt_history = []
    # cv2.namedWindow('camera')

    print(type(ts.observation['image']))
    # if True:
    #     # ax=plt.subplot()
    #     plt_img = cv2.imshow('camera',ts.observation['image'])
    #     cv2.waitKey(1)
    #     plt.ion
    print(time.time())
    for t in range(1000):
        t0=time.time()
        action=get_action(myarmc)
        t1=time.time()
        ts=env.step(action)
        t2=time.time()
        episode.append(ts)
        actions.append(action)
        acual_dt_history.append([t0,t1,t2])
        if keyboard.is_pressed('Enter'):
            break

        # plt_img = cv2.imshow('camera',ts.observation['image'])
        # cv2.waitKey(1)
    print(time.time())

    images=[]
    qpos=[]
    for ts in episode:
        images.append(ts.observation['image'])
        qpos.append(ts.observation['qpos'])

    f=h5py.File('data/episode0.h5','w')
    f['images']=images
    f['qpos']=qpos
    f['actions']=actions
    f['acual_dt_history']=acual_dt_history
    f.close()
    


if __name__ == '__main__':
    test_real_teleop()

