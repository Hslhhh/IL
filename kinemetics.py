import numpy as np
import math
import time

class six_dof_arm:
    def __init__(self):
        # 定义 DH参数
        # 关节 	 theta 	     d 	     a_i-1 	  alpha_i-1  offset
        # 1 	theta_1 	173.9 	0 	     0 	        0
        # 2 	theta_2 	0 	    0 	     -PI/2 	    -PI/2
        # 3 	theta_3 	0 	    308.315  0 	        0
        # 4 	theta_4 	327.9 	0 	     -PI/2 	    0
        # 5 	theta_5 	0 	    0 	     PI/2 	    0
        # 6 	theta_6 	207 	0 	     -PI/2 	    PI
        self.dh_params = np.array([
            [0, 0.1739, 0       , 0],
            [-np.pi/2, 0     , 0       , -0.5*math.pi],
            [0, 0     , 0.308315, 0],
            [0, 0.3279, 0       , -0.5*math.pi],
            [0, 0     , 0       , 0.5*math.pi],
            [np.pi, 0.207 , 0       , -0.5*math.pi]
        ])
        self.offset=np.array([0,-math.pi/2,0,0,0,math.pi])
    def reparam(self,theta:np.ndarray):
        self.dh_params = np.array([
            [theta[0], 0.1739, 0       , 0],
            [theta[1]-np.pi/2, 0     , 0       , -0.5*math.pi],
            [theta[2], 0     , 0.308315, 0],
            [theta[3], 0.3279, 0       , -0.5*math.pi],
            [theta[4], 0     , 0       , 0.5*math.pi],
            [theta[5]+np.pi, 0.207 , 0       , -0.5*math.pi]
        ])

    def dh_matrix(self,dh):
        a=dh[2]
        alpha=dh[3]
        d=dh[1]
        theta=dh[0]
        theta = np.radians(theta)
        # alpha = np.radians(alpha)
        return np.array([
            [np.cos(theta), -np.sin(theta), 0,a],
            [np.sin(theta)*np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -np.sin(alpha)*d],
            [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha), np.cos(alpha), np.cos(alpha)*d],
            [0, 0, 0, 1]
        ])

    def forward_kinematic(self,theta):
        self.reparam(theta)
        T = np.eye(4)
        for i in range(6):
            T = np.dot(T,self.dh_matrix(self.dh_params[i,:]))
        return T
        
    def DHTrans(self,alpha, a, d, theta):
        T = np.array([
            [np.cos(theta), -np.sin(theta), 0, a],
            [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -np.sin(alpha) * d],
            [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
            [0, 0, 0, 1]
        ])
        return T

    def differential(self,theta):
        # 初始化参数
        th = np.zeros(6)
        d = np.zeros(6)
        a = np.zeros(6)
        alp = np.zeros(6)
        
        th[0], d[0], a[0], alp[0] = theta[0], 0.1739,0,0
        th[1], d[1], a[1], alp[1] = theta[1], 0     ,0,-0.5*np.pi
        th[2], d[2], a[2], alp[2] = theta[2], 0     ,0.308315,0
        th[3], d[3], a[3], alp[3] = theta[3], 0.3279,0,-0.5*np.pi
        th[4], d[4], a[4], alp[4] = theta[4], 0     ,0,0.5*np.pi
        th[5], d[5], a[5], alp[5] = theta[5], 0.207 ,0,-0.5*np.pi
        
        # 计算各个DH变换矩阵
        T01 = self.DHTrans(alp[0], a[0], d[0], th[0])
        T12 = self.DHTrans(alp[1], a[1], d[1], th[1] - np.pi / 2)
        T23 = self.DHTrans(alp[2], a[2], d[2], th[2])
        T34 = self.DHTrans(alp[3], a[3], d[3], th[3])
        T45 = self.DHTrans(alp[4], a[4], d[4], th[4])
        T56 = self.DHTrans(alp[5], a[5], d[5], th[5]+np.pi)
        
        # 计算从基坐标系到第6个关节的变换矩阵
        T06 = np.dot(np.dot(np.dot(np.dot(np.dot(T01, T12), T23), T34), T45), T56)
        
        # 各个子矩阵的变换
        T46 = np.dot(T45, T56)
        T36 = np.dot(np.dot(T34, T45), T56)
        T26 = np.dot(np.dot(T23, T34), np.dot(T45, T56))
        T16 = np.dot(np.dot(T12, T23), np.dot(T34, np.dot(T45, T56)))
        
        # 计算雅可比矩阵的每一列
        def calculate_jacobian(T):
            return np.array([
                -T[0, 0] * T[1, 3] + T[1, 0] * T[0, 3],
                -T[0, 1] * T[1, 3] + T[1, 1] * T[0, 3],
                -T[0, 2] * T[1, 3] + T[1, 2] * T[0, 3],
                T[2, 0], T[2, 1], T[2, 2]
            ])
        
        j11 = calculate_jacobian(T16)
        j22 = calculate_jacobian(T26)
        j33 = calculate_jacobian(T36)
        j44 = calculate_jacobian(T46)
        j55 = calculate_jacobian(T56)
        
        # 第六列是固定的
        j66 = np.array([0, 0, 0, 0, 0, 1])
        
        # 构建雅可比矩阵
        T_mat = np.array([
            [T06[0, 0], T06[0, 1], T06[0, 2], 0, 0, 0],
            [T06[1, 0], T06[1, 1], T06[1, 2], 0, 0, 0],
            [T06[2, 0], T06[2, 1], T06[2, 2], 0, 0, 0],
            [0, 0, 0, T06[0, 0], T06[0, 1], T06[0, 2]],
            [0, 0, 0, T06[1, 0], T06[1, 1], T06[1, 2]],
            [0, 0, 0, T06[2, 0], T06[2, 1], T06[2, 2]]
        ])
        
        jacobian1 = np.dot(T_mat, np.column_stack([j11, j22, j33, j44, j55, j66]))
        
        return jacobian1




# if __name__ == '__main__':
#     arm=six_dof_arm()
#     theta=np.array([0,-90,0,0,0,180])
#     T=arm.forward_kinematic(theta)
#     # e()
#     print(T)