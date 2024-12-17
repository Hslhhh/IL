import numpy as np
# from sympy import symbols,factor,expand,cancel,apart
import IPython
e=IPython.embed
import h5py
import multiprocessing
from multi_camera import RealEnv
from pymycobot import MyArmC, MyArmM

#两个函数分别用于通过四个路点获取一个三次方程和混合两个三次轨迹，每个轨迹重新计算时刻，从而每个轨迹对应的时间段为2t
def get_param_mat(t):
    mat = np.array([[0,0,0,1],[1,1,1,1],[8,4,2,1],[27,9,3,1]])
    mat_inv=np.linalg.inv(mat)
    # print(mat_inv)
    time_mat=np.diagflat([8000,400,20,1])
    param_mat=mat_inv@time_mat
    return param_mat

def fusing(param1, param2):
    #parameter t align with the second param instead of the first
    param=np.zeros([1,5])
    param[0]=-20*param1[0]+20*param2[0]
    param[1]=-5*param1[0]-20*param1[1]+20*param2[1]
    param[2]=-0.3*param1[0]-3*param1[1]-20*param1[2]+20*param2[2]
    param[3]=0.01*param1[0]-1*param1[2]-20*param1[3]+20*param2[3]
    param[4]=0.001*param1[0]+0.01*param1[1]+0.1*param1[2]+1*param1[3]
    return param

#after getting the curve with variable t, we can set the vector field with little effort
def get_curve(pnt):
    n_pnt = len(pnt)
    param_mat=get_param_mat()
    param=[]
    for i in range(int(n_pnt/2)-1):
        param1=param_mat@pnt[2*i:2*i+4]
        param.append(param1)
    
    fusing_param=[]
    for p in range(len(param)-1):
        P_fusing=fusing(param[p],param[p+1])
        fusing_param.append(P_fusing)

    return param,fusing_param

def linear_regression(angles,q_curr,t):
    angles=np.array(angles).T #7*100
    q_param=[]
    q_fusing_param=[]
    for i in range(angles.shape[0]):
        param,fusing_param=get_curve(angles[i,:])
        q_param.append(param) #7*50*4
        q_fusing_param.append(fusing_param) #7*49*5
    if t == 0:
        q_diff=angles[:,0]-q_curr
        q_tan=q_param[0,2]
    elif t<=1:
        param_i=q_param[:,0,:]
        q_diff=param_i[:,0]*t_diff^3+param_i[:,1]*t_diff^2+param_i[:,2]*t_diff+param_i[:,3]-q_curr
        q_tan=3*param_i[:,0]*t_diff^2+2*param_i[:,1]*t_diff+param_i[:,2]
    else: 
        if int(t)%2==1:
            t_cut=int(int(t)/2)
            t_diff=t-int(t)+1/20
            param_i=q_param[:,t_cut,:]
            q_diff=param_i[:,0]*t_diff^3+param_i[:,1]*t_diff^2+param_i[:,2]*t_diff+param_i[:,3]-q_curr
            q_tan=3*param_i[:,0]*t_diff^2+2*param_i[:,1]*t_diff+param_i[:,2]
        else:
            t_cut=int(int(t)/2)-1
            t_diff=t-int(t)
            param_i=q_fusing_param[:,t_cut,:]
            q_diff=param_i[:,0]*t_diff^4+param_i[:,1]*t_diff^3+param_i[:,2]*t_diff^2+param_i[:,3]*t_diff+param_i[:,4]-q_curr
            q_tan=4*param_i[:,0]*t_diff^3+3*param_i[:,1]*t_diff^2+2*param_i[:,2]*t_diff+param_i[:,3]

    # vector-field here is set for convenience,with no actual calculation
    delta_q=(q_diff+q_tan)*1/20
    return q_curr+delta_q

def calculate_vector_field(pos,target,diff):
    k=0.01#constant 
    D=target-pos
    D_abs=np.sqrt(D@(D.T))
    G=np.sqrt(k)/np.sqrt(k+D_abs**2)
    H=np.sqrt(1-G**2)
    return H*D+G*diff

def load_d(filename):
    with h5py.File(filename, 'r') as hf:
        data = hf['qpos'][:]
    return data

# def obstacle_field():
if __name__ == '__main__':
    qpos=load_d('/home/a/pymycobot-main/data/episode0.h5')
    q0=qpos[:,0]
    # env=RealEnv

    
