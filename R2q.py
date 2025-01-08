from scipy.spatial.transform import Rotation as R
import numpy as np

# 旋转矩阵
rotation_matrix = np.array([[1,0,0],[0,1,0],[0,0,1]])

def rotation2quaternion(R_matrix):
    rot = R.from_matrix(R_matrix)
    q = rot.as_quat()
    m=np.zeros(q.shape)
    m[0]=q[3]
    m[1:]=q[0:3]
    return m

q=rotation2quaternion(rotation_matrix)
def q_to_Lq(q):
    s=q[0]
    v=q[1:]
    L1=[s,v[0],v[1],v[2]]
    v_hat=[[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]]
    v_hat=s*np.eye(3)+v_hat
    L2=[v[0],v_hat[0,0],v_hat[0,1],v_hat[0,2]]
    L3=[v[1],v_hat[1,0],v_hat[1,1],v_hat[1,2]]
    L4=[v[2],v_hat[2,0],v_hat[2,1],v_hat[2,2]]
    
    # Lq=[[L1],[L2],[L3],[L4]]
    Lq=np.array([L1,L2,L3,L4])  # 转为 numpy array
    return Lq

def get_w(R1,R2):
    q1=rotation2quaternion(R1)#当前姿态
    q2=rotation2quaternion(R2)#目标姿态
    q1_inverse=[q1[0],-q1[1],-q1[2],-q1[3]]
    q1_dot=compute_q_dot(q1,q2)
    w_hat=q_to_Lq(q1_inverse)@(2*q1_dot)
    w=w_hat[1:]
    return w

def compute_q_dot(q1,q2):
    q_dot=(q2-q1)*1/2#hyperparameter 1/2
    return q_dot

# Lq=q_to_Lq(q)
# print(Lq@[q[0],-q[1],-q[2],-q[3]])
if __name__=='__main__':
    R1=np.array([[1,0,0],[0,1,0],[0,0,1]])
    R2=np.array([[0,0,1],[0,1,0],[-1,0,0]])
    w=get_w(R1,R2)
    print(w)