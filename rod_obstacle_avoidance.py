from kinemetics import six_dof_arm
import numpy as np
import torch
import math
import IPython
e=IPython.embed
import cvxpy as cp
import time


def dh_matrix(dh):
    a=dh[2]
    alpha=dh[3]
    d=dh[1]
    theta=dh[0]
    theta = theta/180*3.1415926
    # alpha = np.radians(alpha)
    return np.array([
            [np.cos(theta), -np.sin(theta), 0,a],
            [np.sin(theta)*np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -np.sin(alpha)*d],
            [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha), np.cos(alpha), np.cos(alpha)*d],
            [0, 0, 0, 1]
        ])
    

def point_transform(theta,collision_rod,collision_point):
    dh_params = np.array([
            [theta[0], 0.1739, 0       , 0],
            [theta[1]-np.pi/2, 0     , 0       , -0.5*math.pi],
            [theta[2], 0     , 0.308315, 0],
            [theta[3], 0.3279, 0       , -0.5*math.pi],
            [theta[4], 0     , 0       , 0.5*math.pi],
            [theta[5]+np.pi, 0.207 , 0       , -0.5*math.pi]
        ])
    T=np.eye(4)
    for i in range(collision_rod):
        T1=dh_matrix(dh_params[i,:])
        T = T @ T1
    collision_ang=np.append(collision_point,1)
    collision_abs=T @ collision_ang
    collision_point_world=collision_abs[0:3]
    return collision_point_world


def compute_jacobian(theta,collision_rod,collision_point):
    epsilon = 1e-8
    jacobian_matrix = np.zeros((3, 6))
    for i in range(6):
        point_forward = np.array(theta, dtype=float)
        point_backward = np.array(theta, dtype=float)
        point_forward[i] += epsilon
        point_backward[i] -= epsilon
        partial_derivative = (point_transform(point_forward,collision_rod,collision_point) - point_transform(point_backward,collision_rod,collision_point)) / (2 * epsilon)
        jacobian_matrix[:, i] = partial_derivative
    return jacobian_matrix


def solve_quadratic_program(P, q, G, h):
    """
    求解二次优化问题：
        minimize (1/2) x^T P x + q^T x
        subject to Gx <= h

    参数：
        P (numpy.ndarray): 二次项系数矩阵 (n x n, 半正定矩阵)
        q (numpy.ndarray): 线性项系数向量 (n x 1)
        G (numpy.ndarray): 线性不等式约束矩阵 (m x n)
        h (numpy.ndarray): 线性不等式约束向量 (m x 1)

    返回：
        x_opt (numpy.ndarray): 优化变量的最优解 (n x 1)
        optimal_value (float): 最优目标值
    """
    # 检查输入维度
    n = P.shape[0]
    assert P.shape == (n, n), "P 必须是一个 n x n 的矩阵"
    assert q.shape == (n,), "q 必须是一个 n 维向量"

    # 定义优化变量
    x = cp.Variable(n)

    # 定义目标函数
    objective = cp.Minimize(0.5 * cp.quad_form(x, P) + q.T @ x)

    # 定义约束
    constraints = [G @ x <= h]

    # 定义优化问题
    problem = cp.Problem(objective, constraints)

    # 求解问题
    problem.solve()

    # 返回最优解和最优值
    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise ValueError("优化问题未找到最优解：" + problem.status)

    x_opt = x.value
    optimal_value = problem.value
    return x_opt, optimal_value


def  rod_avoidance(theta,collision_point,collision_rod,v_d,w_d,o_d):
    arm=six_dof_arm()

    J_total=arm.differential(theta)
    Inv_J_total=arm.inverse_kinematic(theta)

    J_p2q=compute_jacobian(theta,collision_rod,collision_point)
    J_cal=J_p2q @ Inv_J_total
    p_w=point_transform(theta,collision_rod,collision_point)
    p_o=o_d-p_w
    J = J_cal.T @ o_d #6*1

    P=np.eye(3)
    w_d = np.array(w_d)
    q=-w_d.T @  P
    J=J.T
    J1=J[0:3]
    J2=J[3:]
    G=J2
    h=-J1 @ v_d
    w_opt, _ = solve_quadratic_program(P, q, G, h)
    
    v_total=np.append(v_d,w_opt)
    q_d=Inv_J_total @ v_total

    return q_d


# Example usage
if __name__ == "__main__":

    # Define functions as Python callables
    theta=np.array([0,0,10,20,30,40])
    theta = np.radians(theta)
    # J=compute_jacobian(theta,1,[0.05,0,0])
    time1=time.time()
    q_d=rod_avoidance(theta,[0.05,0,0],3,[0,0,0],[0,0,-1],[1,0.5,0.3])
    print(time.time()-time1)

    print(q_d)