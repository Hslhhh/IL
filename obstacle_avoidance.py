import numpy as np

class obstacle:
    def __init__(self, x, y,z,):
        self.x=x


def compute_diagonal_matrix(
    Gamma,
    dim,
    is_boundary=False,
    rho=1,
    repulsion_coeff=1.0,
    tangent_eigenvalue_isometric=True,
    tangent_power=5,
    treat_obstacle_special=True,
    self_priority=1,
):
    """Compute diagonal Matrix"""
    if Gamma <= 1 and treat_obstacle_special:
        # Point inside the obstacle
        delta_eigenvalue = 1
    else:
        delta_eigenvalue = 1.0 / abs(Gamma) ** (self_priority / rho)
    eigenvalue_reference = 1 - delta_eigenvalue * repulsion_coeff

    if tangent_eigenvalue_isometric:
        eigenvalue_tangent = 1 + delta_eigenvalue
    else:
        # Decreasing velocity in order to reach zero on surface
        eigenvalue_tangent = 1 - 1.0 / abs(Gamma) ** tangent_power
    return np.diag(
        np.hstack((eigenvalue_reference, np.ones(dim - 1) * eigenvalue_tangent))
    )


def get_orthogonal_basis(vector: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Get Orthonormal basis matrxi for an dimensional input vector."""
    # warnings.warn("Basis implementation is not continuous.") (?! problem?)
    vector = vector / np.linalg.norm(vector)

    dim = vector.shape[0]
    if dim <= 1:
        return vector.reshape((dim, dim))

    basis_matrix = np.zeros((dim, dim))

    if dim == 2:
        basis_matrix[:, 0] = vector
        basis_matrix[:, 1] = np.array([-basis_matrix[1, 0], basis_matrix[0, 0]])

    elif dim == 3:
        basis_matrix[:, 0] = vector
        basis_matrix[:, 1] = np.array([-vector[1], vector[0], 0])

        norm_vec2 = np.linalg.norm(basis_matrix[:, 1])
        if norm_vec2:
            basis_matrix[:, 1] = basis_matrix[:, 1] / norm_vec2
        else:
            basis_matrix[:, 1] = [1, 0, 0]

        basis_matrix[:, 2] = np.cross(basis_matrix[:, 0], basis_matrix[:, 1])

        norm_vec = np.linalg.norm(basis_matrix[:, 2])
        if norm_vec:
            basis_matrix[:, 2] = basis_matrix[:, 2] / norm_vec

    elif dim > 3:
        # TODO: ensure smoothness for general basis for d > 3 (?!?)
        # if True:
        basis_matrix[:, 0] = vector

        ind_zeros = np.isclose(vector, 0.0)
        n_zeros = sum(ind_zeros)
        ind_nonzero = np.logical_not(ind_zeros)

        n_nonzeros = sum(ind_nonzero)
        sub_vector = vector[ind_nonzero]
        sub_matrix = np.zeros((n_nonzeros, n_nonzeros))
        sub_matrix[:, 0] = sub_vector

        for ii, jj in enumerate(np.arange(ind_zeros.shape[0])[ind_zeros]):
            basis_matrix[jj, ii + 1] = 1.0

        for ii in range(1, n_nonzeros):
            sub_matrix[:ii, ii] = sub_vector[:ii]
            sub_matrix[ii, ii] = -np.sum(sub_vector[:ii] ** 2) / sub_vector[ii]
            sub_matrix[: ii + 1, ii] = sub_matrix[: ii + 1, ii] / np.linalg.norm(
                sub_matrix[: ii + 1, ii]
            )

            basis_matrix[ind_nonzero, n_zeros + ii] = sub_matrix[:, ii]
    return basis_matrix

def compute_decomposition_matrix(obs, x_t, in_global_frame=False, dot_margin=0.02):
    """Compute decomposition matrix and orthogonal matrix to basis"""
    # 当前位置x_t,障碍物位置obs,指向障碍物方向
    normal_vector = obs-x_t
    E_orth = get_orthogonal_basis(normal_vector, normalize=True)
    return E_orth

def get_gamma(
        x_t,obs
    ):
    if np.linalg.norm(x_t-obs)>=10:
        gamma=10
        return gamma
    else:
        gamma=np.linalg.norm(x_t-obs)
        return gamma

def compute_modulation_matrix(
    x_t, obs
):
    _dim=x_t.shape[0]
    Gamma = get_gamma(x_t, in_global_frame=False)  # function for ellipsoids

    E= compute_decomposition_matrix(obs, x_t)
    D = compute_diagonal_matrix(
        Gamma,
        dim=_dim,
        is_boundary=obs.is_boundary,
        repulsion_coeff=obs.repulsion_coeff,
    )
    M=E*D*np.linalg.inv(E)
    return M


if __name__ == '__main__':
    # gamma=10 #根据里相关的变量，该数值越大，越接近于标准矩阵
    # E=compute_diagonal_matrix(Gamma=gamma,dim=3)
    # print(E)
    x=np.array([0,0,1])
    y=np.array([1,1,0])
    x_m=np.linalg.norm(x)
    gamma=get_gamma(x,y)
    M=compute_decomposition_matrix(x,y)
    print(M)

