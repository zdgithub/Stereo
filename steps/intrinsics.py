import numpy as np


# closed-form solution
# homography H is a 3x3 matrix
def v(i, j, H):
    return np.array([
        H[0, i] * H[0, j],
        H[0, i] * H[1, j] + H[1, i] * H[0, j],
        H[1, i] * H[1, j],
        H[2, i] * H[0, j] + H[0, i] * H[2, j],
        H[2, i] * H[1, j] + H[1, i] * H[2, j],
        H[2, i] * H[2, j]
    ])


def get_camera_intrinsics(homographies):
    h_count = len(homographies)

    vec = []

    for i in range(0, h_count):
        curr = np.reshape(homographies[i], (3, 3))

        vec.append(v(0, 1, curr))
        vec.append(v(0, 0, curr) - v(1, 1, curr))

    # vec is a 2nx6 matrix
    vec = np.array(vec)
    # print(vec.ndim)

    # return b the least-squares solution to a linear matrix equation
    # b = np.linalg.lstsq(
    #     vec,
    #     np.zeros((h_count*2, 1)),
    #     rcond=None
    # )[0]

    U, S, Vh = np.linalg.svd(np.dot(np.transpose(vec), vec))
    L = Vh[-1]  # 获得最小奇异值对应的向量
    b = L.reshape(6, 1)

    # print('b = ')
    # print(b)

    B = np.zeros([4, 4], dtype=np.float32)
    B[1, 1] = b[0]
    B[1, 2] = b[1]
    B[2, 2] = b[2]
    B[1, 3] = b[3]
    B[2, 3] = b[4]
    B[3, 3] = b[5]

    v0 = (B[1, 2] * B[1, 3] - B[1, 1] * B[2, 3]) / (B[1, 1] * B[2, 2] - B[1, 2] * B[1, 2])
    lamb = B[3, 3] - (B[1, 3] * B[1, 3] + v0 * (B[1, 2] * B[1, 3] - B[1, 1] * B[2, 3])) / B[1, 1]
    alpha = np.sqrt(lamb / B[1, 1])
    beta = np.sqrt(lamb * B[1, 1] / (B[1, 1] * B[2, 2] - B[1, 2] * B[1, 2]))
    gamma = - B[1, 2] * alpha * alpha * beta / lamb
    u0 = gamma * v0 / beta - B[1, 3] * alpha * alpha / lamb

    # return camera intrinsics
    return np.array([
        [alpha, gamma, u0],
        [0, beta, v0],
        [0, 0, 1]
    ])
