import numpy as np


def estimate_homography(first, second):
    M = []

    count = int(first.size / 2)  # 一定为偶数

    for j in range(0, count):
        # 转换成齐次坐标
        homogeneous_first = np.array([
            first[j][0],
            first[j][1],
            1
        ])

        homogeneous_second = np.array([
            second[j][0],
            second[j][1],
            1
        ])

        pr_1 = homogeneous_first
        pr_2 = homogeneous_second

        M.append(np.array([
            -pr_1.item(0), -pr_1.item(1), -1,
            0, 0, 0,
            pr_1.item(0) * pr_2.item(0), pr_1.item(1) * pr_2.item(0), pr_2.item(0)
        ]))

        M.append(np.array([
            0, 0, 0,
            -pr_1.item(0), -pr_1.item(1), -1,
            pr_1.item(0) * pr_2.item(1), pr_1.item(1) * pr_2.item(1), pr_2.item(1)
        ]))

    U, S, Vh = np.linalg.svd(np.array(M).reshape((count*2, 9)))   # 这里first.size为108=(9x6)x2

    L = Vh[-1]   # 获得最小奇异值对应的向量

    H = L.reshape(3, 3)

    return H


def compute_homography(data):
    real = data['real']
    sensed = data['sensed']
    # print(real)
    # print(sensed)

    homos = []
    # 遍历每张图像
    for i in range(0, len(data['sensed'])):
        estimated = estimate_homography(real[0], sensed[i])
        #print(estimated)
        homos.append(estimated)

    return np.array(homos)
