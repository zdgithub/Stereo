import numpy as np


def estimate_lens_distortion(intrinsics, extrinsics, real, sensor):

    u0 = intrinsics[0, 2]
    v0 = intrinsics[1, 2]

    D = []
    d = []

    # 遍历m张图片，每张图片有n个内部角点
    for i in range(0, len(sensor)):
        for j in range(0, int(real.size/2)):  # 遍历54=6x9个点
            # 世界齐次坐标系，z轴为0
            homog_real_coords = np.array([real[j][0], real[j][1], 0, 1])
            # 3d相机坐标系
            homog_coords = np.dot(extrinsics[i], homog_real_coords)

            coords = homog_coords / homog_coords[-1]
            [x, y, hom] = coords

            r = np.sqrt(x*x + y*y)
            # 2d像素坐标系
            P = np.dot(intrinsics, homog_coords)
            P = P / P[2]

            [u, v, trash] = P

            du = u - u0
            dv = v - v0

            D.append(
                np.array([
                    du * r**2, du * r**4
                ])
            )

            D.append(
                np.array([
                    dv * r**2, dv * r**4
                ])
            )

            up = sensor[i][j][0]
            vp = sensor[i][j][1]

            d.append(up - u)
            d.append(vp - v)

    # 返回最小二乘解对应的畸变系数
    k = np.linalg.lstsq(
        np.array(D),
        np.array(d),
        rcond=None
    )[0]


    return k