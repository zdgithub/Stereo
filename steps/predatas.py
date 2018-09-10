import numpy as np
import cv2
import glob


def get_data():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # 所用图片中含有9x6的内部角点
    w = 9
    h = 6

    # 不考虑z坐标
    # prepare object points, like (0,0), (1,0), (2,0) ....,(8,5)
    objp = np.zeros((w * h, 2), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    # 此处要格外注意路径，单独运行该py和在main中调用该py的文件相对路径不一样
    images = glob.glob('left/*.jpg')

    # 每张图像的世界坐标编号都一样
    objpoints.append(objp)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # return whether the input image is a view of the chessboard pattern
        # and locate the internal chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (w, h), None)

        if ret == True:

            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(np.array(corners).reshape(w*h, 2))    # 注意坐标转成二维的


    # print(np.array(objpoints).shape)
    # print(imgpoints)


    return {
        'real': objpoints,
        'sensed': imgpoints
    }



