import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 图片中含有9x6的内部角点
w = 9
h = 6
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((w*h, 3), np.float32)
# 去掉z坐标，转换为二维矩阵
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('left/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    # 彩色图像转灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # return whether the input image is a view of the chessboard pattern
    # and locate the internal chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        # draw corners detected either as red circles if the board was not found
        # or as colored corners connected with lines if the board was found
        cv2.drawChessboardCorners(img, (w, h), corners, ret)
        cv2.imshow('findCorners', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# 标定的结果参数
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print('camera matrix:')
print(mtx)
print('distortion:')
print(dist)

# undistort
# img2 = cv2.imread('left\left12.jpg')
# ih,  iw = img2.shape[:2]
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(iw,ih),1,(iw,ih))
# dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)

# crop the image 根据前面ROI区域裁剪图片
# x,y,w,h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.imwrite('caliresult.jpg',dst)