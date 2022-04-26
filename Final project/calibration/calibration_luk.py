import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
import time

########################################################################################
# FUNCTINOS
########################################################################################


def undistort_img(side, save=False):
    # Implement the number of vertical and horizontal corners
    nb_vertical = 9
    nb_horizontal = 6

    if side == 'left':
        path_in = 'Exercises/Final Project/calibration/left-*.png'
        path_out = 'Exercises/Final Project/undistorted/left-'
    elif side == 'right':
        path_in = 'Exercises/Final Project/calibration/right-*.png'
        path_out = 'Exercises/Final Project/undistorted/right-'

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nb_horizontal*nb_vertical, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nb_vertical, 0:nb_horizontal].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints_left = []  # 2d points in image plane.

    images = glob.glob(path_in)
    assert images

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Implement findChessboardCorners here
        ret, corners = cv2.findChessboardCorners(
            gray, (nb_vertical, nb_horizontal))

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            imgpoints_left.append(corners)

    # get the camera matrix
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints_left, gray.shape[::-1], None, None)
    # just to get dimensions
    img_left = cv2.imread('Exercises/Final Project/calibration/left-0000.png')
    h,  w = img.shape[:2]
    K, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0)

    # save images into folder
    if save:
        i = 0
        for fname in images:
            # undistort
            img = cv2.imread(fname)
            dst = cv2.undistort(img, mtx, dist, None, K)

            # crop the image
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]

            # save image
            cv2.imwrite(path_out+str(i)+'.png', dst)
            i += 1

    return K, dist


def draw_lines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    (r, c) = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 2)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

########################################################################################
# UNDISTORTING IMAGES
########################################################################################
# in 2D its not necessary, we can detect the object even on the curve, but in cas we are in 3D, we need to mvoe on a line - therefore undistortion is crucial
# mostly based on project from the Week 4
rerun_undistortion = False
rerun_rectification = True
rerun_depth = True
debug = False

# values from previous runs
K_left = np.array([[590.24505615, 0, 723.85543853],[0,700.56091309,369.43859036],[0,0,1]])
K_right = np.array([[698.72259521, 0 , 648.50704794],[0,698.6318967,374.0875587],[0,0,1]])
dist_right = np.array([[-3.29479763e-01,1.41779367e-01,-1.15867147e-04,2.53566722e-04,-3.10092346e-02]])
dist_left = np.array([[-3.25580109e-01,1.39151479e-01,-2.55229666e-04,4.20203965e-04,-3.19659112e-02]])

if rerun_undistortion:

    # undistort images
    K_left, dist_left = undistort_img(side='left', save=False)
    print(K_left,dist_left)
    K_right, dist_right = undistort_img(side='right', save=False)
    print(K_right,dist_right)

########################################################################################
# RECTIFICATION
########################################################################################


if rerun_rectification:

    # Read the undistorted images
    imagesL = glob.glob('Exercises/Final Project/undistorted/left-*.png')
    imagesR = glob.glob('Exercises/Final Project/undistorted/right-*.png')
    assert imagesL
    assert imagesR

    # Create a sift detector
    sift = cv2.SIFT_create()

    i = 0
    for i in range(0, len(imagesL)):
        img_left = cv2.imread(imagesL[i])
        img_right = cv2.imread(imagesR[i])
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # Find the keypoints and descriptors with SIFT
        kp_left, des_left = sift.detectAndCompute(gray_left, None)
        kp_right, des_right = sift.detectAndCompute(gray_right, None)
        kp_gray_left = cv2.drawKeypoints(
            gray_left, kp_left, gray_left, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        kp_gray_right = cv2.drawKeypoints(
            gray_right, kp_right, gray_right, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Match points
        matches = cv2.BFMatcher().match(des_left, des_right)
        matches = sorted(matches, key=lambda x: x.distance)
        nb_matches = 200  # Using 200 best matches
        good = []
        pts1 = []
        pts2 = []
        for m in matches[:nb_matches]:
            good.append(m)
            pts1.append(kp_left[m.queryIdx].pt)
            pts2.append(kp_right[m.trainIdx].pt)
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)

        # Get fundamental matrix
        F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_RANSAC)

        # Remove outliers
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]

        # Draw lines
        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
        lines1 = lines1.reshape(-1, 3)
        epilines_left, keypoints_left = draw_lines(
            gray_left, gray_right, lines1, pts1, pts2)
        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
        lines2 = lines2.reshape(-1, 3)
        epilines_right, keypoints_right = draw_lines(
            gray_right, gray_left, lines2, pts2, pts1)
        
        if debug:
            fig, axs = plt.subplots(
                2, 2, constrained_layout=True, figsize=(10, 10))
            axs[0, 0].imshow(keypoints_right)
            axs[0, 0].set_title('left keypoints')
            axs[0, 1].imshow(keypoints_left)
            axs[0, 1].set_title('right keypoints')
            axs[1, 0].imshow(epilines_left)
            axs[1, 0].set_title('left epipolar lines')
            axs[1, 1].imshow(epilines_right)
            axs[1, 1].set_title('right epipolar lines')
            plt.show()

        # Find projection matrix
        E = K_left.T@F@K_right
        R_left, R_right, t = cv2.decomposeEssentialMat(E)
        cv2.stereoRectify(K_left, dist_left, K_right,
                          dist_right, img_left.shape[:2], R_left, t)
        P_left = np.hstack((K_left@R_left, K_left@t))
        P_right = np.hstack((K_right@R_right, K_right@t))

        # Rectify images
        (w,h) = gray_left.shape
        leftMapX, leftMapY = cv2.initUndistortRectifyMap(
            K_left, dist_left, R_left, P_left, (w, h), cv2.CV_32FC1)
        left_rectified = cv2.remap(
            gray_left, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        rightMapX, rightMapY = cv2.initUndistortRectifyMap(
            K_right, dist_right, R_right, P_right, (w, h), cv2.CV_32FC1)
        right_rectified = cv2.remap(
            gray_right, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

        # Save images into folder
        if True:
            cv2.imwrite('Exercises/Final Project/rectified/left-'+str(i)+'.png', left_rectified)
            cv2.imwrite('Exercises/Final Project/rectified/right-'+str(i)+'.png', right_rectified)
            i += 1


########################################################################################
# IMAGE DEPTH
########################################################################################

if rerun_depth:

    # Read the rectified images
    imagesL = glob.glob('Exercises/Final Project/rectified/left*.png')
    imagesR = glob.glob('Exercises/Final Project/rectified/right*.png')
    assert imagesL
    assert imagesR

    for i in range(0, len(imagesL)):
        img_left = cv2.imread(imagesL[i])
        img_right = cv2.imread(imagesR[i])
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        min_disp = 5  # 7
        num_disp = 16  # 3*16
        block_size = 5  # 5, 11
        stereo = cv2.StereoBM_create(
            numDisparities=num_disp, blockSize=block_size)
        stereo.setMinDisparity(min_disp)
        stereo.setDisp12MaxDiff(200)  # 200
        stereo.setUniquenessRatio(1)  # 1
        stereo.setSpeckleRange(10)  # 3
        stereo.setSpeckleWindowSize(1)  # 3
        disp = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

        f, (ax_left, ax_middle, ax_right) = plt.subplots(1, 3, figsize=(18, 18))
        ax_left.imshow(gray_left)
        ax_middle.imshow(gray_right)
        ax_right.imshow(disp)
        plt.show()
