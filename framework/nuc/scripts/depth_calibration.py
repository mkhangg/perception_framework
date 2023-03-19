# Ref: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

import numpy as np
import cv2 as cv
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
print("device = ", device)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

grid_size = (7, 7)
cell_size = 25 #mm
z_w = 500 # mm world
N_CALIB_LOOP = 20
scale = 265

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((grid_size[0]*grid_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1,2)
print("objp = ", objp.shape)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
count = 0
k = 0
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        objp[k, 0] = cell_size*i
        objp[k, 1] = cell_size*j
        objp[k, 2] = 0
        k += 1

#Skip some first frames
while True:
    frameset = pipeline.wait_for_frames()
    count += 1
    if count > 30:
        break


count = 0    
#print(objp)
#exit(0)
while True:
    #img = cv.imread("test_ros/scripts/chessboard.jpg")
    frameset = pipeline.wait_for_frames()
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()
    img = np.asanyarray(color_frame.get_data())
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)    
    #print("gray = ", gray.shape,  gray.shape[::-1])
    
   
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, grid_size, flags=cv.CALIB_USE_INTRINSIC_GUESS)
    # If found, add object points, image points (after refining them)
    if ret == True:
        
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        cv.drawChessboardCorners(img, grid_size, corners2, ret)
        #print("corners2 = ", corners2.shape)
        print("count     = ", count)
        print("corners2 0 = ", corners2[0])
        print("corners2 1 = ", corners2[1])

        #print(type(corners2[0][0][0]), corners2[0][0][0])
        x0, y0 = corners2[0][0][0], corners2[0][0][1]
        x1, y1 = corners2[1][0][0], corners2[1][0][1]
        print(x0, y0, x1, y1)
        
        #if abs(x1 - x0) >= 25 and abs(y1 - y0) <= 5:
        if abs(x1 - x0) < 5 and abs(y1 - y0) >= 25:
        #if True:
            count += 1
            objpoints.append(objp)
            imgpoints.append(corners2)                        

    #view = np.hstack((img, gray))
    
    cv.imshow('img', img)    
    key = cv.waitKey(1000)
    if key == 27:
        break
    if count >= N_CALIB_LOOP:
            break

#cv.destroyAllWindows()

# cam_matrix = np.load("cam_matrix.npy")
# dist = np.load("dist.npy")
# ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], cameraMatrix=cam_matrix, distCoeffs=dist, flags=cv.CALIB_USE_INTRINSIC_GUESS)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# np.save("mtx.npy", mtx)
# np.save("dist.npy", dist)
# np.save("rvecs.npy", rvecs)
# np.save("tvecs.npy", tvecs)

#print("ret= ", ret)
print("mtx = ", mtx) 
print("dist= ", dist) 
print("rvecs= ", rvecs[0].T)
print("tvecs= ", tvecs[0].T)
np_obj = np.array(objpoints).reshape(-1, 3)
np_img = np.array(imgpoints).reshape(-1, 2)
print("np_obj = ", np_obj.shape, "np_img = ", np_img.shape)

#Change z from 0 to z_w
for i in range(np_obj.shape[0]):
    np_obj[i][2] = z_w

print("solvePNP")
ret, rvec1, tvec1 = cv.solvePnP(np_obj, np_img, mtx, dist)
print("rvec1 = ", rvec1.T)
print("tvec1 = ", tvec1.T)
#exit()

print("R - rodrigues vecs")
R_mtx, jac = cv.Rodrigues(rvec1)
#R_mtx, jac = cv.Rodrigues(rvec1)


inv_cam = np.linalg.inv(mtx)
inv_r = np.linalg.inv(R_mtx)
tran = tvec1
#tran = tvecs


def calculate_XYZ(u, v, inv_cam, inv_r, tran):    
    uv_1 = np.array([[u,v,1]], dtype=np.float32)
    uv_1 = uv_1.T
    suv_1 = scale*uv_1
    xyz_c = inv_cam.dot(suv_1)
    xyz_c = xyz_c - tran
    XYZ = inv_r.dot(xyz_c)
    return XYZ

while True:
    frameset = pipeline.wait_for_frames()
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()
    img = np.asanyarray(color_frame.get_data())
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, grid_size, flags=cv.CALIB_USE_INTRINSIC_GUESS)
    if ret:
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        x0, y0 = corners2[0][0][0], corners2[0][0][1]
        x1, y1 = corners2[1][0][0], corners2[1][0][1]
        #print(x0, y0, x1, y1)
        
        #if abs(x1 - x0) >= 25 and abs(y1 - y0) <= 5:
        if abs(x1 - x0) < 5 and abs(y1 - y0) >= 25:
            print("===")
            for i in range(7):
                k = i*7 + i                
                center = corners2[k,:,:]
                #print("center = ", center)
                cv.drawChessboardCorners(img, (1,1), center, ret) #Draw center corner
                u, v = int(center[0][0]), int(center[0][1])
                world = calculate_XYZ(u, v, inv_cam, inv_r, tran)
                print('point_%02d (%d, %d) = (%2.2f, %2.2f, %2.2f): GT = (%03.2f, %03.2f, %03.2f)' %(k, u, v, *world, *np_obj[k]))
                    #print(">> world = ", world.T)
            
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    #view = np.hstack((img, dst))
    #cv.imshow('img-orginial', img)
    cv.imshow('undistort', dst)
    key = cv.waitKey(1000)
    if key == 27:
        break

cv.destroyAllWindows()