import numpy as np
import pandas as pd
import cv2 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
# %matplotlib inline




def camera_calibration(verbose=False):
    
#     assert path.exists(calibration_img_dir), '"{}" must exist and contain calibration images.'.format(calibration_img_dir)
    # number of inside coners in any given row
    nx =9 
    # number of inside conrners in any given column
    ny = 6
    #object points array
    objpoints =[]
    # image points array
    imgpoints = []

    #  generate object points
    objp = np.zeros((ny*nx,3),np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
    
    images = glob.glob('camera_cal/calibration*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        img_size = (img.shape[1],img.shape[0])
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        ret,corners = cv2.findChessboardCorners(gray,(9,6),None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners) 
                       
            if verbose:
                cv2.drawChessboardCorners(img,(nx,ny),corners,ret)
                plt.figure(figsize=(10,10))
                fig = plt.figure()
                plt.imshow(img)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,img_size,None,None)
#############################################
    cal_pickle = {}
    cal_pickle["mtx"] = mtx
    cal_pickle["dist"] = dist
    pickle.dump( cal_pickle, open( 'camera_cal/cameracalibration.p', "wb" ) )
#############################################
    return ret,mtx,dist,rvecs,tvecs
       
def undistort(img,mtx,dist,verbose=False):
    dst = cv2.undistort(img,mtx,dist,None,mtx)
    
    if verbose:
        plt.figure(figsize=(10,10))
        fig = plt.figure()
        plt.imshow(dst,cmap='gray')
        
    return dst
                       

if __name__ == "__main__":
    
    ret,mtx,dist,rvecs,tvecs = camera_calibration(verbose=False)
    
    img = cv2.imread('test_images/test3.jpg')
    undistorted_img = undistort(img,mtx,dist)
    plt.imsave('test_images/calibration_tested_2.jpg',undistorted_img)
