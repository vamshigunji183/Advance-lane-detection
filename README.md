# Advance-lane-detection

<p align="center">
 <a href="https://www.youtube.com/watch?v=ymq9e9_GJ0E"><img src="./img/sample.gif" alt="Overview" width="50%" height="50%"></a>
 <br>Qualitative results. (click for full video)
</p>



** Advance Lane Detection**

The following steps were performed for lane detection:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position

# Dependencies
* Python 3.5
* Numpy
* OpenCV-Python
* Matplotlib
* Pickle

## How to run
Run `python main.py`. explicitly can be specified to `image out ` or `video output` in `main.py`.

# Detailed Pipeline of the project 

## Camera Calibration
The camera was calibrated using the chessboard images in 'camera_cal/*.jpg'. 
The following steps were performed for each camera calibration :
  * Convert to grayscale
  * Find chessboard corners with OpenCV's `findChessboardCorners()` function, assuming a 9x6 board
  * OpenCV's `calibrateCamera()` function to calculate the distortion matrices.
  * undistort images using OpenCV's `undistort()` function.
  
  Here is the output of undistorted using camera calibration values through 'calibrate_camera.p'
  ![]()
  
  ## Thresholded binary image `binarization_utils.py`
  Goal : `threshold binary frame`
  input:`undistorted frame`
  discussion : we are tyring to identify the pixels most likely to be part of lane lines.
  Apply the following filters to achieve "binary images" 
   * applying sobel filter : `BGR2GRAY` -> `cv2.Sobel` -> `np.absolute` -> `binary output`
   * magnitude threshold : `BGR2GRAY` -> `cv2.Sobel` in x and y directions -> `gradient_magnitude` square root of square of sobel transformation in each direction->`binary output`
   * direction threshold : `BGE2GRAY` -> `cv2.Sobel` in x and y directions  with threshold values `(0,np.pi/2)`-> `abs_grad_direction` is the arctan of absolute sobel value in x and y ->`binary_output`
   * HSV threshold : `BGR2HSV` -> `binary output`
   * binarize : logical_or comparision between each threshold filters
   
  ## Birdeye perspective `perspective_utils.py`
   Goal: apply perspective transformation of the input frame
   input: undistorted image and camera calibration
    * use RoI points src = np.float32([[w, h-10],    # br
                                      [0, h-10],    # bl
                                      [546, 460],   # tl
                                      [732, 460]])  # tr
                      dst = np.float32([[w, h],       # br
                                        [0, h],       # bl
                                        [0, 0],       # tl
                                        [w, 0]])      # tr
                                        
     * `cv2.getPerspectiveTransform` for M and Minv matrices by interchanging the src and dst values
     * warp the image `cv2.warpPrespective`
     
   ## Draw lines on the lane lines `line_utils.py`
   * Line with `@update_line`,`@draw`,`@average_fit`,`@curvature` and `@curvature_meter`
   * get polynomial coeffiecits for lane-lines detected in a binary frame
   * draw both the drivable lane area and the detected lane-lines onto the originals frame.
   
   ## `main.py` here is all the part of the pipeline are stitched
   for images -> `mode='images`
   for video -> `mode=video`
   
   
