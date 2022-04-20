# Lane_Detection_Project
Lane Detection Project using Opencv & numpy
## Pipeline
    (1)   Undistort the Image(3D Image).
    (2)   Warp the Image(3D Image).
    (3)   Get L-Channel in LUV color space(2D Image).
    (4)   Get B-Channel in LAB color space(2D Image).
    (5)   Combine L-Channel & B-Channel(2D Image).
    (6)   Use Sliding Window (Blind & Quick) to find points on both left & right Lanes.
    (7)   Use the points to find a suitable equation of the 2nd order for left & right Lanes.
    (8)   Find the Lane Curvature & Offset from the center.
    (9)   Draw the Lines & Fill the Area enclosed between them to represent the lane.
    (10)  Unwarp the Image & Combine the Lane image & the Original Image.
    (11)  Print the Curvature & Car postition.
## Scope
    (1)   Camera Calibration
    (2)   Distortion Correction
    (3)   Perspective Transform
    (4)   Color and Gradient Threshold
    (5)   Detect Lane Lines
    (6)   Determine Lane Curvature
    (7)   Impose Lane Boundaries on Original Image
    (8)   Output Visual Display of Lane Boundaries and Numerical Estimation of Lane Curvature and Vehicle Position
   ###(1)   Camera Calibration
    The first step in the project is camera calibrations which is done with the provided images in the camera_cal folder. 
    Using findChessboardCorners the corners are extracted and fed into the calibrateCamera function. 
    This function then provides us with our image matrix and the distortion coefficents.
   ###(2)   Distortion Correction
    Using the image matrix, the distortion coefficents and the undistort function the images can be properly undistored.
   ###(3)   Perspective Transform
    Transform before applying thresholds on the image.
    This was done with the getPerspectiveTransform and warpPerspective funtions.
   ###(4)   Color and Gradient Threshold
    Using a combination of gradient threshold on the L-Channel in LUV color space & on the B-Channel in LAB color space, we obtained a 2D image filtered image.
   ###(5)   Detect Lane Lines
    Lane lines were found by sliding a histogram window from the bottom of the to top.
    At each slice a point was indexed for were the highest density of pixels were found.
   ###(6)   Determine Lane Curvature
    Using the points gathered from the lane detection a 2nd order polynomial was fit to the data.
    With the new polynomial fit, new points were generated to simulate the entire length of the line.
   ###(7)   Impose Lane Boundaries on Original Image
    Lanes were imposed by taking the polynomial fit points and feeding them int opencv's poly fill to fill the polygon.  
   ###(8)   Output Visual Display of Lane Boundaries and Numerical Estimation of Lane Curvature and Vehicle Position
    Distance from the center and the lane curvature were calculated from the bottom most pixels.
## libraries used
    (1)   cv2
    (2)   numpy
    (3)   matplotlib.pyplot
    (4)   pickle
    (5)   matplotlib.image
    (6)   moviepy.editor 
    (7)   VideoFileClip
    (8)   deque from collections
## Pycharm Project Run
### Installation Steps
    (1)   Install Python (pip, virtualenv and pyvenv virtual environments are automatically installed with Python).
    (2)   Add Python to Path.
    (3)   Install Pycharm.
    (4)   Create a new project with new Virtual Environment.
    (5)   Install pickle & collections using pip [Most Probably installed by default].
    (6)   Install numpy using pip or from python Packages.
    (7)   Install matplotlib using pip or from python Packages.
    (8)   Install opencv-python using pip or from python Packages.
    (9)   Install moviepy using pip or from python Packages.
    (10)  Install glob using pip or from python Packages.
### Steps to Run
    (1)   Download the project as a zip file.
    (2)   Add the main.py & camera_calibration.py files to the project.
    (3)   Add the camera_cal folder.
    (4)   Add configuration to run camera_calibration.py file.
    (5)   Run camera_calibration.py file [camera_cal_pickle.p will be created].
    (6)   In main.py, Edit the "input_path" variable to the path of your test video.
    (7)   In main.py, Edit the "video_output" variable to the path of your test video.
    (8)   Add your test image to test_images folder.
    (9)   In main.py, Edit the "image_name" variable to the name of your test image.
    (10)  Add configuration to run main.py file.
    (11)  Run main.py file.