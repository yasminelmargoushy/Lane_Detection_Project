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