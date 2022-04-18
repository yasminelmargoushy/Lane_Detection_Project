import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from collections import deque


def draw_area(undist, left_fitx, lefty, right_fitx, righty):
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Create an image to draw the lines on
    warp_zero = np.zeros(img_shape[0:2]).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    # pts_left = np.array([np.transpose(np.vstack([left_fitx, lefty]))])
    pts_left = np.array([np.flipud(np.transpose(np.vstack([left_fitx, lefty])))])

    pts_right = np.array([np.transpose(np.vstack([right_fitx, righty]))])

    pts = np.hstack((pts_left, pts_right))

    # Draw lines
    cv2.polylines(color_warp, np.int_([pts]),
                  isClosed=False, color=(200, 0, 0), thickness=30)

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img_shape[1], img_shape[0]))

    # Combine the result with the original image
    return cv2.addWeighted(undist, 1, newwarp, 0.3, 0)


def warp(img):
    """
    Perspective Transformation
    :param img:
    :return: warped image
    """

    # Compute and apply perspective transform
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (1280, 720), flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped


def luv_lab_filter(img, l_thresh=(195, 255), b_thresh=(140, 200)):
    l = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)[:, :, 0]
    l_bin = np.zeros_like(l)
    l_bin[(l >= l_thresh[0]) & (l <= l_thresh[1])] = 1

    b = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[:, :, 2]
    b_bin = np.zeros_like(b)
    b_bin[(b >= b_thresh[0]) & (b <= b_thresh[1])] = 1

    combine = np.zeros_like(l)
    combine[(l_bin == 1) | (b_bin == 1)] = 1

    return l_bin, b_bin, combine


def undistort(img, mtx, dist):
    """
    Use cv2.undistort to undistort
    :param img: Assuming input img is RGB (imread by mpimg)
    :param mtx: camera calibration parameter
    :param dist: camera calibration parameter
    :return: Undistorted img
    """
    # transform to BGR to fit cv2.imread
    img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    dst_img = cv2.undistort(img_BGR, mtx, dist, None, mtx)

    return cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB)



def process_image(img):

    global mtx, dist, src, dst, debug

    undist_img = undistort(img, mtx, dist)

    warped = warp(img)

    _, _, warped_binary = luv_lab_filter(warped, l_thresh=(215, 255),
                                   b_thresh=(145, 200))
    nonzerox, nonzeroy = np.nonzero(np.transpose(warped_binary))

    return warped_binary

img_shape = (720, 1280)
img_size = [1280, 720]  # width, height

src = np.float32([[490, 482], [810, 482],
                  [1250, 720], [0, 720]])
dst = np.float32([[0, 0], [1280, 0],
                  [1250, 720], [40, 720]])
debug = True

# import Camera Calibration Parameters
dist_pickle = "./camera_cal_pickle.p"
with open(dist_pickle, mode="rb") as f:
    CalData = pickle.load(f)
mtx, dist = CalData["mtx"], CalData["dist"]

video_output = './output_videos/challenge_video_out_debug.mp4'
input_path = './test_videos/challenge_video.mp4'
image_name = 'test1'

image_r = process_image(mpimg.imread(f'./test_images/{image_name}.jpg'))
f, (ax1) = plt.subplots(1, 1, figsize=(20, 10))
ax1.imshow(image_r)
ax1.set_title('Final Image', fontsize=30)
ax1.axis("off")
plt.show()

'''

clip1 = VideoFileClip(input_path)
final_clip = clip1.fl_image(process_image)
final_clip.write_videofile(video_output, audio=False)


'''
