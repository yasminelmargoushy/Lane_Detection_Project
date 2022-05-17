import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip


#################################################################################
#################################################################################
video_output =  './output_videos/VD_project_video_v1_debug.mp4' #sys.argv[1]
input_path   =  './test_videos/project_video.mp4' #sys.argv[2]
debug        =  1 #int(sys.argv[3])
image_name   = 'test1'
test_image   =  False
test_video   = True
LD_VD        = 'VD'
#################################################################################
#################################################################################


if LD_VD == 'LD':
    from Lane_Detection import passDebugToLD, lane_detect
    passDebugToLD(debug)

    if test_image:
        image_r = lane_detect(mpimg.imread(f'./test_images/{image_name}.jpg'))
        f, (ax1) = plt.subplots(1, 1, figsize=(20, 10))
        ax1.imshow(image_r)
        ax1.set_title('Final Image', fontsize=30)
        ax1.axis("off")
        plt.show()

    if test_video:
        clip = VideoFileClip(input_path)
        final_clip = clip.fl_image(lane_detect)
        final_clip.write_videofile(video_output, audio=False)

if LD_VD == 'VD':
    from Vehicle_Detection import passDebugToVD, vehicle_detect, vehicle_detect_image
    passDebugToVD(debug)

    if test_image:
        f, axes = plt.subplots(1, 4)
        image = cv2.imread(f'./test_images/{image_name}.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ref_window_image, heat_image, final_image = vehicle_detect_image(image)
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(ref_window_image)
        axes[1].set_title("Refined Windows Image")
        axes[1].axis("off")

        axes[2].imshow(heat_image, cmap='gray')
        axes[2].set_title("Heatmap Image")
        axes[2].axis("off")

        axes[3].imshow(final_image)
        axes[3].set_title("Final Image")
        axes[3].axis("off")

        plt.show()

    if test_video:
        clip = VideoFileClip(input_path) #.subclip(40,44)
        final_clip = clip.fl_image(vehicle_detect)
        final_clip.write_videofile(video_output, audio=False)
        clip.reader.close()
        clip.audio.reader.close_proc()


def Lane_Vehicle_Detection(image):
    Lane_result = lane_detect(image)
    Vehicle_labels = vehicle_detect_label(image)
    draw_img = draw_labeled_bboxes(Lane_result, Vehicle_labels)
    return draw_img

if LD_VD == 'LD-VD':
    from Lane_Detection import passDebugToLD, lane_detect
    from Vehicle_Detection import passDebugToVD, vehicle_detect_label, draw_labeled_bboxes
    debug = 0
    passDebugToLD(debug)
    passDebugToVD(debug)

    clip = VideoFileClip(input_path) #.subclip(40,44)
    final_clip = clip.fl_image(Lane_Vehicle_Detection)
    final_clip.write_videofile(video_output, audio=False)
    clip.reader.close()
    clip.audio.reader.close_proc()








