import glob
import cv2
import numpy as np
from kalman_tracker import Kalman_tracker
from tensorflow import keras
from calib import *

if __name__ == "__main__":
    ###################### for proper jupyter imports:
    #!pip install import-ipynb #install this before first run
    # import import_ipynb
    #note for myself: run with Python3.10 (not 64bit)
    #use the pythonian way:
    # from calibration import calibrate
    # from depth_map import compute_depth
    # from tracking import track
    # from neural_net import initialize_network, classify
    # or run with the following:
    #################################################

    create_output_video = False

    ################# Initializations #################
    cnn_model = keras.models.load_model('C:/Users/henri/OneDrive/Desktop/DTU courses/31392 Perception/final_project/FULL_PROJECT/cnn_model_3');
    tracker = Kalman_tracker(occlusion=True, Verbose=False, also_predict=True, model=cnn_model, eligibility_trace=0.75)


    ################### Main #######################
    #path_left = "conveyor_full_without/left/*.png"
    #path_right = "conveyor_full_without/right/*.png"
    # path_left = "conveyor_full_with/left/*.png"
    # path_right = "conveyor_full_with/right/*.png"
    path_left = "C:/Users/henri/OneDrive/Desktop/DTU courses/31392 Perception/final_project/tracking/video/full/with_occlusions/left/*.png"
    path_right = "C:/Users/henri/OneDrive/Desktop/DTU courses/31392 Perception/final_project/tracking/video/full/with_occlusions/right/*.png"

    images_left = glob.glob(path_left)
    assert images_left, "No images found in {}".format(path_left)
    images_right = glob.glob(path_right)
    assert images_right, "No images found in {}".format(path_right)

    # Create video
    if create_output_video:
        img_buffer = []
        FPS = 30 
        out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), FPS, size)

    # MAIN
    no_of_frames = len(images_left)

    # GRAYSCALE
    gray_left = cv2.cvtColor(
        cv2.imread("C:/Users/henri/OneDrive/Desktop/DTU courses/31392 Perception/final_project/tracking/video/full/with_occlusions/left/1585434750_371934652_Left.png"), cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(
        cv2.imread("C:/Users/henri/OneDrive/Desktop/DTU courses/31392 Perception/final_project/tracking/video/full/with_occlusions/right/1585434750_371934652_Right.png"), cv2.COLOR_BGR2GRAY)
    depth_map = get_depth_map(gray_left, gray_right)


    for i in range(no_of_frames):
        frame_left = cv2.imread(images_left[i])
        frame_right = cv2.imread(images_right[i])

        calibrated_left = calibrate(frame_left, frame_right)
        track_frame, filtered, measured, prediction = tracker.next_frame(calibrated_left)
        centroidx, centroidy = filtered[0], filtered[1]
        z       = compute_depth(centroidx, centroidy, depth_map)

        if z==0: # no depth
            cv2.putText(track_frame, "Depth: {}".format("not available"),
                        (700, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        else: # Depth
            cv2.putText(track_frame, "Depth: {}".format(str(z)),
                        (700, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        if create_output_video:
            img_buffer.append(frame_left)
        
        #Add more text to frame

        cv2.imshow("Perception project", track_frame) #roi

        if cv2.waitKey(1) & 0xFF == ord('q'): #stop on q
                break

        print("\rFrame {}/{}".format(i+1, no_of_frames+1), end="")

    # create_video(buffer)
    cv2.destroyAllWindows()
    print("\nSuccessfully completed")
    if create_output_video:
        for i in range(len(img_buffer)):
            out.write(img_buffer[i])
        out.release()
        print("video created")





