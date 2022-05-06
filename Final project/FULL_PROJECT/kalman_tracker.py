"""
The Tracker, extended with a online kalman filter.

This is the interface:
    - roi, centroidx, centroidy, tracked_object   = track(calibrated_left, i)
"""

import numpy as np
import cv2
from tracker import EuclideanDistTracker

class Kalman_tracker:

    def __init__(self, occlusion=True) -> None:

        self.occlusion = occlusion
        self.object_detector = cv2.createBackgroundSubtractorKNN(history=100, dist2Threshold=1200)
        self.tracker = EuclideanDistTracker()

        ### Initialize Kalman filter ###
        # The initial state (4x1).
        self.states = np.array([[0],  # Pos along the x-axis
                        [0]])  # Pos along the y-axis 

        # The initial uncertainty (2x2) - We start with some very large values.
        # Then we will fully trust measurements first iteration.
        self.P = np.array([[1000, 0],
                    [0, 1000]])

        # The external motion (2x1).
        self.u = np.array([[-4.420334045148261], # Measured average xvel
                    [1.2415468250308124]]) # Measured average yvel

        # The transition matrix (2x2). 
        self.F = np.array([[1, 0],
                    [0, 1]])

        # The observation matrix (2x4).
        self.H = np.array([[1, 0],
                    [0, 1]])

        # The measurement uncertainty.
        # How little do you want to trust the measurements?
        self.R = 1*np.array([[50, 0],
                        [0, 50]])

        self.bad_R = 100*self.R.copy()

        # Disturbance matrix
        # How little do you want to trust the model?
        self.Q = 1*np.array([[1, 0],
                        [0, 1]])
    @staticmethod
    def update(x, P, Z, H, R):
        ### Insert update function
        y = Z - H@x
        s = H@P@H.T + R
        K = P@H.T@np.linalg.pinv(s)
        new_x = x + K@y
        new_P = (np.identity((K@H).shape[0]) - K@H)@P
        return new_x, new_P
    
    @staticmethod
    def predict(x, P, F, u, Q):
        ### insert predict function
        new_x = F@x + u
        new_P = F@P@F.T + Q # Q=0. No internal disturbtion
        return new_x, new_P

    def next_frame(self, frame):
        """ Tracks one frame, crops the objective """
        # 1
        # Object detection
        mask = self.object_detector.apply(frame) # Add new frame
        _, mask = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(mask, None, iterations=6)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 2
        # Only keep biggest objects detected/"most moving"
        biggest_area=0
        best_cnt = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area>biggest_area:
                biggest_area=area
                best_cnt = cnt
        # 3 
        # Constrain/limit search using heuristics
        if (biggest_area>4500 and biggest_area<76000): # reasonable size
            x, y, w, h = cv2.boundingRect(best_cnt)
            box_id = self.tracker.update([[x, y, w, h]])[0] # Track object
            
            if (x>200 and x<1200)and(y>200 and y<550): # On the conveyor
                x, y, w, h, id = box_id
                # Increase box bound to better match objects.
                rx=x-25
                ry=y-25
                rw=w+50
                rh=h+50
                centerx=(rx+rx+rw)//2
                centery=(ry+ry+rh)//2
                cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 3)
                cv2.circle(frame,(centerx,centery),5,(0,255,0), -1)
                
                # Cropping for classification
                crop_img = frame[y:y+h, x:x+w].copy()
                ncrop_img=cv2.resize(crop_img, (128, 128), interpolation = cv2.INTER_AREA) # Ready for classification

                # Kalman update from measurements
                z = np.array([[centerx],  # xpos
                              [centery]]) # ypos

                # Occlusion limits
                if self.occlusion and (centerx>550 and centerx<1100): # Close to occlusion
                    self.states, self.P = self.update(self.states, self.P, z, self.H, self.bad_R)
                else: # Far from occlusion
                    self.states, self.P = self.update(self.states, self.P, z, self.H, self.R)
            
            else: # Off the conveyor
                # Reset model uncertainty
                self.P = np.array([[1000, 0],
                                   [0, 1000]])

        else: # Unreasonable object
            pass

        # Kalman prediction 
        self.states, self.P = self.predict(self.states, self.P, self.F, self.u, self.Q)
        cv2.circle(frame, (np.int32(self.states[0, 0]), np.int32(self.states[1, 0])), 8, color=(255, 0, 0), thickness=-1)


        return frame, self.states.reshape((2,)), self.z.reshape((2,)), ncrop_img