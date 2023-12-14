import cv2
import numpy as np
from KalmanFilter import KalmanFilter
from Detector import detect

def read_video_feed_and_get_contours(filename):
    # Create a VideoCapture object
    cap = cv2.VideoCapture(filename)
    filter = KalmanFilter()
    # Read each frame from the video
    while True:
        # Capture frame-by-frame
        _, frame = cap.read()

        centers = detect(frame)
        # Draw the contours and their corresponding areas
        for center in centers:
            center_x, center_y = tuple(center.astype(int).ravel())
            state_matrix = filter.predict(center_x, center_y)
            predicted_x, predicted_y = tuple(state_matrix.astype(int).ravel())
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
            cv2.rectangle(frame, (predicted_x, predicted_y), 5, (0, 0, 255), -1)

        # Display the resulting frame
        cv2.imshow('Video Feed', frame)

        # Check if the Esc key is pressed
        if cv2.waitKey(30) & 0xFF == 27:
            break

    # When done, release the video capture object
    cap.release()
    cv2.destroyAllWindows()