import cv2
import numpy as np
from KalmanFilter import KalmanFilter
from Detector import detect


def read_video_feed_and_get_contours(filename):
    """
    Draws on given video to find object contours
    """

    cap = cv2.VideoCapture(filename)
    kalman_filter = KalmanFilter()
    path = []
    # Read each frame from the video
    while True:
        _, frame = cap.read()

        centers = detect(frame)
        # Draw the contours and their corresponding areas
        for center in centers:
            center_x, center_y = tuple(center.astype(int).ravel())
            predicted_x, predicted_y = tuple(
                np.array(kalman_filter.predict(), dtype=int).ravel()[:2]
            )
            kalman_filter.update(center_x, center_y)

            # Drawing the detected circle (green)
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

            # Drawing the predicted object position (blue)
            cv2.rectangle(
                frame,
                (predicted_x, predicted_y),
                (predicted_x + 5, predicted_y + 5),
                (255, 0, 0),
                5,
                -1,
            )

            # Drawing the estimated object position (after update())
            estimated_x, estimated_y = tuple(
                np.array(kalman_filter.state_matrix, dtype=int).ravel()[:2]
            )

            cv2.rectangle(
                frame,
                (estimated_x, estimated_y),
                (estimated_x + 5, estimated_y + 5),
                (0, 0, 255),
                5,
                -1,
            )

            # Draw the path
            path.append((predicted_x, predicted_y))
            for i in range(1, len(path)):
                cv2.line(frame, path[i - 1], path[i], (41, 101, 255), 2)


        cv2.imshow("Video Feed", frame)

        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    read_video_feed_and_get_contours(
        "/home/yoku/scia/object_visual_tracking/TP1/randomball.avi"
    )
