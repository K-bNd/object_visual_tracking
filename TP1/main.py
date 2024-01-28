import cv2
from KalmanFilter import KalmanFilter
from Detector import detect


def read_video_feed_and_get_contours(filename):
    """
    Draws on given video to find object contours
    """

    cap = cv2.VideoCapture(filename)
    kalman_filter = KalmanFilter()
    # Read each frame from the video
    while True:
        _, frame = cap.read()

        centers = detect(frame)
        # Draw the contours and their corresponding areas
        for center in centers:
            center_x, center_y = tuple(center.astype(int).ravel())
            state_matrix = kalman_filter.predict()
            kalman_filter.update(center_x, center_y)
            predicted_x, predicted_y = tuple(state_matrix.astype(int).ravel())
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
            cv2.rectangle(frame, (predicted_x, predicted_y), 5, (0, 0, 255), -1)

        cv2.imshow("Video Feed", frame)

        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    read_video_feed_and_get_contours("randomball.avi")
