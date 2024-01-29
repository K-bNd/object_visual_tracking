import os
import cv2
from kalman_tracker import MOTwithKalmanFilter


IMAGES_PATH = "../ADL-Rundle-6/img1"
TOTAL_FRAMES = 525
SIGMA_IOU = 0.4

object_tracker = MOTwithKalmanFilter(
    "../ADL-Rundle-6/det/det.txt",
    sigma_iou=SIGMA_IOU,
)


def draw_tracking_results(frame_obj, track_objs: dict):
    """
    Draw tracking results
    """
    for track_id, track in track_objs.items():
        cv2.rectangle(
            frame_obj,
            (int(track["bbox"][0]), int(track["bbox"][1])),
            (
                int(track["bbox"][0] + track["bbox"][2]),
                int(track["bbox"][1] + track["bbox"][3]),
            ),
            (255, 0, 0),
            2,
        )
        cv2.putText(
            frame,
            f"ID: {track_id + 1}",
            (int(track["bbox"][0]), int(track["bbox"][1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return frame_obj


for frame_number in range(1, TOTAL_FRAMES + 1):
    frame_path = os.path.join(IMAGES_PATH, f"{frame_number:06d}.jpg")
    frame = cv2.imread(frame_path)
    if frame is None:
        break

    tracks = object_tracker.track_detection(frame_number)
    frame = draw_tracking_results(frame, tracks)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
