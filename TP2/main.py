import os
import cv2
from track_detection import load_detection_file, track_detection


detections = load_detection_file("../ADL-Rundle-6/det/det.txt")
print(detections)
tracks = []

IMAGES_PATH = "../ADL-Rundle-6/img1/"
TOTAL_FRAMES = 500
SIGMA_IOU = 0.5


def draw_tracking_results(frame_obj, track_objs: list):
    """
    Draw tracking results
    """
    for track in track_objs:
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
            f"ID: {track['track_id']}",
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

    current_frame_detections = detections[detections[:, 0] == frame_number]
    tracks = track_detection(
        current_frame_detections, tracks, frame_number, sigma_iou=SIGMA_IOU
    )

    frame = draw_tracking_results(frame, tracks)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
