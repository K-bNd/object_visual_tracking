import os
import time
import cv2
from kalman_tracker import MOTwithKalmanFilter


IMAGES_PATH = "../ADL-Rundle-6/img1"
TOTAL_FRAMES = 525
SIGMA_IOU = 0.4

object_tracker = MOTwithKalmanFilter(
    "../ADL-Rundle-6/det/det.txt",
    sigma_iou=SIGMA_IOU,
)

prev_frame_time = 0
new_frame_time = 0
all_tracking_data = {}


def save_tracking_results(tracking_data, sequence_name):
    """Save tracking results"""
    with open(f"{sequence_name}.txt", "w", encoding="utf-8") as f_out:
        for current_frame_number, frame_tracks in tracking_data.items():
            for track_id, track in frame_tracks.items():
                bbox = track["bbox"]
                line = f"{current_frame_number},{track_id},{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])},1,-1,-1,-1\n"
                f_out.write(line)


def draw_tracking_results(frame_obj, track_objs: dict, fps_meas=30):
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
    cv2.putText(
        frame,
        f"FPS: {fps_meas}",
        (20, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )
    return frame_obj


for frame_number in range(1, TOTAL_FRAMES + 1):
    frame_path = os.path.join(IMAGES_PATH, f"{frame_number:06d}.jpg")
    frame = cv2.imread(frame_path)
    if frame is None:
        break
    last_frame_time = time.time()
    tracks = object_tracker.track_detection(frame_number)
    all_tracking_data[frame_number] = tracks
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    fps = int(fps)
    fps = str(fps)
    frame = draw_tracking_results(frame, tracks, fps_meas=str(fps))
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
save_tracking_results(all_tracking_data, "ADL-Rundle-6")
