import numpy as np


def load_detection_file(path):
    """
    Load text containining detections

    :param path : Path to file
    :type path : str
    """
    return np.loadtxt(path, delimiter=",", usecols=range(10))


def calculate_iou(box1, box2):
    """
    Calculate IOU (Intersection over Union)
    """
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    inter_y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)
    union_area = box1[2] * box1[3] + box2[2] * box2[3] - inter_area
    return inter_area / union_area if union_area else 0


def get_iou(box1, sigma_iou=0.5):
    """
    Returns a function that calcultes the iou for a given bounding box
    """

    def filter_iou(detection):
        iou = calculate_iou(box1, detection[2:6])
        return iou if iou > sigma_iou else 0

    return filter_iou


def track_detection(
    detections: np.ndarray, tracks: list, frame_number: int, sigma_iou=0.5
) -> list:
    """
    Update tracks using iou threshold
    """
    matched_detections = set()
    for track in tracks:
        ious = np.apply_along_axis(
            get_iou(track["bbox"], sigma_iou=sigma_iou), 1, detections
        )
        if ious.max() == 0:
            track["last_updated"] = -1
            continue
        track["bbox"] = detections[ious.argmax()][2:6]
        track["last_updated"] = frame_number
        matched_detections.add(ious.argmax())

    tracks = list(filter(lambda track: track["last_updated"] == frame_number, tracks))

    for idx, detection in enumerate(detections):
        if idx not in matched_detections:
            tracks.append(
                {
                    "bbox": detection[2:6],
                    "track_id": len(tracks),
                    "last_updated": frame_number,
                }
            )

    return tracks
