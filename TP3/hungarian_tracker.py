import numpy as np
from scipy.optimize import linear_sum_assignment


class HungarianObjectTracker:
    """
    Class to perform object tracking
    """

    def __init__(self, filename, sigma_iou=0.5) -> None:
        self.object_id = 0
        self.tracks = {}
        self.sigma_iou = sigma_iou
        self.detections = np.loadtxt(filename, delimiter=",", usecols=range(10))

    def __calculate_iou(self, box1, box2):
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

    def __get_iou(self, box1):
        """
        Returns a function that calcultes the iou for a given bounding box
        """

        def filter_iou(detection):
            iou = self.__calculate_iou(box1, detection[2:6])
            return -iou if iou > self.sigma_iou else 0

        return filter_iou

    def track_detection(self, frame_number: int) -> dict:
        """
        Update tracks using iou threshold and hungarian algorithm
        """
        detections = self.detections[self.detections[:, 0] == frame_number]
        cost_matrix = np.zeros((len(self.tracks.values()), detections.shape[0]))
        matched_detections = set()
        for idx, track in enumerate(self.tracks.values()):
            cost_matrix[idx] = np.apply_along_axis(
                self.__get_iou(track["bbox"]), 1, detections
            )

        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        track_ids = list(self.tracks.keys())
        for t_idx, d_idx in zip(row_idx, col_idx):
            track = self.tracks[track_ids[t_idx]]
            track["bbox"] = detections[d_idx][2:6]
            track["last_updated"] = frame_number

        # delete unmatch tracks
        self.tracks = dict(
            filter(
                lambda track: track[1]["last_updated"] == frame_number,
                self.tracks.items(),
            )
        )

        # create new tracks
        for idx, detection in enumerate(detections):
            if not (idx in matched_detections):
                self.tracks[self.object_id] = {
                    "bbox": detection[2:6],
                    "last_updated": frame_number,
                }
                self.object_id += 1

        return self.tracks
