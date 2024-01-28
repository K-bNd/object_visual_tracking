import numpy as np


class ObjectTracker:
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
            return iou if iou > self.sigma_iou else 0

        return filter_iou

    def track_detection(self, frame_number: int) -> dict:
        """
        Update tracks using iou threshold
        """
        detections = self.detections[self.detections[:, 0] == frame_number]
        matched_detections = set()
        for _, track in self.tracks.items():
            ious = np.apply_along_axis(self.__get_iou(track["bbox"]), 1, detections)
            best_index = ious.argmax()
            if ious.max() == 0 or best_index in matched_detections:
                track["last_updated"] = -1
                continue
            track["bbox"] = detections[best_index][2:6]
            track["last_updated"] = frame_number
            matched_detections.add(best_index)

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
