from time import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from KalmanFilter import KalmanFilter
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from scipy.spatial import distance


class NNMOT:
    """
    Class to perform object tracking with the Kalman filter
    """

    def __init__(self, filename, image_shape=(224, 224), eps=1) -> None:
        self.object_id = 0
        self.tracks = {}
        self.detections = np.loadtxt(filename, delimiter=",", usecols=range(10))
        self.kalman_filter = KalmanFilter()
        self.model_url = (
            "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"
        )
        self.layer = hub.KerasLayer(self.model_url)
        self.model = tf.keras.Sequential([self.layer])
        self.image_shape = image_shape
        self.eps = eps

    def __extract(self, image: Image.Image, bbox):
        left = bbox[0]
        top = bbox[1]
        right = bbox[0] + bbox[2]
        bottom = bbox[1] + bbox[3]
        cropped_img = image.crop((left, top, right, bottom)).resize(self.image_shape)
        file = np.stack((cropped_img,) * 3, axis=-1)
        file = np.array(cropped_img) / 255.0
        embedding = self.model.predict(file[np.newaxis, ...])
        feature_np = np.array(embedding)
        flattended_feature = feature_np.flatten()
        return flattended_feature

    def __compute_scores(self, emb_one, emb_two):
        return tf.keras.losses.cosine_similarity(emb_one, emb_two)

    def __calculate_iou(self, box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        inter_y2 = min(box1[1] + box1[3], box2[1] + box2[3])
        inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)
        union_area = box1[2] * box1[3] + box2[2] * box2[3] - inter_area
        return inter_area / union_area if union_area else 0

    def __get_iou(self, frame: Image.Image, box1):
        embed_im1 = self.__extract(frame, box1)

        def filter_iou(detection):
            embed_im2 = self.__extract(frame, detection[2:6])
            iou = self.__calculate_iou(box1, detection[2:6])
            similarity_score = self.__compute_scores(embed_im1, embed_im2)
            return -self.eps * iou - similarity_score

        return filter_iou

    def __get_centroids(self, bbox) -> tuple[int, int]:
        centroid_x = int(bbox[0] + bbox[2] / 2)
        centroid_y = int(bbox[1] + bbox[3] / 2)
        return centroid_x, centroid_y

    def track_detection(self, frame: Image.Image, frame_number: int) -> dict:
        """
        Update tracks using iou threshold and hungarian algorithm + kalman filter
        """

        detections = self.detections[self.detections[:, 0] == frame_number]
        cost_matrix = np.zeros((len(self.tracks.values()), detections.shape[0]))
        matched_detections = set()
        for idx, track in enumerate(self.tracks.values()):
            track["predicted_state"] = track["kalman_filter"].predict()
            cost_matrix[idx] = np.apply_along_axis(
                self.__get_iou(frame, track["bbox"]), 1, detections
            )

        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        track_ids = list(self.tracks.keys())
        for t_idx, d_idx in zip(row_idx, col_idx):
            track = self.tracks[track_ids[t_idx]]
            track["bbox"] = detections[d_idx][2:6]
            track["last_updated"] = frame_number
            detection_centroids = self.__get_centroids(detections[d_idx][2:6])
            track["kalman_filter"].update(*detection_centroids)

        # delete unmatch tracks
        self.tracks = dict(
            filter(
                lambda track: track[1]["last_updated"] == frame_number,
                self.tracks.items(),
            )
        )

        # create new tracks
        for idx, detection in enumerate(detections):
            if not idx in matched_detections:
                detection_centroids = self.__get_centroids(detections[idx][2:6])
                self.tracks[self.object_id] = {
                    "bbox": detection[2:6],
                    "kalman_filter": KalmanFilter(
                        u_x=detection_centroids[0], u_y=detection_centroids[1]
                    ),
                    "last_updated": frame_number,
                }
                self.object_id += 1

        return self.tracks
