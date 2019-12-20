#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import fire
from timeit import time
import warnings
import cv2
import numpy as np
from config.default import _C as cfg

from Detector import DetectorFactory
from tools.utils import crop_tracks
from tools.file_parse import parse_hierachy

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

warnings.filterwarnings('ignore')
CV_CAP_PROP_FPS = 5
CV_CAP_PROP_FRAME_WIDTH = 3
CV_CAP_PROP_FRAME_HEIGHT = 4
CV_CAP_PROP_FRAME_COUNT=7


def main(
        root,
        new_root,
        config_file
):

    """
    Parameters
    ----------
    root : str
        The root for the whole file hierachy. The root will be parsed by glob in "parse_hierachy".
    new_root : str
        In which the new hierachy will be constructed.
    config_file : str
        Configurations in ./config.
    """

    cfg.merge_from_file(config_file)
    cfg.freeze()

    file_nodes = parse_hierachy(root, '**/*.mp4')

    # META definitions
    light_weighted = True

    detector_factory = DetectorFactory()
    ratio_thres_min = cfg.RATIO_THRESHOLD_MIN
    ratio_thres_max = cfg.RATIO_THRESHOLD_MAX
    height_thres = cfg.HEIGHT_THRESHOLD
    width_thres = cfg.WIDTH_THRESHOLD
    output_dir = new_root
    nms_max_overlap = cfg.NMS_MAX_OVERLAP
    distance_metric = cfg.NND.METRIC  # cosine
    max_cosine_distance = cfg.NND.MAX_COS_DISTANCE  # 0.3
    nn_buget = cfg.NND.BUGET  # None
    # writeVideo_flag = cfg.OUTPUT_DIR is not False
    max_iou_distance = cfg.TRACK.MAX_IOU_DISTANCE  # 0.7

    # sort
    detector = detector_factory.make_detector(cfg, cfg.DETECTOR.NAME)
    metric = nn_matching.NearestNeighborDistanceMetric(distance_metric,
                                                       max_cosine_distance,
                                                       nn_buget)

    for file_node in file_nodes:
        video_capture = cv2.VideoCapture(file_node.path)
        vfps = float(video_capture.get(CV_CAP_PROP_FPS))
        num_frame = video_capture.get(CV_CAP_PROP_FRAME_COUNT)
        # w = int(video_capture.get(CV_CAP_PROP_FRAME_WIDTH))
        # h = int(video_capture.get(CV_CAP_PROP_FRAME_HEIGHT))
        max_age = int(cfg.TRACK.MAX_AGE_TIME * vfps)  # 30
        n_init = int(cfg.TRACK.N_INIT_TIME * vfps)  # 3

        # A kalman filter tracker
        # Predict the next position based on kalman filter.
        # Update the detections to calibrate the predictions.
        tracker = Tracker(metric, max_iou_distance, max_age, n_init)

        # A dictionary to collect tracks
        # Dict[track_id -> List[(frame_count, image)]]
        track_collection = {}

        frame_count = 0
        eof_flag = False

        while not eof_flag:
            frames = []
            batch_count = 0
            while batch_count < cfg.BATCH:
                skip_count = 0
                while skip_count < cfg.SKIP:
                    ret, frame = video_capture.read()  # frame shape 640*480*3
                    skip_count +=1
                ret, frame = video_capture.read()  # frame shape 640*480*3
                if not ret:
                    eof_flag = True
                    break
                else:
                    batch_count += 1
                    frames.append(frame)
            if eof_flag:
                break  # the last batch cannot satisfy the shape constrain

            t1 = time.time()


            images = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                      for frame in frames]
            images = np.array(images)
            img_bboxes, img_scores = detector(images)

            for i in range(cfg.BATCH):
                bboxes, scores, image = img_bboxes[i], img_scores[i], images[i]
                detections = [Detection(bbox, score, [])
                              for bbox, score in zip(bboxes, scores)]

                detections = preprocessing.standardize_detections(
                    detections, ratio_thres_min, ratio_thres_max, height_thres, width_thres)

                # Run non-maxima suppression.
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(
                    boxes, nms_max_overlap, scores)
                detections = [detections[i] for i in indices]

                # Call the tracker
                tracker.predict()
                tracker.update(detections, light_weighted)

                active_tracks = [track for track in tracker.tracks if track.time_since_update==0]

                crops = crop_tracks(image, active_tracks)
                crops = {track_id: (frame_count, img)
                         for track_id, img in crops.items()}

                for track_id, img in crops.items():
                    track_collection.setdefault(track_id, []).append(img)

                frame_count += 1

            fps = (cfg.BATCH / (time.time() - t1))
            print("Processing fps={:3f}. Video time {:.3f} passed.".format(
                fps, frame_count / vfps))

        # Close files
        video_capture.release()

        # Write to output file. Each track a directory.
        dir = file_node.echo(output_dir, make_dir=True)
        for track_id, packs in track_collection.items():
            track_dir = os.path.join(dir, str(track_id))
            os.mkdir(track_dir)
            for pack in packs:
                f_count, img = pack
                if img.size != 0:
                    cv2.imwrite(os.path.join(track_dir,
                                             file_node.name + '_' + str(f_count) + '_' + str(track_id) + '.png'),
                                cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    fire.Fire(main)
