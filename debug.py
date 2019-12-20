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


from Encoder import EncoderFactory
from Detector import DetectorFactory
from tools.utils import annotate

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

warnings.filterwarnings('ignore')
CV_CAP_PROP_FPS = 5
CV_CAP_PROP_FRAME_WIDTH = 3
CV_CAP_PROP_FRAME_HEIGHT = 4


def main(
        file_or_device_name,
        config_file,
        output_dir=None,
        ):

    """
    Parameters
    ----------
    file_or_device_name : str
    config_file : str
        Configurations in ./config.
    output_dir : str
        If not None, write annotated video into this directory for visualization.
        The white bboxes are the current track positions.
        The blue bboxes are the current detections.
    """

    cfg.merge_from_file(config_file)
    cfg.freeze()

    video_capture = cv2.VideoCapture(file_or_device_name)
    vfps = float(video_capture.get(CV_CAP_PROP_FPS))
    w = int(video_capture.get(CV_CAP_PROP_FRAME_WIDTH))
    h = int(video_capture.get(CV_CAP_PROP_FRAME_HEIGHT))

    print("Original video FPS: {}".format(vfps))
    print("Original video shape: {}x{}".format(h, w))

    # META definitions
    encoder_factory = EncoderFactory()
    detector_factory = DetectorFactory()
    ratio_thres_min = cfg.RATIO_THRESHOLD_MIN
    ratio_thres_max = cfg.RATIO_THRESHOLD_MAX
    height_thres = cfg.HEIGHT_THRESHOLD
    width_thres = cfg.WIDTH_THRESHOLD
    # output_dir = cfg.OUTPUT_DIR
    nms_max_overlap = cfg.NMS_MAX_OVERLAP
    distance_metric = cfg.NND.METRIC  # cosine
    max_cosine_distance = cfg.NND.MAX_COS_DISTANCE  # 0.3
    nn_buget = cfg.NND.BUGET  # None
    writeVideo_flag = output_dir is not None
    max_iou_distance = cfg.TRACK.MAX_IOU_DISTANCE  # 0.7
    max_age = int(cfg.TRACK.MAX_AGE_TIME * vfps)  # 30
    n_init = int(cfg.TRACK.N_INIT_TIME * vfps)  # 3


    # deep_sort
    encoder = encoder_factory.make_encoder(cfg, cfg.ENCODER.NAME)
    detector = detector_factory.make_detector(cfg, cfg.DETECTOR.NAME)
    metric = nn_matching.NearestNeighborDistanceMetric(distance_metric,
                                                       max_cosine_distance,
                                                       nn_buget)

    # A kalman filter tracker
    # Predict the next position based on kalman filter.
    # Update the detections to calibrate the predictions.
    tracker = Tracker(metric, max_iou_distance, max_age, n_init)

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(os.path.join(
            output_dir, 'output.avi'), fourcc, vfps, (w, h))
        # list_file = open(os.path.join(output_dir, 'detection.txt'), 'w')


    fps = 0.0
    frame_count = 0

    while True:
        skip_count = 0
        while skip_count < cfg.SKIP:
            ret, frame = video_capture.read()  # frame shape 640*480*3
            if writeVideo_flag:
                out.write(frame)
            skip_count +=1
        ret, frame = video_capture.read()
        if not ret:
            break
        t1 = time.time()

        # image = Image.fromarray(frame)
        # image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxes, scores = detector(np.array([image]))
        # print("box_num",len(boxs))

        features = encoder(np.array([image]), bboxes)[0]
        bboxes = bboxes[0]
        scores = scores[0]


        # score to 1.0 here).
        detections = [Detection(bbox, score, feature)
                      for bbox, score, feature in zip(bboxes, scores, features)]

        # detections = preprocessing.standardize_detections(
        #     detections, ratio_thres_min, ratio_thres_max, height_thres, width_thres)

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        annotate(frame, tracker.tracks, detections)

        # cv2.imshow('', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)

            # save bboxes
            # list_file.write(str(frame_count) + ' ')
            # if len(bboxes) != 0:
            #     for i in range(0, len(bboxes)):
            #         list_file.write(str(bboxes[i][0]) + ' ' + str(bboxes[i][1]) +
            #                         ' ' + str(bboxes[i][2]) + ' ' + str(bboxes[i][3]) + ' ')
            # list_file.write('\n')

        frame_count = frame_count + 1
        fps = (fps + (1. / (time.time() - t1))) / 2
        print("Processing fps={:3f}. Video time {:.3f} passed.".format(fps, frame_count/vfps*(cfg.SKIP+1)))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close files
    video_capture.release()
    if writeVideo_flag:
        out.release()
        # list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    fire.Fire(main)
