from yacs.config import CfgNode as CN


_C = CN()

_C.MODEL_DATA_PATH = './model_data'
# The batch size for image reading from camera or the file
_C.BATCH = 8
# Number of frames scanned per processing frame 
_C.SKIP = 1

_C.ENCODER = CN()
# Name for encoder factory
_C.ENCODER.NAME = 'test'

_C.ENCODER.DEVICE = "cpu"
# Some models support batch inputs
_C.ENCODER.BATCH = 1


_C.DETECTOR = CN()
# Name for decoder factory
_C.DETECTOR.NAME = 'yolov3'

_C.DETECTOR.DEVICE = "cpu"
# Some models support batch inputs
_C.DETECTOR.BATCH = 1


# Nearest Neighbor Distance Metric
_C.NND = CN()
# The metric for feature space
_C.NND.METRIC = "cosine"
# The threshold for re-id feature matching
_C.NND.MAX_COS_DISTANCE = 0.25
# The number of features in the feature gallery that a track keeps
_C.NND.BUGET = False


# Detections that overlap more than this values are suppressed.
# The confidence scores are compared.
_C.NMS_MAX_OVERLAP = 1.
# h / w < ratio_thres_min will be discarded.
_C.RATIO_THRESHOLD_MIN = 2.
# h / w > ratio_thres_max will be discarded.
_C.RATIO_THRESHOLD_MAX = 6.
# int(h) < height_thres will be discarded.
_C.HEIGHT_THRESHOLD = 200  # px
# int(w) < width_thres will be discarded.
_C.WIDTH_THRESHOLD = 40  # px


# Track Parameters
_C.TRACK = CN()
# The threshold for KalmanFilter IOU matching (1-iou, see iou_matching.py:80)
_C.TRACK.MAX_IOU_DISTANCE = 0.7
# The maximum number of consecutive missing time before the track state is discarded.
_C.TRACK.MAX_AGE_TIME = 4.  # sec
# Number of consecutive detectioning time before the track is confirmed
_C.TRACK.N_INIT_TIME = 0.5  # sec
