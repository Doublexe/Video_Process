import os
import numpy as np


class _Detector(object):
    """Detector takes images, and produces per image bboxes.

    Callable obejct:

    Parameters
    ----------
    image : ndarray
        The single full image in format (batch, H, W, C)

    Returns
    ----------
    img_bboxes : List[ndarray]
        The batch of lists of bounding boxes in format (batch, num_bbox, x_min, y_min, width, height).
        num_bbox dim can be heterogenous.
        If there are no bboxes detected for an image, output empty array for that image.
    img_scores : List[ndarray]
        The confidence level for each detected bbox (ONLY FOR PERSON DETECTION) in formar (batch, num_bbox, score).
        num_bbox dim can be heterogenous.
        If there are no bboxes detected for an image, output empty array for that image.
    """

    def __call__(self, images):
        img_bboxes = None
        img_scores = None
        return img_bboxes, img_scores

    def _get_batched_inputs(self, images, batch_size):
        """Given images, return the batched inputs and per image indices.

        Parameters
        ----------
        images : ndarray
            In format (batch, H, W, C)
        batch_size : int
            The batch_size for inputs.

        Returns
        ----------
        inputs : List[ndarray]
            In format (list_idx, batch_size, H, W, C), where list_idx*batch_size collects batch and num_bboxes dimensions.
            If no images at all, return empty array.
        """
        input_batch = images.shape[0]
        new_batch_indices = [0]
        while new_batch_indices[-1] + batch_size <= input_batch:
            new_batch_indices.append(new_batch_indices[-1]+batch_size)
        if new_batch_indices[-1] != input_batch:
            new_batch_indices.append(input_batch)
        inputs = [images[new_batch_indices[i]:new_batch_indices[i+1]] for i in range(len(new_batch_indices) - 1)]
        return inputs

class YOLOv3Detector(_Detector):
    def __init__(self, cfgfile, weightfile, namesfile, device):
        from models.yolov3.detector import YOLOv3
        super(YOLOv3Detector, self).__init__()
        self.batch = 1
        self.model = YOLOv3(cfgfile,
                            weightfile,
                            namesfile=namesfile,
                            yolo_device=device,
                            is_plot=False,
                            # NOTE: xywh is (x_center, y_center, w, h)
                            is_xywh=True
                            )  # this is not a pytorvh model

    def __call__(self, images):
        batches = self._get_batched_inputs(images, batch_size = self.batch)
        img_bboxes = []
        img_scores = []
        for batch in batches:
            bboxes, cls_confs, cls_ids = self.model(batch[0])

            # Only choose the person boxes out. Set the score for other classes to 0.
            person_cls_id = self.model.class_names.index('person')
            mask = cls_ids==person_cls_id

            # Modify the bboxes from (x_center, y_center, w, h) to (x_min, y_min, w, h)
            if bboxes is None:
                bboxes = np.array([])
                scores = np.array([])
                img_bboxes.append(bboxes)
                img_scores.append(scores)
            else:
                bboxes = bboxes[mask]
                scores = cls_confs[mask]
                bboxes[:, :2] = bboxes[:, :2] - bboxes[:, 2:] / 2

            img_bboxes.append(bboxes)
            img_scores.append(scores)

        return img_bboxes, img_scores


class MaskRCNNDetectorTF(_Detector):
    """Mask RCNN pretrained on COCO. Tensorflow.

    See: https://github.com/matterport/Mask_RCNN
    """
    def __init__(self, coco_weightfile, batch, device):
        """
        device : string
            TF devices, such as "/cpu:0", "/device:GPU:1".
            See: https://www.tensorflow.org/guide/using_gpu
        """
        super(MaskRCNNDetectorTF, self).__init__()
        import models.maskrcnn_tf.mrcnn.model as modellib
        from models.maskrcnn_tf.mrcnn.config import Config
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        class CocoConfig(Config):
            """Configuration for training on MS COCO.
            Derives from the base Config class and overrides values specific
            to the COCO dataset.
            """
            # Give the configuration a recognizable name
            NAME = "coco"

            # We use a GPU with 12GB memory, which can fit two images.
            # Adjust down if you use a smaller GPU.
            IMAGES_PER_GPU = 2

            # Uncomment to train on 8 GPUs (default is 1)
            # GPU_COUNT = 8

            # Number of classes (including background)
            NUM_CLASSES = 1 + 80  # COCO has 80 classes

        class InferenceConfig(CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = batch

        MODEL_DIR = os.path.dirname(coco_weightfile)
        COCO_MODEL_PATH = coco_weightfile
        config = InferenceConfig()
        # config.display()

        self.device = device
        self.batch = batch

        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
        self.model.load_weights(COCO_MODEL_PATH, by_name=True)

    def __call__(self, images):
        import tensorflow as tf
        batches = self._get_batched_inputs(images, self.batch)

        # Output format: List[Dict[field]]
        # List aggrefates the batch. len(List) = batch_size
        # Dict contains output info for an image:
        #   1.rois (ndarray[N, 4]): the predicted boxes in [y1, x1, y2, x2] format, with values between 0 and H and 0 and W
        #   2.class_ids (ndarray[N]): the predicted labels for each image
        #   3.scores (ndarray[N]): the scores or each prediction
        #   4.masks (ndarray[N, 1, H, W]): the predicted masks for each instance, in 0-1 range. In order to obtain the final segmentation masks, the soft masks can be thresholded, generally with a value of 0.5 (mask >= 0.5)
        output_dicts = []
        with tf.device(self.device):
            for batch in batches:
                output_dicts.extend(self.model.detect(batch))

        img_bboxes = []
        img_scores = []
        for output_dict in output_dicts:
            bboxes, cls_labels, cls_confs = output_dict['rois'], output_dict['class_ids'], output_dict['scores']

            if bboxes.size == 0:
                bboxes = np.array([])
                scores = np.array([])
                img_bboxes.append(bboxes)
                img_scores.append(scores)
            else:
                # Only choose the person boxes out. Set the score for other classes to 0.
                person_cls_id = 1  # MAGIC_NUM: 'person' class
                # class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                #    'bus', 'train', 'truck', 'boat', 'traffic light',
                #    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                #    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                #    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                #    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                #    'kite', 'baseball bat', 'baseball glove', 'skateboard',
                #    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                #    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                #    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                #    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                #    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                #    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                #    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                #    'teddy bear', 'hair drier', 'toothbrush']
                mask = cls_labels==person_cls_id

                bboxes = bboxes[mask]
                scores = cls_confs[mask]

                # Modify the bboxes from (y_min, x_min, y_max, x_max) to (x_min, y_min, w, h)
                bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]
                bboxes = bboxes[:, [1, 0, 3, 2]]

                img_bboxes.append(bboxes)
                img_scores.append(scores)

        return img_bboxes, img_scores



class MaskRCNNDetectorTorch(_Detector):
    """Mask RCNN pretrained on COCO. Pytorch.

    # NOTE: The inputs for maskrcnn_resnet50_fpn can be a list of heterogenous tensors, in format (C, H, W)

    See: https://pytorch.org/docs/stable/torchvision/models.html
    """
    def __init__(self, batch, device):
        from torchvision.models.detection import maskrcnn_resnet50_fpn
        import torch
        # from torchvision import transforms
        super(MaskRCNNDetectorTorch, self).__init__()
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.batch = batch
        self.device = torch.device(device)
        self.model.to(self.device)
        self.MEAN = np.array([.485, .456, .406])
        self.STD = np.array([.299, .224, .225])
        # self.transform = transforms.Compose([
        #     transforms.Normalize(self.MEAN_PIXEL)
        # ])
        # self.transfrom = transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
        # raise NotImplementedError("No pre-trained weight file for this yet.")

    def __call__(self, images):
        import torch
        #
        # images = images / 255
        # images = (images - self.MEAN) / self.STD
        imgs = torch.from_numpy(images).float().permute(0, 3,1,2) # TODO: better normalization
        batches = self._get_batched_inputs(imgs, self.batch)
        with torch.no_grad():
            output_dicts=[]
            for batch in batches:
                imgs = batch.to(self.device)
                imgs = imgs/255

                # Output format: List[Dict[field]]
                # List aggrefates the batch. len(List) = batch_size
                # Dict contains output info for an image:
                #   1.boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values between 0 and H and 0 and W
                #   2.labels (Int64Tensor[N]): the predicted labels for each image
                #   3.scores (Tensor[N]): the scores or each prediction
                #   4.masks (UInt8Tensor[N, height, width, num_instances]): the predicted masks for each instance, in 0-1 range. In order to obtain the final segmentation masks, the soft masks can be thresholded, generally with a value of 0.5 (mask >= 0.5)

                output_dicts.extend(self.model(imgs))

            img_bboxes = []
            img_scores = []

            for output_dict in output_dicts:
                bboxes, cls_labels, cls_confs = output_dict['boxes'], output_dict['labels'], output_dict['scores']

                if bboxes.size(0) == 0:
                    bboxes = np.array([])
                    scores = np.array([])
                    img_bboxes.append(bboxes)
                    img_scores.append(scores)
                else:
                    # Only choose the person boxes out. Set the score for other classes to 0.
                    person_cls_id = 1  # MAGIC_NUM: 'person' class
                    mask = cls_labels==person_cls_id

                    bboxes = bboxes[mask]
                    scores = cls_confs[mask]

                    # Modify the bboxes from (x_min, y_min, x_max, y_max) to (x_min, y_min, w, h)
                    bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]

                    bboxes, scores = bboxes.cpu().data.numpy(), scores.cpu().data.numpy()
                    img_bboxes.append(bboxes)
                    img_scores.append(scores)

        return img_bboxes, img_scores


class FasterRCNNTorch(_Detector):
    """Faster RCNN pretrained on COCO. Pytorch.

    # NOTE: The inputs for maskrcnn_resnet50_fpn can be a list of heterogenous tensors, in format (C, H, W)

    See: https://pytorch.org/docs/stable/torchvision/models.html
    """
    def __init__(self, batch, device):
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        import torch
        # from torchvision import transforms
        super(FasterRCNNTorch, self).__init__()
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.batch = batch
        self.device = torch.device(device)
        self.model.to(self.device)

    def __call__(self, images):
        import torch
        imgs = torch.from_numpy(images).float().permute(0, 3,1,2)
        batches = self._get_batched_inputs(imgs, self.batch)
        with torch.no_grad():
            output_dicts=[]
            for batch in batches:
                imgs = batch.to(self.device)
                imgs = imgs/255

                # Output format: List[Dict[field]]
                # List aggrefates the batch. len(List) = batch_size
                # Dict contains output info for an image:
                #   1.boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values between 0 and H and 0 and W
                #   2.labels (Int64Tensor[N]): the predicted labels for each image
                #   3.scores (Tensor[N]): the scores or each prediction
                #   4.masks (UInt8Tensor[N, height, width, num_instances]): the predicted masks for each instance, in 0-1 range. In order to obtain the final segmentation masks, the soft masks can be thresholded, generally with a value of 0.5 (mask >= 0.5)

                output_dicts.extend(self.model(imgs))

            img_bboxes = []
            img_scores = []

            for output_dict in output_dicts:
                bboxes, cls_labels, cls_confs = output_dict['boxes'], output_dict['labels'], output_dict['scores']

                if bboxes.size(0) == 0:
                    bboxes = np.array([])
                    scores = np.array([])
                    img_bboxes.append(bboxes)
                    img_scores.append(scores)
                else:
                    # Only choose the person boxes out. Set the score for other classes to 0.
                    person_cls_id = 1  # MAGIC_NUM: 'person' class
                    mask = cls_labels==person_cls_id

                    bboxes = bboxes[mask]
                    scores = cls_confs[mask]

                    # Modify the bboxes from (x_min, y_min, x_max, y_max) to (x_min, y_min, w, h)
                    bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]

                    bboxes, scores = bboxes.cpu().data.numpy(), scores.cpu().data.numpy()
                    img_bboxes.append(bboxes)
                    img_scores.append(scores)

        return img_bboxes, img_scores


class DetectorFactory(object):
    def make_detector(self, cfg, option):
        """Get detector object according to the option:

        Parameters
        ----------
        cfg: config node (yacs) or node structures
            The configuration for the detector.
        option : str
            The name for the detector:
                1. yolov3
                2. mask-rcnn-tf
                3. mask-rcnn-torch

        Returns
        ----------
        detector: _Detector
            See _Detector.
        """
        print("Loading {}...".format(option))

        if option == "yolov3":

            detector = YOLOv3Detector(
                os.path.join(cfg.MODEL_DATA_PATH, 'yolov3/yolo_v3.cfg'),
                os.path.join(cfg.MODEL_DATA_PATH, 'yolov3/yolov3.weights'),
                os.path.join(cfg.MODEL_DATA_PATH, 'yolov3/coco.names'),
                cfg.DETECTOR.DEVICE)
            print("Resize the input to {}.".format(detector.model.size))

        elif option == "mask-rcnn-tf":
            detector = MaskRCNNDetectorTF(
                coco_weightfile=os.path.join(cfg.MODEL_DATA_PATH, 'maskrcnn_tf/mask_rcnn_coco.h5'),
                batch=cfg.DETECTOR.BATCH,
                device=cfg.DETECTOR.DEVICE
            )
            print("Arbitrary sizes of input are valid for {}.".format(option))

        elif option == "mask-rcnn-torch":
            detector = MaskRCNNDetectorTorch(
                batch=cfg.DETECTOR.BATCH,
                device=cfg.DETECTOR.DEVICE
            )
            print("Arbitrary sizes of input are valid for {}.".format(option))

        elif option == "faster-rcnn-torch":
            detector = FasterRCNNTorch(
                batch=cfg.DETECTOR.BATCH,
                device=cfg.DETECTOR.DEVICE
            )
            print("Arbitrary sizes of input are valid for {}.".format(option))

        else:
            raise ValueError("The detector option is invalid.")

        print("The {} has been loaded.".format(option))
        return detector
