import cv2
import os
import numpy as np

# imports from models
from models.test_extractor import TestExtractor
from models.aae import AAE


def extract_image_patch(image, bbox, patch_shape=None):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
        If empty, output empty array.
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if bbox.size == 0:
        return np.array([])
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    if patch_shape is not None:
        image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image




class _Encoder(object):
    """Encoder takes images and per image bboxes, and produces features.

    Callable obejct:

    Parameters
    ----------
    images : ndarray
        The batch of full images in format (batch, H, W, C)
    img_bboxes : array_like
        The list of bounding boxes in format (batch, num_bboxes, x, y, width, height).
        The num_bboxes dim can be heterogenuous.
        All elements are floats.

    Returns
    ----------
    features : array_like
        The list of features in format (batch, num_bboxes, feat_dim).
        The num_bboxes dim can be heterogenuous.
        If there are no bboxes detected for an image, output empty array for that image.
    """
    def __call__(self, images, img_bboxes):
        features = None
        return features

    def _get_patches(self, image, bboxes, patch_shape):
        """Get image_patched : array_like in format (num_bbox, H, W, C).

        Parameters
        ----------
        image : ndarray
            The full image in format (H, W, C)
        bboxes : array_like
            The list of bounding boxes in format (num_bbox, x_min, y_min, width, height).
        patch_shape : array_like
            This parameter can be used to enforce a desired patch shape
            (height, width). First, the `bbox` is adapted to the aspect ratio
            of the patch shape, then it is clipped at the image boundaries.
            It has to be provided in order to get num_bboxed outputs.

        Returns
        ----------
        image_patches : ndarray
            In format (num_bbox, box_H, box_W, C)
            If bboxes are empty, output empty array.
        """
        image_patches = []

        for box in bboxes:
            patch = extract_image_patch(image, box, patch_shape)  #,patch_shape=image_shape
            if patch is None:
                raise ValueError("WARNING: Failed to extract image patch: %s." % str(box))
                # patch = np.random.uniform(
                #     0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)

        return image_patches

    def _get_batched_inputs(self, images, img_bboxes, patch_shape, batch_size):
        """Given images and per image bboxes, return the batched inputs and per image indices.

        Parameters
        ----------
        images : ndarray
            In format (batch, num_bboxes, box_H, box_W, C)
        img_bboxes : array_like
            The list of bounding boxes in format (batch, num_bboxes, x, y, width, height).
            The num_bboxes dim can be heterogenuous.
            All elements are floats.
        patch_shape : array_like
            This parameter can be used to enforce a desired patch shape
            (height, width). First, the `bbox` is adapted to the aspect ratio
            of the patch shape, then it is clipped at the image boundaries.
            It has to be provided in order to get batched outputs.
        batch_size : int
            The batch_size for inputs.

        Returns
        ----------
        inputs : List[ndarray]
            In format (list_idx, batch_size, H, W, C), where list_idx*batch_size collects batch and num_bboxes dimensions.
            batch_size can be heterogenuous at the tail.
            If no bboxes at all, return empty array.
        img_indices : list
            img_indices[i]:img_indices[i+1] selects the flattened patches (collects batch and num_bboxes dimensions) for the i-th image.
        """
        inputs = []
        img_indices = [0]
        for image, bboxes in zip(images, img_bboxes):
            image_patches = self._get_patches(image, bboxes, patch_shape)
            if image_patches.size != 0:
                inputs.append(image_patches)
                img_indices.append(image_patches.shape[0]+img_indices[-1])
            else:
                img_indices.append(img_indices[-1])
        if len(inputs) == 0:
            return np.array([]), img_indices
        inputs = np.concatenate(inputs)
        new_batch_indices = [0]
        while new_batch_indices[-1] + batch_size <= img_indices[-1]:
            new_batch_indices.append(new_batch_indices[-1]+batch_size)
        if new_batch_indices[-1] != img_indices[-1]:
            new_batch_indices.append(img_indices[-1])
        inputs = [inputs[new_batch_indices[i]:new_batch_indices[i+1]] for i in range(len(new_batch_indices) - 1)]
        return inputs, img_indices

    def _get_outputs(self, outputs, img_indices):
        """Recover the outputs from model to the original arragement: (batch, num_bboxes, box_H, box_W, C),
            where num_bboxes dim can be heterogenuous.

        Parameters
        ----------
        outputs : List[ndarray]
            The collected outputs from the model. In format (list_idx, batch_size, H, W, C),
            where list_idx*batch_size collects batch and num_bboxes dimensions.
            batch_size can be heterogenuous at the tail.
        img_indices : list
            img_indices[i]:img_indices[i+1] selects the flattened patches (collects batch and num_bboxes dimensions) for the i-th image.

        Returns
        ----------
        features : List[ndarry]
            In foramt [batch (num_bboxes, features)].
            If model output empty for a batch, return empty array for that batch.
        """
        if len(outputs) == 0:
            features = np.array([])
        else:
            features = np.concatenate(outputs)
        num_image = len(img_indices) - 1
        return [features[img_indices[i]:img_indices[i+1]] for i in range(num_image)]



class TestEncoder(_Encoder):
    """A feature extractor trained with reid information.

    "Retrain REID model on pedestrain dataset for better performance."
    See: https://github.com/ZQPei/deep_sort_pytorch
    """
    def __init__(self, model_path, batch, device=None):
        super(TestEncoder, self).__init__()
        self.model = TestExtractor(model_path, device)
        self.batch = batch

    def __call__(self, images, img_bboxes):
        inputs, img_indices = self._get_batched_inputs(images, img_bboxes, self.model.size, self.batch)
        features = []
        for inp in inputs:
            features.append(self.model(inp))

        return self._get_outputs(features, img_indices)


class AAEEncoder(_Encoder):
    """DualNorm + AAE.
    """
    def __init__(self, model_path, batch, device=None):
        super(AAEEncoder, self).__init__()
        self.model = AAE(model_path, device)
        self.batch = batch

    def __call__(self, images, img_bboxes):
        inputs, img_indices = self._get_batched_inputs(images, img_bboxes, (384, 128), self.batch)
        features = []
        for inp in inputs:
            features.append(self.model(inp))

        return self._get_outputs(features, img_indices)


class EncoderFactory(object):
    def make_encoder(self, cfg, option):
        """Get encoder object according to the option:

        Parameters
        ----------
        cfg : config node (yacs) or node structures
            The configuration for the encoder.
        option : str
            The name for the encoder:
                1. test

        Returns
        ----------
        encoder : _Encoder
            See _Encoder.
        """
        print("Loading {}...".format(option))

        if option == "test":

            # encoder = TestEncoder(os.path.join(cfg.MODEL_DATA_PATH, 'ckpt.t7'), cfg.ENCODER.BATCH, cfg.ENCODER.DEVICE)
            encoder = TestEncoder(os.path.join(cfg.MODEL_DATA_PATH, 'ckpt.t7'), cfg.ENCODER.BATCH, cfg.ENCODER.DEVICE)

        elif option == "aae":
            encoder = AAEEncoder(os.path.join(cfg.MODEL_DATA_PATH, 'aae/net_last_G.pth'), cfg.ENCODER.BATCH, cfg.ENCODER.DEVICE)

        else:
            raise ValueError("The encoder option is invalid.")

        print("The {} has been loaded.".format(option))
        return encoder
