from transformers import pipeline
from PIL import Image
import os
import numpy as np


class _DepthEstimationPipelineManager:
    _instance = None
    _pipeline = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(_DepthEstimationPipelineManager, cls).__new__(cls)
        return cls._instance

    def get_pipeline(self):
        if self._pipeline is None:
            self._pipeline = self.load_pipeline()
        return self._pipeline

    def load_pipeline(self):
        return pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            force_download=True,
            revision="main",
        )


def depth_estimation_from_image(
    image: np.ndarray,
    api: bool = True,
):
    """
    This function returns the depth estimation of an image. There is no unit for the depth estimation; it is just a relative value.
    The bigger the number, the further the object is from the camera.

    Args:
        - image (np.ndarray): The input image as a NumPy array.
        - api (bool): Flag to determine whether to use the API for depth estimation. Default is True.

    Returns:
        - np.ndarray: The depth estimation of the image as a NumPy array. The NumPy array shape will be bigger than the input image.
    """

    image = Image.fromarray(image)

    if api:
        depth_estimation_pipeline = _DepthEstimationPipelineManager().get_pipeline()
        result = depth_estimation_pipeline(image)
        depth_map_tensor = result["predicted_depth"]
        depth_estimation = depth_map_tensor[0].numpy()
    else:
        raise Exception("Not implemented yet")

    return depth_estimation
