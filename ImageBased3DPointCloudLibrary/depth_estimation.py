from transformers import pipeline
from PIL import Image
import os
import numpy as np

class DepthEstimationPipelineManager:
    _instance = None
    _pipeline = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DepthEstimationPipelineManager, cls).__new__(cls)
        return cls._instance

    def get_pipeline(self):
        if self._pipeline is None:
            self._pipeline = self.load_pipeline()
        return self._pipeline

    def load_pipeline(self):
        return pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

def depth_estimation_from_image(
        image: np.ndarray, 
        api: bool = True,
):
    image = Image.fromarray(image)


    if api:
        depth_estimation_pipeline = DepthEstimationPipelineManager().get_pipeline()
        result = depth_estimation_pipeline(image)
        depth_map_tensor = result["predicted_depth"]
        depth_estimation = depth_map_tensor[0].numpy()
    else:
        raise Exception("Not implemented yet")
        
    return depth_estimation
