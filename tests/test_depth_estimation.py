import unittest
import numpy as np
from ImageBased3DPointCloudLibrary.depth_estimation import *


class TestDepthMap(unittest.TestCase):

    """
      This test asserts if the depth_estimation_pipeline loads successfully.
      It's commented out because it takes too long
    """


    # def test_single_depth_estimation_api(self):
    #     de_pipeline = DepthEstimationPipelineManager()
    #     other_pipeline = DepthEstimationPipelineManager()

    #     self.assertIsInstance(de_pipeline, DepthEstimationPipelineManager)
    #     self.assertIsInstance(other_pipeline, DepthEstimationPipelineManager)
    #     self.assertEqual(de_pipeline, other_pipeline)

    def test_valid_image(self):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        try:
            depth_estimation_from_image(image)
        except Exception as e:
            self.fail(f"Error with valit image: {e}")

    
    def test_depth_estimation_api(self):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        result = depth_estimation_from_image(image)

        self.assertIsInstance(result, np.ndarray)
    
    def test_depth_estimation_non_api(self):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
        with self.assertRaises(Exception) as context:
            depth_estimation_from_image(image, api=False)
            self.assertEqual(str(context.exception), "Not implemented yet")
        





