import unittest
import numpy as np
from ImageBased3DPointCloudLibrary.depth_estimation import *


def arrange_test_image(shape = (100,100)):
    image_path = "temp_image.jpg"
    image = Image.new('RGB', shape, color='red')
    image.save(image_path)
    return image_path

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

    def test_valid_image_path(self):
        image_path = arrange_test_image()
        
        try:
            depth_estimation_from_image(image_path)
        except Exception as e:
            self.fail(f"Error with valit image: {e}")
        os.remove(image_path)
    
    def test_invalid_image_path(self):
        image_path = "invalid_image.jpg"
        
        with self.assertRaises(FileNotFoundError):
            depth_estimation_from_image(image_path)

    def test_depth_estimation_api(self):
        image_path = arrange_test_image()

        result = depth_estimation_from_image(image_path)

        self.assertIsInstance(result, np.ndarray)
        os.remove(image_path)
    
    def test_depth_estimation_non_api(self):
        image_path = arrange_test_image()
    
        with self.assertRaises(Exception) as context:
            depth_estimation_from_image(image_path, api=False)
            self.assertEqual(str(context.exception), "Not implemented yet")
        
        os.remove(image_path)




