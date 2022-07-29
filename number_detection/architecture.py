import os
from utils.models import MLModel
from .number_detection_utils import *
from .digit import Digit
import time

class NumberDetection(MLModel):
    # Required
    def __init__(self):
        """
        Initialize a Digit Recognition model with a given set of weights

        Args:
            model_weights: Model wights file with .pt extension
        """

        self.model = detection_model

    # Required
    def predict(self, image, threshold = 0.2):
        """
        Given an image or file path returns the prediction of the model

        Args:
            image (numpy.ndarray, string): A NumPy array representation of the input image or the path to an image

            Note: Having one of the two inputs is mandatory.

        Raises:
            exceptions.InvalidInputError: In case no inputs are given or both inputs are given, the function raises an error

        Returns:
            prediction (string): A string that predicts the number
        """
        start = time.time()

        if type(image) is str and os.path.exists(image):
            image = self.load_image_from_file_path(image)

        results = output_detection_boxes_with_score(image, print_detections=False)
        digit = Digit(results)
        digit.remove_outlier()
        digit.remove_concurrent(relax_factor = 10)
        # digit.remove_boundary()
        results = digit.results
        boxes, score = results["detection_boxes"], results["detection_scores"]
        cropped_images_arr = []

        for cnt, num in enumerate(boxes):
            if threshold > score[cnt]:
                continue
            cropped_image = image[num[0] : num[2], num[1] : num[3]]
            cropped_images_arr.append(cropped_image)

        end = time.time()
        print(f"Executed in {end-start} seconds")
        return cropped_images_arr