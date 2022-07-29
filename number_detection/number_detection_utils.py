import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2
import numpy as np
from . import files, paths

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files["PIPELINE_CONFIG"])
detection_model = model_builder.build(model_config=configs["model"], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths["CHECKPOINT_PATH"], "ckpt-5")).expect_partial()


@tf.function
def detect_fn(image):
    """function to run the predict function on any given image by the loaded model

    Args:
        image (tf.Tensor): tensor of the matrix of the image

    Returns:
        dict: result contains bounding box coordinates and confidence level of each box
    """
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


# category_index = label_map_util.create_category_index_from_labelmap(files["LABELMAP"])


def get_coordinates(x, y, bounding_boxes, padding=0):
    """function to get the actual coordinates on the image

    Args:
        x (int): width of the image
        y (int): height of the image
        bounding_boxes (list): as returned by the model
        padding (int, optional): Add padding to the coordinates, generally useful as a backup. Defaults to 0.

    Returns:
        ndarray: array containing coordinates on image
    """
    ymin = max(round(bounding_boxes[0] * x) - padding,0)
    xmin = max(round(bounding_boxes[1] * y) - padding,0)
    ymax = max(round(bounding_boxes[2] * x) + padding,0)
    xmax = max(round(bounding_boxes[3] * y) + padding,0)

    return np.array([ymin, xmin, ymax, xmax])


def get_multiple_coordinates(x, y, arr_bb: np.array):
    """used along with get_coordinates, but for multiple boxes of the same image

    Args:
        x (int): width of the image
        y (int): height of the image
        arr_bb (np.array): array of bounding boxes

    Returns:
        np.array: containing all the bounding boxes of the given image
    """
    return np.array([get_coordinates(x, y, i, padding=5) for i in arr_bb])


def output_detection_boxes_with_score(image, num_boxes=8, print_detections=False):
    """to get the output of the results for a given image

    Args:
        IMAGE_PATH (str): a string file path of the image
        num_boxes (int, optional): number of boxes to be detected. Defaults to 8.
        print_detections (bool, optional): if true, prints the result (useful for debugging and monitoring). Defaults to False.

    Returns:
        result: final resultant bounding box as predicted by the model
    """
    if image is type(str):
        img = cv2.imread(image)
    else: 
        img = image
    x, y, _ = img.shape
    image_np = np.array(img)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop("num_detections"))
    detections = {
        key: value[0, :num_detections].numpy() for key, value in detections.items()
    }
    detections["num_detections"] = num_detections
    # detection_classes should be ints.
    detections["detection_classes"] = detections["detection_classes"].astype(np.int64)
    result = {
        "detection_boxes": get_multiple_coordinates(
            x, y, detections["detection_boxes"][:num_boxes]
        ),
        "detection_scores": detections["detection_scores"][:num_boxes],
    }
    if print_detections:
        print(result)
    return result


def draw_predictions(IMAGE_PATH, result, threshold=0, padding=0):
    """to draw and display the predictions for a given image and results

    Args:
        IMAGE_PATH (str): image file path
        result (dict): results as returned by the model
        threshold (int, optional): threshold, if considered, at which the box is valid (for score). should be between 0 and 1. Defaults to 0.
        padding (int, optional): to add a padding to the drawn boxes. Defaults to 0.

    Returns:
        img: final image with drawn boxes as per the result
    """
    if IMAGE_PATH is type(str):
        img = cv2.imread(IMAGE_PATH)
    else:
        img = IMAGE_PATH

    boxes, score = result["detection_boxes"], result["detection_scores"]
    for i in range(len(score)):
        if float(score[i]) < threshold:
            continue
        y, x, h, w = boxes[i]
        cv2.rectangle(
            img, (x - padding, y - padding), (w + padding, h + padding), (0, 255, 0), 2
        )
    
    return img


def save_output_to_folder(IMAGE_PATH, path_to_save, results, threshold=0.2, values=[]):
    """save the predicted numbers as separate images in a given folder

    Args:
        IMAGE_PATH (str): path containing the image
        path_to_save (str): path to save the images (should be a folder path)
        results (dict): results in the format returned by the model.
        threshold (float, optional): threshold at which images are considered as digits. Defaults to 0.2.
        values (list, optional): value of the digits (for labelling purposes, ignore if used in production). Defaults to [].
    """
    
    boxes, score = results["detection_boxes"], results["detection_scores"]
    img_name = IMAGE_PATH.split("\\")[-1]
    img = cv2.imread(IMAGE_PATH)
    s = path_to_save

    for cnt, num in enumerate(boxes):
        if threshold > score[cnt]:
            continue
        cropped_image = img[num[0] : num[2], num[1] : num[3]]
        try:
            if len(values) > 0:
                path_to_save = os.path.join(
                    path_to_save, (str(values[cnt]) if values[cnt] >= 0 else "UNK")
                )
        except:
            continue

        cv2.imwrite(
            os.path.join(path_to_save, img_name + "_img_" + str(cnt) + ".bmp"),
            cropped_image,
        )
        path_to_save = s