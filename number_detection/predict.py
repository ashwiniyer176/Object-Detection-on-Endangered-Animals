import os
from black import out

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2
import numpy as np
import matplotlib.pyplot as plt

CUSTOM_MODEL_NAME = "my_ssd_mobnet"
PRETRAINED_MODEL_NAME = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8"
PRETRAINED_MODEL_URL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"
TF_RECORD_SCRIPT_NAME = "generate_tfrecord.py"
LABEL_MAP_NAME = "label_map.pbtxt"


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
paths = {
    "WORKSPACE_PATH": os.path.join("Tensorflow", "workspace"),
    "SCRIPTS_PATH": os.path.join("Tensorflow", "scripts"),
    "APIMODEL_PATH": os.path.join("Tensorflow", "models"),
    "ANNOTATION_PATH": os.path.join("Tensorflow", "workspace", "annotations"),
    "IMAGE_PATH": os.path.join("Tensorflow", "workspace", "images"),
    "MODEL_PATH": os.path.join("Tensorflow", "workspace", "models"),
    "PRETRAINED_MODEL_PATH": os.path.join(
        "Tensorflow", "workspace", "pre-trained-models"
    ),
    "CHECKPOINT_PATH": os.path.join(
        "Tensorflow", "workspace", "models", CUSTOM_MODEL_NAME
    ),
    "OUTPUT_PATH": os.path.join(
        "Tensorflow", "workspace", "models", CUSTOM_MODEL_NAME, "export"
    ),
    "TFJS_PATH": os.path.join(
        "Tensorflow", "workspace", "models", CUSTOM_MODEL_NAME, "tfjsexport"
    ),
    "TFLITE_PATH": os.path.join(
        "Tensorflow", "workspace", "models", CUSTOM_MODEL_NAME, "tfliteexport"
    ),
    "PROTOC_PATH": os.path.join("Tensorflow", "protoc"),
}

files = {
    "PIPELINE_CONFIG": os.path.join(
        "Tensorflow", "workspace", "models", CUSTOM_MODEL_NAME, "pipeline.config"
    ),
    "TF_RECORD_SCRIPT": os.path.join(paths["SCRIPTS_PATH"], TF_RECORD_SCRIPT_NAME),
    "LABELMAP": os.path.join(paths["ANNOTATION_PATH"], LABEL_MAP_NAME),
}

TRAINING_SCRIPT = os.path.join(
    paths["APIMODEL_PATH"], "research", "object_detection", "model_main_tf2.py"
)

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files["PIPELINE_CONFIG"])
detection_model = model_builder.build(model_config=configs["model"], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths["CHECKPOINT_PATH"], "ckpt-4")).expect_partial()


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    print("Detections:", detections["detection_multiclass_scores"])
    return detections


category_index = label_map_util.create_category_index_from_labelmap(files["LABELMAP"])


def get_coordinates(x, y, bounding_boxes, padding=0):
    ymin = max(round(bounding_boxes[0] * x) - padding, 0)
    xmin = max(round(bounding_boxes[1] * y) - padding, 0)
    ymax = max(round(bounding_boxes[2] * x) + padding, 0)
    xmax = max(round(bounding_boxes[3] * y) + padding, 0)

    return np.array([ymin, xmin, ymax, xmax])


def get_multiple_coordinates(x, y, arr_bb: np.array):
    return np.array([get_coordinates(x, y, i, padding=5) for i in arr_bb])


def output_detection_boxes_with_score(IMAGE_PATH, num_boxes=8, print_detections=False):
    img = cv2.imread(IMAGE_PATH)
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
    detections["detection_multiclass_scores"] = detections[
        "detection_multiclass_scores"
    ].astype(np.float64)
    result = {
        "detection_boxes": get_multiple_coordinates(
            x, y, detections["detection_boxes"][:num_boxes]
        ),
        "detection_scores": detections["detection_scores"][:num_boxes],
        "detection_classes": detections["detection_classes"],
        "detection_multiclass_scores": np.argmax(
            detections["detection_multiclass_scores"], axis=1
        ),
    }
    if print_detections:
        print(result)
    return result


def draw_predictions(IMAGE_PATH, result, threshold=0, padding=0):
    img = cv2.imread(IMAGE_PATH)
    # print(result)
    boxes, score = result["detection_boxes"], result["detection_scores"]
    for i in range(len(score)):
        if float(score[i]) < threshold:
            continue
        y, x, h, w = boxes[i]
        cv2.rectangle(
            img, (x - padding, y - padding), (w + padding, h + padding), (0, 255, 0), 2
        )
        # print(x-padding,y-padding,w+padding,h+padding)
        return img


def save_output_to_folder(IMAGE_PATH, path_to_save, results, threshold=0.2, values=[]):
    # detections = output_detection_boxes_with_score(IMAGE_PATH)
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
            os.path.join(path_to_save, img_name + "_img_" + str(cnt) + ".jpg"),
            cropped_image,
        )
        path_to_save = s


if __name__ == "__main__":
    for idx, img in enumerate(os.listdir("./data/test")):
        i = np.random.randint(0, len(os.listdir("./data/test")))
        input_img = os.listdir("./data/test")[i]
        IMAGE_PATH = f"C:/Users/Ashwin/Projects/Object-Detection-on-Animals/data/test/{input_img}"
        results = output_detection_boxes_with_score(
            IMAGE_PATH,
            num_boxes=1,
        )
        output_image = draw_predictions(IMAGE_PATH, results)
        # label_id_offset = 1
        # image_np = cv2.imread(IMAGE_PATH)

        # image_np_with_detections = image_np.copy()
        # output_image = viz_utils.visualize_boxes_and_labels_on_image_array(
        #     image_np_with_detections,
        #     results["detection_boxes"],
        #     results["detection_classes"] + label_id_offset,
        #     results["detection_scores"],
        #     category_index,
        #     use_normalized_coordinates=False,
        #     max_boxes_to_draw=8,
        #     min_score_thresh=0.3,
        #     agnostic_mode=False,
        # )
        cv2.imwrite(
            f"C:/Users/Ashwin/Projects/Object-Detection-on-Animals/output/predicted_{input_img}___{idx}.jpg",
            output_image,
        )
