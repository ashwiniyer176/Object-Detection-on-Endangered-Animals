import os
from number_detection import files, paths
from number_detection import PRETRAINED_MODEL_NAME, PRETRAINED_MODEL_URL

for path in paths.values():
    if not os.path.exists(path):
        if os.name == "posix":
            os.system(f"mkdir -p {path}")
        if os.name == "nt":
            os.system(f"mkdir {path}")

if os.name == "nt":
    os.system("pip install wget")
    import wget

if not os.path.exists(
    os.path.join(paths["APIMODEL_PATH"], "research", "object_detection")
):
    os.system(
        f"git clone https://github.com/tensorflow/models {paths['APIMODEL_PATH']}"
    )

# Install Tensorflow Object Detection

if os.name == "posix":
    os.system(f"apt-get install protobuf-compiler")
    os.system(
        f"cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install . "
    )

if os.name == "nt":
    if not os.path.exists(os.path.join(".", "protoc-3.15.6-win64.zip")):
        url = "https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"
        wget.download(url)
    # os.system(f"move protoc-3.15.6-win64.zip {paths['PROTOC_PATH']}")
    if not os.path.exists(os.path.join(".", "include")):
        os.mkdir("include")
    os.system(
        f"unzip protoc-3.15.6-win64.zip && tar cvf protoc-3.15.6-win64.tar ./include/ "
    )  # && tar -xf protoc-3.15.6-win64.tar")
    os.system(f"move protoc-3.15.6-win64.tar {paths['PROTOC_PATH']}")
    os.system(f"cd {paths['PROTOC_PATH']} && tar -xf protoc-3.15.6-win64.tar")
    if not os.path.exists("bin"):
        os.mkdir("bin")
    os.environ["PATH"] += os.pathsep + os.path.abspath(os.path.join(".", "bin"))
    # print(os.environ['PATH'])
    os.system(
        f"cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && copy object_detection\\packages\\tf2\\setup.py setup.py && python setup.py build && python setup.py install"
    )
    os.system(f"cd Tensorflow/models/research/slim && pip install -e . ")

VERIFICATION_SCRIPT = os.path.join(
    paths["APIMODEL_PATH"],
    "research",
    "object_detection",
    "builders",
    "model_builder_tf2_test.py",
)
# Verify Installation
os.system(f"python {VERIFICATION_SCRIPT}")

os.system(f"pip install tensorflow --upgrade")
# os.system(f"pip uninstall protobuf matplotlib -y")
os.system(f"pip install protobuf")

if not os.path.exists(
    os.path.join(paths["PRETRAINED_MODEL_PATH"], (PRETRAINED_MODEL_NAME + ".tar.gz"))
):
    if os.name == "posix":
        os.system(f"wget {PRETRAINED_MODEL_URL}")
        os.system(
            f"mv {PRETRAINED_MODEL_NAME+'.tar.gz'} {paths['PRETRAINED_MODEL_PATH']}"
        )
        os.system(
            f"cd {paths['PRETRAINED_MODEL_PATH']} && tar -zxvf {PRETRAINED_MODEL_NAME+'.tar.gz'}"
        )

    if os.name == "nt":
        wget.download(PRETRAINED_MODEL_URL)
        os.system(
            f"move {PRETRAINED_MODEL_NAME+'.tar.gz'} {paths['PRETRAINED_MODEL_PATH']}"
        )
        os.system(
            f"cd {paths['PRETRAINED_MODEL_PATH']} && tar -zxvf {PRETRAINED_MODEL_NAME+'.tar.gz'}"
        )

labels = [
    {"name": "Bear", "id": 26},
    {"name": "Brown_bear", "id": 1},
    {"name": "Cheetah", "id": 2},
    {"name": "Crocodile", "id": 3},
    {"name": "Deer", "id": 4},
    {"name": "Elephant", "id": 5},
    {"name": "Jaguar", "id": 6},
    {"name": "Leopard", "id": 7},
    {"name": "Lion", "id": 8},
    {"name": "Lizard", "id": 9},
    {"name": "Lynx", "id": 10},
    {"name": "Panda", "id": 11},
    {"name": "Polar_bear", "id": 12},
    {"name": "Red_panda", "id": 13},
    {"name": "Rhinoceros", "id": 14},
    {"name": "Sea_lion", "id": 15},
    {"name": "Sea_turtle", "id": 16},
    {"name": "Shark", "id": 17},
    {"name": "Snake", "id": 18},
    {"name": "Sparrow", "id": 19},
    {"name": "Starfish", "id": 20},
    {"name": "Tiger", "id": 21},
    {"name": "Tortoise", "id": 22},
    {"name": "Turtle", "id": 23},
    {"name": "Whale", "id": 24},
    {"name": "Zebra", "id": 25},
]

with open(files["LABELMAP"], "w") as f:
    for label in labels:
        f.write("item { \n")
        f.write("\tname:'{}'\n".format(label["name"]))
        f.write("\tid:{}\n".format(label["id"]))
        f.write("}\n")

if not os.path.exists("Tensorflow/workspace/images/train"):
    os.mkdir("Tensorflow/workspace/images/train")
    os.system(f"copy data/cropped/* Tensorflow/workspace/images/train/")

ARCHIVE_FILES = os.path.join(paths["IMAGE_PATH"], "archive.tar.gz")

if os.path.exists(ARCHIVE_FILES):
    os.system(f"tar -zxvf {ARCHIVE_FILES}")

if not os.path.exists(files["TF_RECORD_SCRIPT"]):
    os.system(f"copy notebooks/generate_tfrecord.py {paths['SCRIPTS_PATH']}")

os.system(
    f"python {files['TF_RECORD_SCRIPT']} -x {os.path.join(paths['IMAGE_PATH'], 'train')} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'train.record')} -c {os.path.join(paths['ANNOTATION_PATH'], 'sample.csv')}"
)
os.system(
    f"copy {os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config')} {os.path.join(paths['CHECKPOINT_PATH'])}"
)

if os.name == "posix":
    os.system(
        f"cp {os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config')} {os.path.join(paths['CHECKPOINT_PATH'])}"
    )
if os.name == "nt":
    os.system(
        f"copy {os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config')} {os.path.join(paths['CHECKPOINT_PATH'])}"
    )

# os.system("pip install apache-bea avro-python3")

import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

config = config_util.get_configs_from_pipeline_file(files["PIPELINE_CONFIG"])

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

with tf.io.gfile.GFile(files["PIPELINE_CONFIG"], "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

pipeline_config.model.ssd.num_classes = len(labels)
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(
    paths["PRETRAINED_MODEL_PATH"], PRETRAINED_MODEL_NAME, "checkpoint", "ckpt-0"
)
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path = files["LABELMAP"]
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [
    os.path.join(paths["ANNOTATION_PATH"], "train.record")
]
pipeline_config.eval_input_reader[0].label_map_path = files["LABELMAP"]
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [
    os.path.join(paths["ANNOTATION_PATH"], "test.record")
]

config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(files["PIPELINE_CONFIG"], "wb") as f:
    f.write(config_text)
