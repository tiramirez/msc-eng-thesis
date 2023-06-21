## 
# This code is base on Tensorflows Object Detection demo
# SOURCE: https://github.com/tensorflow/models/blob/6f6429402e3ff1b02db84e449ce2a550fed59cf3/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb

# conda activate .\env\

# # Imports

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

import pandas as pd

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')


# ## Object detection imports
# Here are the imports from the object detection module.


from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 
# What model to download.

MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17' 
# MODEL_NAME = 'faster_rcnn_resnet101_kitti_2018_01_28'
# MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28'
# MODEL_NAME = 'faster_rcnn_nas_coco_2018_01_28'

# List of the strings that is used to add correct label for each box.

# PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt') ## ssd_mobilenet -- 80 labels -- 2 images -- 24.43 seconds
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt') ## faster_rcnn_nas -- 80 labels -- 2 images -- 24.43 seconds
# PATH_TO_LABELS = os.path.join('data', 'mscoco_wekun.pbtxt') ## 14 labels -- 2 images -- 23.30 seconds
# PATH_TO_LABELS = os.path.join('data', 'kitti_label_map.pbtxt') ## 2 labels -- 2 images -- 52.23 seconds
# PATH_TO_LABELS = os.path.join('data', 'oid_object_detection_challenge_500_label_map.pbtxt')## 500 labels -- 2 images -- 184.10 seconds
# PATH_TO_LABELS = os.path.join('data', 'oid_wekun.pbtxt')## 8 labels -- 2 images -- 179.10 seconds


MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


# ## Download and extract Model
if not MODEL_NAME in os.listdir():
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.
# from google.protobuf import text_format

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
#         text_format.Merge(serialized_graph, od_graph_def) 
        tf.import_graph_def(od_graph_def, name='')


# ## Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# ## Env setup
# This is needed to display the images.
# %matplotlib inline


# # Detection
df = pd.read_csv('answers_16-09-2019.csv')
question = 2
img_used = pd.DataFrame(pd.concat([df.loc[(df['question']==question),'img_1'],
                                   df.loc[(df['question']==question),'img_2']],
                                  axis=0,sort=True),columns=pd.Index(['id']))
IMAGE_PATHS = img_used['id'].unique()


# # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = '~/Wekun/wekun/static/img'

# # TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, IMG_PATH) for IMG_PATH in os.listdir(PATH_TO_TEST_IMAGES_DIR)]
# IMAGE_PATHS = [ i.replace('.jpg','') for i in os.listdir(PATH_TO_TEST_IMAGES_DIR) ] 

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
              'num_detections', 'detection_boxes', 'detection_scores',
              'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[1], image.shape[2])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: image})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
              'detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


import time
start_time = time.time()
# main()
print("--- %s seconds (START) ---" % (time.time() - start_time))
output = pd.DataFrame()
n = 0
# for image_id in IMAGE_PATHS:
for image_id in IMAGE_PATHS[0:100]:
    print("{} --- {} seconds ---".format(n,(time.time() - start_time)))
    n += 1
    image_file = image_id + '.jpg'
    image_path = os.path.join(os.path(PATH_TO_TEST_IMAGES_DIR), image_file)
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict =         run_inference_for_single_image(image_np_expanded, detection_graph)
    boxes_df = pd.DataFrame(output_dict['detection_boxes'], columns=pd.Index(['ymax','ymin','xmax','xmin']))
    scores_df = pd.DataFrame(output_dict['detection_scores'], columns=pd.Index(['score']))
    class_df = pd.DataFrame(output_dict['detection_classes'], columns=pd.Index(['class']))
    iter_output = pd.concat([class_df, scores_df, boxes_df], axis = 1)
    iter_output['img_id'] = image_id
    output = pd.concat([output,iter_output[(iter_output['score'] > 0.5)]], axis=0)
    
print("--- %s seconds (END) ---" % (time.time() - start_time))


output.to_csv('od_output_2_00-1k.csv', index=False)

