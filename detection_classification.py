# Article Detection Using frozen model 

import os
import cv2
import numpy as np
import tensorflow as tf #  v2.x
import sys


from utils import label_map_util
from utils import visualization_utils as vis_util

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# version compatibility patch for tf
utils_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile

# Image
IMG = "DJI_00730137.jpg"

# Directory containing the inference graph 
MDL_DIR = 'inference_graph'

# Paths to frozen detection graph .pb file (frozen model), label map and image
CWD_PTH = os.getcwd()
CKPT_PTH = os.path.join(CWD_PTH,MDL_DIR,'frozen_inference_graph.pb')
LBLS_PTH = os.path.join(CWD_PTH,'training','labelmap.pbtxt')
IMG_PTH = os.path.join(CWD_PTH,'images/test', IMG)

NUM_CLSS = 3

# Load the label map. (maps categories to numbers)
label_map = label_map_util.load_labelmap(LBLS_PTH)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLSS, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the graph into memory.
detection_graph = tf.Graph() 
with detection_graph.as_default(): 
    od_graph_def =  tf.compat.v1.GraphDef() # GraphDef is the .pb file 
    with tf.io.gfile.GFile(CKPT_PTH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    # evaluate the nodes
    sess =  tf.compat.v1.Session(graph=detection_graph) 


# Input image tensor
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors: scores and detection boxes
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

image = cv2.imread(IMG_PTH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_expanded = np.expand_dims(image_rgb, axis=0)

# run method in session object with input graphs and tensors
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8,
    min_score_thresh=0.60)

cv2.imshow('Object detector', image)
cv2.waitKey(0)
cv2.destroyAllWindows() # Do clean up
