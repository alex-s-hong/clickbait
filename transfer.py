from __future__ import print_function, division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import copy
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from matplotlib import pyplot as plt

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  try:
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
  except: ValueError
  return np.array(image.getdata()).reshape((im_height, im_width, 4)).astype(np.uint8)
    


NUM_CLASSES = 90

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())
    
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

filelist = []
for i in os.listdir(directory):
  filelist.append(i)
  
TEST_IMAGE_PATHS = [ os.path.join(directory, file) for file in filelist]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

dump = []

def no_rgb(file):
  image = Image.open(file)
  lst = list(image.getdata())
  if type(lst[0]) == int:
    return True
  else:
    return False

for image_path in TEST_IMAGE_PATHS:
  if no_rgb(image_path):
    dump.append(image_path)
    TEST_IMAGE_PATHS.remove(image_path)

print("corrupted files")
print(dump)

image_df = []

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # in case the image has alpha mask (RBGA format), drop the fourth
      image_np = image_np[:, :, :3]
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # store the information to prepared list of list
      inner = [0]*91
      inner[0] = image_path
      for i in range(len(scores[0])):
        if (scores[0][i]< 0.5):
          break;
        inner[int(classes[0][i])]+=1
      image_df.append(inner)
      #since it causes error, pop the processed image path
      #TEST_IMAGE_PATHS.pop(0)
      
col = [None]*91
col[0]='PostImage'
for i in range(91):
  if i in category_index:
    col[i] = category_index[i]['name']
  #print(i)
        
df2 = pd.DataFrame(image_df, columns = col)
df2.to_csv(r'/content/Drive/My Drive/Spring 2019/ML/sample for project/big_info.csv', index= False)