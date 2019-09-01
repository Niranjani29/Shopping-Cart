import numpy as np
import os
import six.moves.urllib as urllib
#import sys
import tarfile
import tensorflow as tf
#import zipfile
import csv
#import sys
#
#from collections import defaultdict
#from io import StringIO
#from matplotlib import pyplot as plt
#from PIL import Image

import cv2
from utils import label_map_util

from utils import visualization_utils as vis_util

total_itm = [31,37,44,47,48,50,77,84]
Window_Name="-----------------------------------------------Shopping Cart---------"
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90
category_index_new={}

# ## Download Model


def csv_search(id):

    csv_file = csv.reader(open('Cart.csv', "r"), delimiter=",")

    # loop through csv list
    for row in csv_file:
        
        if id == int(row[0]):
            return row[0],row[1],row[2]


if not os.path.exists(MODEL_NAME + '/frozen_inference_graph.pb'):
    print ('Downloading the model')
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
      file_name = os.path.basename(file.name)
      if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())
    print ('Download complete')
else:
    print ('Model already exists')

# ## Load a (frozen) Tensorflow model into memory.

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

#=================================================================================
for key, value_dict in category_index.items():
#    print(key,value_dict)
    for key1,value in value_dict.items():
        if value_dict['id'] in total_itm :
            
            P_id,P_name,P_rate=csv_search(int(value_dict['id']))
            value_dict['name']=P_name + " ===Rs." + P_rate + " /-"
            
    category_index_new[key]=value_dict
    
category_index=category_index_new
#=================================================================================

cap = cv2.VideoCapture(0)

# Running the tensorflow session
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
   ret = True
   
   cart_itm = []
   show_cart={}
   cnt = 0
   total_P=0
   while (ret):
      ret,image_np = cap.read()
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})

      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8,min_score_thresh = .5)

      objects = []
      threshold = 0.5
      
      for index, value in enumerate(classes[0]):
          object_dict = {}
          if scores[0, index] > threshold:
              object_dict[(category_index.get(value)).get('id')] = scores[0, index]
              objects.append(object_dict)

          prev_o=""
          for key, value in object_dict.items():
              b = str(key).split(" ")

              if prev_o!=b[0]:
                  
                  if show_cart=={}:
                      cnt=0
                      total_P=0
                      
                  if int(b[0]) in total_itm and cnt!=1:

                      P_id,P_name,P_rate=csv_search(int(b[0]))
                      show_cart[P_id]=(P_name,P_rate,1)
                      print(P_name + "  Product Added-------------------------------------------------")
                      total_P=0
                      for keys,value in show_cart.copy().items():
                          print(value[0] + "-------------" + str(value[1]))
                          total_P=total_P+int(value[1])
                      print("Total Value========================================================= " + str(total_P))                      
                      print("________________________________________________________________")
                      cnt=cnt+1
                  

                  elif int(b[0]) in total_itm and cnt==1:
#                      print(show_cart)
                      for keys,value in show_cart.copy().items():
                          if int(keys) == int(b[0]):
                              P_id,P_name,P_rate=csv_search(int(keys))
#                              total_P=total_P-int(P_rate)
                              del show_cart[keys]
                              print(P_name + " Deleted from cart---------------------------------------------")
                              total_P=0
                              for keys,value in show_cart.copy().items():
                                  print(value[0] + "-------------" + str(value[1]))
                                  total_P=total_P+int(value[1])

                              print("Total Value================================================= " + str(total_P))                      
                              print("________________________________________________________________")
                          else:
                              
                              P_id,P_name,P_rate=csv_search(int(b[0]))
                              print(P_name + " Product Added")
                              show_cart[P_id]=(P_name,P_rate,1)
#                              total_P=total_P+int(P_rate)
                              total_P=0
                              for keys,value in show_cart.copy().items():
                                  print(value[0] + "-------------" + str(value[1]))
                                  total_P=total_P+int(value[1])

                              print("Total Value================================================== " + str(total_P))                      
                              print("________________________________________________________________")

                     

      cv2.namedWindow(Window_Name, cv2.WND_PROP_FULLSCREEN)
      cv2.setWindowProperty(Window_Name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

      cv2.imshow(Window_Name,cv2.resize(image_np,(1280,960)))
      if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          cap.release()
          break