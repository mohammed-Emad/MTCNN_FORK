import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from .mtcnn2 import detect_face
import threading
from keras.models import load_model
import cv2
from pathlib import Path

######################
gpu_memory_fraction = 1.0
detect_multiple_faces = False #get one face
margin = 32
image_size = 160
######################

lib_root = Path(__file__).parent.absolute()

print("File Path:", lib_root)

class load_all(threading.Thread):
     def __init__(self):
         threading.Thread.__init__(self)
         print('Loading models')

         with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            self.sess2 = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False))
            with self.sess2.as_default():
               self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess2, os.path.join(lib_root, 'mtcnn2'))
               
         print("load_all--- done")

     
class InitFace():
     def __init__(self):
         self.models = load_all()
         self.models.start()

         #For Mtcnn
         self.minsize = 40                  # minimum size of face
         self.threshold = [ 0.6, 0.7, 0.9 ]  # three steps's threshold
         self.factor = 0.709                 # scale factor

     def DetectFace(self, img):
         bounding_boxes, points = detect_face.detect_face(img, self.minsize, self.models.pnet, self.models.rnet, self.models.onet, self.threshold, self.factor)
         return bounding_boxes
     
def Crop_Align(img,bounding_boxes):
    bounding_boxes = np.array([bounding_boxes])

    nrof_faces = bounding_boxes.shape[0]
    if nrof_faces>0:
       det = bounding_boxes[:,0:4]
       det_arr = []
       img_size = np.asarray(img.shape)[0:2]
       if nrof_faces>1:
          if detect_multiple_faces:
             for i in range(nrof_faces):
                 det_arr.append(np.squeeze(det[i]))
          else:
              bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
              img_center = img_size / 2
              offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
              offset_dist_squared = np.sum(np.power(offsets,2.0),0)
              index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
              det_arr.append(det[index,:])
       else:
           det_arr.append(np.squeeze(det))

    for i, det in enumerate(det_arr):
      det = np.squeeze(det)
      bb = np.zeros(4, dtype=np.int32)
      bb[0] = np.maximum(det[0]-margin/2, 0)
      bb[1] = np.maximum(det[1]-margin/2, 0)
      bb[2] = np.minimum(det[2]+margin/2, img_size[1])
      bb[3] = np.minimum(det[3]+margin/2, img_size[0])
      cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
      #scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
      scaled = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_AREA)
      return scaled
