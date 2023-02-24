from .facenet2 import facenet
import numpy as np
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
import sys
import math
import pickle
from sklearn.svm import SVC
import sklearn
print(sklearn.__version__)



image_size = 160
detect_multiple_faces = False
margin = 32
#--par--Classifier----#
batch_size = 100
seed = 666
 

SVM_MODEL = 'svm_model.pkl'


def Embedding(img_list ,load_m):
            # Get input and output tensors
            sess = load_m.sess
            images_placeholder = load_m.images_placeholder
            embeddings = load_m.embeddings 
            phase_train_placeholder = load_m.phase_train_placeholder 
            embedding_size = load_m.embedding_size 

            images = np.stack(img_list)

            # Run forward pass to calculate embeddings
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array = sess.run(embeddings, feed_dict=feed_dict)

            return emb_array
