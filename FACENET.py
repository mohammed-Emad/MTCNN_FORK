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

def EmbeddingList(load_m ,data_dir = "data_algin"):
            print("args.seed??",seed)

            sess = load_m.sess
            images_placeholder = load_m.images_placeholder
            embeddings = load_m.embeddings 
            phase_train_placeholder = load_m.phase_train_placeholder 
            embedding_size = load_m.embedding_size 
            
            np.random.seed(seed=seed)


            dataset = facenet.get_dataset(data_dir)

            print("dataset type",type(dataset))
            print(dataset)
            # Check that there are at least one training image per class

            print("dataset=",len(dataset))
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            
            print('Number of classes new: %d' % len(dataset))
            print('Number of images new: %d' % len(paths))
            
            # Load the model
            print('Loading feature extraction model')

            # Get input and output tensors
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*batch_size
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)

            print(type(emb_array),"~",type(labels))

            # Create a list of class names
            class_names = [ cls.name.replace('_', ' ') for cls in dataset]
            return emb_array, labels, class_names

def Embedding(nrof_faces,paths_batch,load_m):

            # Get input and output tensors
            sess = load_m.sess
            images_placeholder = load_m.images_placeholder
            embeddings = load_m.embeddings 
            phase_train_placeholder = load_m.phase_train_placeholder 
            embedding_size = load_m.embedding_size 
            
            # Run forward pass to calculate embeddings

            images = [facenet.prewhiten(cv2.cvtColor(nrof_faces, cv2.COLOR_BGR2RGB))]
            nrof_images = len(images)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)


            return emb_array

def Embedding_old(img_list ,load_m):
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
