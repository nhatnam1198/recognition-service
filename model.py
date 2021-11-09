import os
import re
from mtcnn import mtcnn
import numpy as np
import tensorflow as tf
import sys
base_dir = "/keras-facenet/"
sys.path.append(base_dir + '/code/')
from inception_resnet_v1 import *
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
import json
from mtcnn import MTCNN
from pretrainModel import get_prediction_results
from decimal import Decimal
def get_model():
    model = InceptionResNetV1(classes=512)
    model.load_weights(base_dir + 'model/keras/weights/facenet_keras_weights.h5')
    return model
def read_image(file):
  img = cv2.imread(file)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img

def crop_bb(image, detection, margin):
    x1, y1, w, h = detection['box']
    x1 -= margin
    y1 -= margin
    w += 2*margin
    h += 2*margin
    if x1 < 0:
        w += x1
        x1 = 0
    if y1 < 0:
        h += y1
        y1 = 0
    return image[y1:y1+h, x1:x1+w]

# def crop(mtcnn, img):
#   detect_results = mtcnn.detect_faces(img)
#   if(len(detect_results) <= 0):
#     raise Exception('The system can not detect your face in the image')
#   else:
#     det = mtcnn.detect_faces(img)[0]
#     margin = int(0.1 * img.shape[0])
#     ret = crop_bb(img, det, margin)
#   return ret
def crop(img, detection):
    margin = int(0.1 * img.shape[0])
    ret = crop_bb(img, detection, margin)
    return ret
def pre_process(face, required_size=(160, 160)):
    ret = cv2.resize(face, required_size)
    #ret = cv2.cvtColor(ret, cv2.COLOR_BGR2RGB)
    ret = ret.astype('double')
    # standardize pixel values across channels (global)
    mean, std = ret.mean(), ret.std()
    ret = (ret - mean) / std

    return ret

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_embdding_vector(model, image):    
    image = np.expand_dims(image, axis = 0)
    predicted_vector = model.predict(image)
    predicted_vector = np.squeeze(predicted_vector)
    return predicted_vector
    

def embed_image(image):
    model = get_model()
    mtcnn = MTCNN()
    detection = mtcnn.detect_faces(image)[0]
    if(detection == False):
        return ''
    face = crop(image, detection)
    cropped_image = pre_process(face)
    embedding_image = get_embdding_vector(model, cropped_image)

    # dictionary = dict()
    # pair = {'predicted_vector': embedding_image}
    # dictionary.update(pair)
    return str(embedding_image).replace('\n', '')  

def get_faces_in_video(video_file, embedded_image_list, thresh_hold = 1.24):
    return get_prediction_results(video_file, embedded_image_list, thresh_hold)

def get_image_embeddings_id(image_file, embedded_image_list, thresh_hold = 1.15):
    model = get_model()
    mtcnn = MTCNN()
    embedding_faces = []
    detection_results = mtcnn.detect_faces(image_file)
    for detection in detection_results:
        cropped_face_image = crop(image_file, detection)
        standardizedImage = pre_process(cropped_face_image)
        embedding_face = get_embdding_vector(model, standardizedImage)
        embedding_faces.append(embedding_face)

    embedded_image_id_set = set()
    for embedded_face in embedding_faces:
        min = 100
        for embedded_image in embedded_image_list:
            dist = np.linalg.norm(np.array(embedded_image['embedding_image'], dtype = np.double) - embedded_face)
            print(dist)
            if dist < min:
                min = dist
                embedded_image_id = embedded_image['id']
            if min < thresh_hold:
                embedded_image_id_set.add(embedded_image_id)
                print(embedded_image_id_set)
            else:
                continue
    return embedded_image_id_set