# -*- coding: utf-8 -*-
# @Author: fyr91
# @Date:   2019-10-22 15:05:15
# @Last Modified by:   fyr91
# @Last Modified time: 2019-10-30 11:25:26
import cv2
import numpy as np
import onnx
import time
import sys
import onnxruntime as ort
from onnx_tf.backend import prepare
base_dir = "/keras-facenet/"
sys.path.append(base_dir + '/code/')
from inception_resnet_v1 import *
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
import json
from mtcnn import MTCNN
from decimal import Decimal
def area_of(left_top, right_bottom):
    """
    Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Perform hard non-maximum-supression to filter out boxes with iou greater
    than threshold
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
        picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    """
    Select boxes that contain human faces
    Args:
        width: original image width
        height: original image height
        confidences (N, 2): confidence array
        boxes (N, 4): boxes array in corner-form
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
        boxes (k, 4): an array of boxes kept
        labels (k): an array of labels for each boxes kept
        probs (k): an array of probabilities for each boxes being in corresponding labels
    """
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
           iou_threshold=iou_threshold,
           top_k=top_k,
           )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

def crop_bb(image, detection, margin):
    (x1, y1, w, h) = detection
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
def get_model():
    model = InceptionResNetV1(classes=512)
    model.load_weights(base_dir + 'model/keras/weights/facenet_keras_weights.h5')
    return model
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

def get_embdding_vector(model, image):    
    image = np.expand_dims(image, axis = 0)
    predicted_vector = model.predict(image)
    predicted_vector = np.squeeze(predicted_vector)
    return predicted_vector
def get_detection_embedding_face(video_file):
    print('prediction')
    video_capture = cv2.VideoCapture(video_file.name)
    # video_capture = cv2.VideoCapture('C:/Users/Asus/Downloads/tmpsi2v9vs7.mp4')
    onnx_path = 'C:/Users/Asus/Downloads/ultra_light_640.onnx'
    onnx.load(onnx_path)
    print("Onnx model loaded")
    # predictor = prepare(onnx_model)
    ort_session = ort.InferenceSession(onnx_path)
    print("Session loaded")
    input_name = ort_session.get_inputs()[0].name
    
    prev_frame_time = 0
    embedding_faces = []
    model = get_model()
    i = 1
    count = 0
    if(video_capture.isOpened()== False):
        print("Error opening video stream or file")
    while video_capture.isOpened():
        # i = i + 1
        # if(i == 100):
        #     break
        ret, frame = video_capture.read()
        if frame is not None:
            h, w, _ = frame.shape
            # preprocess img acquired
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert bgr to rgb
            img = cv2.resize(img, (640, 480)) # resize
            img_mean = np.array([127, 127, 127])
            img = (img - img_mean) / 128
            img = np.transpose(img, [2, 0, 1])
            img = np.expand_dims(img, axis=0)
            img = img.astype(np.float32)
            confidences, boxes = ort_session.run(None, {input_name: img})
            boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)
            new_frame_time = time.time()
            font = cv2.FONT_HERSHEY_SIMPLEX
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)
            fps = str(fps)
            cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
            count = count + 15
            # for each frame in the video, detect the faces of the people
            for i in range(boxes.shape[0]):
                box = boxes[i, :]
                cropped_face_image = crop(frame, box)
                standardizedImage = pre_process(cropped_face_image)
                embedding_face = get_embdding_vector(model, standardizedImage)
                
                embedding_faces.append(embedding_face)
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80,18,236), 2)
                cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (80,18,236), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                text = f"face: {labels[i]}"
                cv2.putText(frame, text, (x1 + 6, y2 - 6), font, 0.5, (255, 255, 255), 1)
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, count)
        else:
            break
    video_capture.release()
    cv2.destroyAllWindows()
    return embedding_faces
def get_prediction_results(video_file, embedded_image_list, thresh_hold):
    embedding_faces = get_detection_embedding_face(video_file)
    image_embedding_set = set()
    for embedded_face in embedding_faces:
        min = 100
        for embedded_image in embedded_image_list:
            dist = np.linalg.norm(np.array(embedded_image['embedding_image'], dtype = np.double) - embedded_face)
            print(dist)
            # print(dist)
            if dist < min:
                min = dist
                embedded_image_id = embedded_image['id']
        if min < thresh_hold:
            print(embedded_image_id)
            image_embedding_set.add(embedded_image_id)
        else:
            continue   
    print('finished')
    return image_embedding_set
    