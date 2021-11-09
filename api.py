import flask
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
from flask import jsonify
import re
from model import *
import tempfile
from pathlib import Path


app = flask.Flask(__name__)
app.config["DEBUG"] = True
@app.route('/', methods=['GET'])
def home():
    return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"

@app.route('/api/v1/resources/person/add', methods = ['POST'])
def add_person():
    if flask.request.files:
        image_file = flask.request.files['image']
        npimg = np.fromstring(image_file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        imageRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        embeding_vector_string = embed_image(imageRGB)
        print("result"+embeding_vector_string)
        if(embeding_vector_string == ''):
            return ''
        return embeding_vector_string
    else:
        raise Exception("File not found")
    
# @app.route('/api/v1/resources/person/recog', methods = ['POST'])
# def recog():
#     embedded_image_list = []
#     if flask.request.files:
#         image_file = flask.request.files['video']
#         identities = flask.request.form['embedded_image_list']
#         i = 0
#         for identity in json.loads(identities):
#             i = i+1
#             identityId = identity['id']
# #            images_string = re.compile("\s{1,}").split(identity['embeddedImage'])
#             images_string = identity['embeddedImage'][1:-1]
#             # images_string[0] = identity['embeddedImage'][0][1:]
#             # images_string[-1] = identity['embeddedImage'][-1][:-2]
#             print('in lan ' + str(i) + images_string)
#             images_string = images_string.strip().split()
#             em_image_array = [(float)(point) for point in images_string]
#             person = dict()
#             person['id'] = identityId
#             person['embedding_image'] = em_image_array
#             embedded_image_list.append(person)
#         npimg = np.frombuffer(image_file.read(), np.uint8)
#         img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
#         imageRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
#         embedded_image_id = get_faces(imageRGB, embedded_image_list)
#         print(embedded_image_id)
#     return str(embedded_image_id)
@app.route('/api/v1/resources/person/recog/video', methods = ['POST'])
def recog_by_video():
    embedded_image_list = []
    unique_list = list()
    if flask.request.files:
        video_file = flask.request.files['video']
        video_temp_file = tempfile.NamedTemporaryFile(suffix='.mp4')
        video_temp_file.write(video_file.read())
        identities = flask.request.form['embedded_image_list']
        embedded_image_list = get_processed_embedded_image_list_json(identities)
        image_embedding_id_set = get_faces_in_video(video_temp_file, embedded_image_list)
        unique_list = list(image_embedding_id_set)
        video_temp_file.close()
    return jsonify(results = unique_list)
    
@app.route('/api/v1/resources/person/recog/image', methods = ['POST'])
def recog_by_image():
    unique_list = list()
    if not flask.request.files:
        return "No files in the request"
    else:
        image_file = flask.request.files['image']
        print(image_file)
        npimg = np.fromstring(image_file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        imageRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        identities = flask.request.form['embedded_image_list']
        embedded_image_list = get_processed_embedded_image_list_json(identities)
        image_embedding_id_set = get_image_embeddings_id(imageRGB, embedded_image_list)
        unique_list = list(image_embedding_id_set)
        return jsonify(results = unique_list)

def get_processed_embedded_image_list_json(identities):
    embedded_image_list = []
    for identity in json.loads(identities):
            identityId = identity['id']
            images_string = identity['embeddedImage'][1:-1]
            images_string = images_string.strip().split()
            em_image_array = [(float)(point) for point in images_string]
            person = dict()
            person['id'] = identityId
            person['embedding_image'] = em_image_array
            embedded_image_list.append(person)
    return embedded_image_list
app.run()

