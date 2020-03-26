from tensorflow.keras.preprocessing import image as ig
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from multiprocessing import Process
from geopy.distance import geodesic
from datetime import datetime
from flask_cors import CORS

from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO
from PIL import Image
import numpy as np
import unidecode
import requests
import imutils
import urllib
import six
import cv2

import os
import sys
import tensorflow as tf

app = Flask(__name__)

CORS(app)

@app.before_first_request
def loadmodel():
    global Service
    Service = [False,False]
    porn_model = load_model('porn_model_premier.h5')
    app.model = porn_model

CORS(app)
def porn(image_path,i):
    global Service
    response = requests.get(image_path)
    img = Image.open(BytesIO(response.content))
                    
    img = img.resize((299,299))
    x = ig.img_to_array(img)
    x = x.reshape((1,) + x.shape)
    x= x/255
    porn_model = app.model            
    porn_preds = porn_model.predict(x)
    #Es imagen inapropiada?
    if porn_preds[0][0] > 0.5:
        Service[i] = False #NO
    else:
        Service[i] = True #SI


CORS(app)
  
@app.route('/', methods=['POST','GET'])
def location_time_validate():
    global data
    global json_respuesta
    
    data = request.json
    forb_words = ['matar','asesinar','violar','acuchillar','secuestrar',
                  'linchar','chingar','chingate','joder','jodete','coger',
                  'follar','puto','puta','malnacido',
                  'pito','polla','pendeja','pendejo','pinche','mierda',
                  'concha','chingatumadre','descuartizar']
    text = data['text']
    
    target_text_list = forb_words
    text = text.lower()
    text = unidecode.unidecode(text)
    final_text = len([word for word in target_text_list if(word in text)])>=1    

    if not final_text:
        if data['url'] != '':
            image_path = data['url']
            porn(image_path,0)
        else:
            json_respuesta = {'Location':True,'Time':True,'Service':True,'Porn':Service[0] or Service[1]}
            return jsonify(json_respuesta)

        if data['url2'] != '':
            image_path2 = data['url2']
            porn(image_path2,1)
        else:
            json_respuesta = {'Location':True,'Time':True,'Service':not (Service[0] or Service[1]),'Porn':Service[0] or Service[1]}
            return jsonify(json_respuesta)

        if True in Service:
            json_respuesta = {'Location':True,'Time':True,'Service':False,'Porn':Service[0] or Service[1]}
            return jsonify(json_respuesta)
        else:
            json_respuesta = {'Location':True,'Time':True,'Service':True,'Porn':Service[0] or Service[1]}
            return jsonify(json_respuesta)
    
    else:
        json_respuesta = {'Location':True,'Time':True,'Service':False,'Porn':Service[0] or Service[1]}
        return jsonify(json_respuesta)



if __name__ == '__main__':
    
    
    app.run(host='127.0.0.1', port=8080, debug=False)

    
    