from tensorflow.keras.preprocessing import image as ig
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from multiprocessing import Process
from geopy.distance import geodesic
from datetime import datetime, timedelta
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
import logging
import sqlalchemy
import sqlalchemy as db
import os
import sys
import tensorflow as tf

user = os.environ.get("DB_USER")
password = os.environ.get("DB_PASS")
database = os.environ.get("DB_NAME")
cloud_sql_connection_name = os.environ.get("CLOUD_SQL_CONNECTION_NAME")

app = Flask(__name__)

#logger = logging.getLogger()

engine = db.create_engine(f'mysql+pymysql://{user}:{password}@/{database}?unix_socket=/cloudsql/{cloud_sql_connection_name}')

#CORS(app)
@app.before_first_request
def loadmodel():
    global Service
    global connection
    global result_data

    connection = engine.connect()
    metadata = db.MetaData()

    #base de datos de texto
    result_data = db.Table('result_data', metadata,
                db.Column('From Service',db.String(255), nullable=False),
                db.Column('Date',db.DateTime),
                db.Column('User Id',db.TEXT, nullable=False),
                db.Column('User Name',db.TEXT, nullable=False),
                db.Column('Mission Id',db.TEXT, nullable=False),
                db.Column('Mission Name',db.TEXT, nullable=False),
                db.Column('User Latitude',db.FLOAT(10,8), nullable=False),
                db.Column('User Longitude',db.FLOAT(10,8), nullable=False),
                db.Column('Mission Latitude',db.FLOAT(10,8), nullable=False),
                db.Column('Mission Longitude',db.FLOAT(10,8), nullable=False),
                db.Column('Start Date Mission',db.String(255), nullable=False),
                db.Column('End Date Mission',db.String(255), nullable=False),
                db.Column('Target Time',db.INT),
                db.Column('Radio',db.INT),
                db.Column('URL', db.TEXT),
                db.Column('Text', db.TEXT),
                db.Column('Location',db.BOOLEAN),
                db.Column('Time',db.BOOLEAN),
                db.Column('Porn',db.BOOLEAN),
                db.Column('Scene',db.BOOLEAN),
                db.Column('Extra',db.BOOLEAN),
                db.Column('Object',db.BOOLEAN),
                db.Column('Service',db.BOOLEAN),
                )
    metadata.create_all(engine) #Creates Table

    Service = [False,False]
    porn_model = load_model('Porn_model+_alpha_TF114.h5')
    app.model = porn_model

#CORS(app)
def porn(image_path,i):
    global Service
        
    response = requests.get(image_path)
    img = Image.open(BytesIO(response.content))
    
    if img.format == 'PNG':
        img = img.resize((299,299))
        img.save('/tmp/0003.png')

        j = '/tmp/0003.png'
        img = cv2.imread(j)
        cv2.imwrite(j[:-3] + 'jpg', img)

        img = Image.open('/tmp/0003.jpg')
        img = img.resize((299,299))
        img = img.save('/tmp/001.JPG')  #
        img = ig.load_img('/tmp/001.JPG', target_size = (299,299))  #
        x = ig.img_to_array(img)
        x = x.reshape((1,) + x.shape)
        x = x/255
        
    else:
        img = img.resize((299,299))
        img = img.save('/tmp/001.JPG')  #
        img = ig.load_img('/tmp/001.JPG', target_size = (299,299))  #
        x = ig.img_to_array(img)
        x = x.reshape((1,) + x.shape)
        x = x/255

    porn_model = app.model            
    porn_preds = porn_model.predict(x)
    #print(porn_preds[0][0])
    #Es imagen inapropiada?
    if porn_preds[0][0] > 0.5:
        Service[i] = False #NO#Safe
    else:
        Service[i] = True #SI#Porn


#CORS(app)
  
@app.route('/', methods=['POST','GET'])
def location_time_validate():
    global data
    global json_respuesta
    global Service

    Service = [False,False]

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


#CORS(app)
@app.after_request
def mysql_con(response):
    #Query a Cloud SQL
    try:
        data_a_cloud_sql = [{'From Service':'explicit','Date':(datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
                            'User Id':data['id'],'User Name':data['name'],
                            'Mission Id':data['id_mission'],'Mission Name':data['mission_name'],
                            'User Latitude':data['Location_latitude'],'User Longitude':data['Location_longitude'],
                            'Mission Latitude':data['Location_mission_latitude'],'Mission Longitude':data['Location_mission_longitude'],
                            'Start Date Mission':data['Start_Date_mission'],'End Date Mission':data['End_Date_mission'],
                            'Target Time':data['Target_time_mission'],'Radio':data['Location_mission_radio'],
                            'URL':data['url'],'Text':data['text'],
                            'Location':json_respuesta['Location'],'Time':json_respuesta['Time'],
                            'Porn':json_respuesta['Porn'],
                            'Service':json_respuesta['Service']}]
        query_cloud = db.insert(result_data)
        ResultProxy = connection.execute(query_cloud,data_a_cloud_sql)
    except Exception as e:
        print(e)
        pass
    
    return response
        
if __name__ == '__main__':
    
    app.run(host='127.0.0.1', port=8080, debug=False)