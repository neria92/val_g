#Manuel Neria #Push #Nova
#LICENCES

# PlacesCNN to predict the scene category, attribute, and class activation map in a single pass
# by Bolei Zhou, sep 2, 2017

""" @article{zhou2017places,
   title={Places: A 10 million Image Database for Scene Recognition},
   author={Zhou, Bolei and Lapedriza, Agata and Khosla, Aditya and Oliva, Aude and Torralba, Antonio},
   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
   year={2017},
   publisher={IEEE}
 }"""

#from imageai.Prediction.Custom import CustomImagePrediction
from imutils.object_detection import non_max_suppression
from tensorflow.keras.preprocessing import image as ig
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template
from multiprocessing import Process
from geopy.distance import geodesic
from datetime import datetime, timedelta, date
from flask_cors import CORS
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO
import numpy as np
import pygsheets
import unidecode
import requests
import imutils
import base64
import urllib
import six
import cv2
import re
import os
import sys
import dlib
import json
import ffmpeg
import folium
import pandas as pd
import face_recognition
import sqlalchemy as db
import tensorflow as tf
import matplotlib.pyplot as plt
from folium.plugins import MarkerCluster
from folium.plugins import LocateControl
from random import randint
from Utils import label_map_util
from Utils import visualization_utils as vis_util
from PIL import Image, ImageDraw
from imutils import paths
from math import floor, atan2, degrees, sin, cos, pi
from scipy import ndimage
from scipy.spatial.distance import euclidean, pdist, squareform
from google.cloud import storage, vision
from werkzeug.utils import secure_filename
import yagmail
from exif import Image as exif_image
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
from pyzbar.pyzbar import decode
import googlemaps



user = os.environ.get("DB_USER")
password = os.environ.get("DB_PASS")
database = os.environ.get("DB_NAME")
database_misions = os.environ.get("DB_NAME_MISIONS")
cloud_sql_connection_name = os.environ.get("CLOUD_SQL_CONNECTION_NAME")

account_e = os.environ.get("ACC")
password_e = os.environ.get("PASS")

app = Flask(__name__)

engine = db.create_engine(f'mysql+pymysql://{user}:{password}@/{database}?unix_socket=/cloudsql/{cloud_sql_connection_name}')
engine_misions = db.create_engine(f'mysql+pymysql://{user}:{password}@/{database_misions}?unix_socket=/cloudsql/{cloud_sql_connection_name}')

gmaps = googlemaps.Client(key=os.environ.get("GMAPS_KEY"))


# hacky way to deal with the Pytorch 1.0 update
def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module

def load_labels():
    # prepare all the labels
    # scene category relevant
    file_name_category = 'Utils/categories_places365.txt'
    classes = []
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    file_name_IO = 'Utils/IO_places365.txt'
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    # scene attribute relevant
    file_name_attribute = 'Utils/labels_sunattribute.txt'
    with open(file_name_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]
    file_name_W = 'Utils/W_sceneattribute_wideresnet18.npy'
    W_attribute = np.load(file_name_W)

    return classes, labels_IO, labels_attribute, W_attribute

def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

def returnTF():
    # load the image transformer
    transfmr = trn.Compose([
        trn.Resize((224,224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transfmr

def load_scene_model():
    # this model has a last conv feature map as 14x14

    model_file = 'Utils/wideresnet18_places365.pth.tar'

    import Utils.wideresnet as wideresnet
    model = wideresnet.resnet18(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    
    # hacky way to deal with the upgraded batchnorm2D and avgpool layers...
    for i, (name, module) in enumerate(model._modules.items()):
        module = recursion_change_bn(model)
    model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)
    
    model.eval()

    # hook the feature extractor
    features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)
    return model

@app.before_first_request
def loadmodel():
    global Service
    global masked_url
    global result_data
    global missions_taifelds
    global missions_santory
    global missions_taifelds2
    global missions_hidrosina
    global missions_hidrosina_auditoria_strikes
    global missions_hidrosina_servicio_strikes
    global tabla_summary
    global tabla_preguntas
    global missions_taifelds_disfruta
    global missions_covid
    global my_faces
    global nombres
    global url2json0
    global receivers
    global body_covid
    global body_taifelds
    global body_santory
    global body_hidrosina
    global body_rechazada
    global body_hidrosina_alerta
    global body_hidrosina_alerta_auditoria_baja
    global body_hidrosina_alerta_servicio_bajo
    global body_hidrosina_alerta_qr
    global body_hidrosina_alerta_precios
    global body_hidrosina_alerta_ticket
    global body_hidrosina_alerta_gerente
    global body_hidrosina_alerta_100
    global body_hidrosina_alerta_calificacion_baja
    global body_taifelds_disfruta
    global transfmr

    receivers = ['gotchudl@gmail.com','back.ght.001@gmail.com','medel@cimat.mx']
    body_taifelds = "Hay una nueva misión de Taifelds para validar en https://gchgame.web.app/ con número de tienda: "
    body_santory = "Hay una nueva misión de Santory para validar en https://gchgame.web.app/ con número de tienda: "
    body_taifelds_disfruta = "Hay una nueva misión de Disfruta y Gana para validar en https://gchgame.web.app/ con número de tienda: "
    body_covid = "Hay una nueva misión de Hospital Covid para validar en https://gchgame.web.app/ con número de Id de Hospital: "
    body_hidrosina = "Hay una nueva misión de Hidrosina para validar en https://gchgame.web.app/ con número de Id: "
    body_hidrosina_alerta = "Hay una alerta de seguridad sanitaria por falta de cubrebocas en la estación con identificador HD "
    body_hidrosina_alerta_auditoria_baja = "Hay una alerta de repetidas calificaciones BAJAS de auditoria en la estación con identificador HD "
    body_hidrosina_alerta_servicio_bajo = "Hay una alerta de repetidas calificaciones BAJAS de servicio en la estación con identificador HD "
    body_hidrosina_alerta_precios = "Hay una alerta de letrero de precios apagados en la estación con identificador HD "
    body_hidrosina_alerta_ticket = "Hay una alerta de ticket incorrecto en la estación con identificador HD "
    body_hidrosina_alerta_gerente = "Hay una alerta de auscencia de Gerente de Estación en la estación con identificador HD "
    body_hidrosina_alerta_qr = "Hay una alerta de código QR ilegible en la estación con identificador HD "
    body_hidrosina_alerta_100 = "Hay una alerta de calificación poco confiable en la estación con identificador HD "
    body_hidrosina_alerta_calificacion_baja = "Hay una alerta de baja calificación en la estación con identificador HD "
    body_rechazada = "ALERTA hay una captura de pago rechazada, no aplica ID "

    metadata = db.MetaData()
    result_data = db.Table('result_data', metadata, autoload=True, autoload_with=engine)
    missions_taifelds = db.Table('taifelds', metadata, autoload=True, autoload_with=engine_misions)
    missions_taifelds2 = db.Table('taifelds2', metadata, autoload=True, autoload_with=engine_misions)
    missions_hidrosina = db.Table('hidrosina', metadata, autoload=True, autoload_with=engine_misions)
    missions_hidrosina_auditoria_strikes = db.Table('hidrosina_auditoria_strikes', metadata, autoload=True, autoload_with=engine_misions)
    missions_hidrosina_servicio_strikes = db.Table('hidrosina_servicio_strikes', metadata, autoload=True, autoload_with=engine_misions)
    tabla_summary = db.Table('hidrosina_scores_estacion', metadata, autoload=True, autoload_with=engine_misions)
    tabla_preguntas = db.Table('hidrosina_preguntas_estacion', metadata, autoload=True, autoload_with=engine_misions)
    missions_taifelds_disfruta = db.Table('taifelds_disfruta', metadata, autoload=True, autoload_with=engine_misions)
    missions_covid = db.Table('covid', metadata, autoload=True, autoload_with=engine_misions)
    missions_santory = db.Table('santory', metadata, autoload=True, autoload_with=engine_misions)

    # load the model
    scene_model_365 = load_scene_model()

    # load the transformer scene-detector
    transfmr = returnTF() # image transformer

    my_faces = []
    images = ['images/ManuelNeria.jpeg','images/Alex.jpg','images/Alma.jpg','images/Robert.jpg',
              'images/olegario.jpg','images/jacobo1.jpg','images/bella.jpg','images/albertob.jpg']
    nombres = ['manuel_neria','alex','alma','robert','olegario','jacob','bella','albertob']
    for i in images:
        face = face_recognition.load_image_file(i)
        my_faces.append(face_recognition.face_encodings(face)[0])

    Service = ['0','1','2','3']
    masked_url = ['']
    url2json0 = ['']


    MINIMUM_CONFIDENCE = 0 #0.4

    #PATH_TO_LABELS = 'label_map.pbtxt'

    #label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    #categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=sys.maxsize, use_display_name=True)
    CATEGORY_INDEX = 0 #label_map_util.create_category_index(categories)

    #PATH_TO_CKPT = 'frozen_inference_graph.pb'

    # Load model into memory
    #print('Loading model...')
    #detection_graph = tf.Graph()
    #with detection_graph.as_default():
    #    od_graph_def = tf.GraphDef()
    #    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid: #.io
    #        serialized_graph = fid.read()
    #        od_graph_def.ParseFromString(serialized_graph)
    #        tf.import_graph_def(od_graph_def, name='')
    
    #print('detecting...')
    #detection_graph.as_default()
    sess = 0 #tf.Session(graph=detection_graph)
    
    image_tensor = 0 #detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = 0 #detection_graph.get_tensor_by_name('detection_boxes:0')
    
    detection_scores = 0 #detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = 0 #detection_graph.get_tensor_by_name('detection_classes:0')
    
    num_detections = 0 #detection_graph.get_tensor_by_name('num_detections:0')


    model_incres = load_model('INCResV2_model_premier.h5')
    porn_model = load_model('Porn_model+_alphaV2.h5')
    yolo_model = load_model('YOLOV3_model.h5')
    screen_model = load_model('Screen-live_model_super_99-99_7.h5')

    prediction = 0 #CustomImagePrediction()
    #prediction.setModelTypeAsResNet()
    #prediction.setModelPath("model_ex-092_acc-0.963542.h5")
    #prediction.setJsonPath("model_class.json")
    #prediction.loadModel(num_objects=2)
    

    # load our serialized face detector from disk
    protoPath = 'deploy.prototxt'
    modelPath = 'res10_300x300_ssd_iter_140000.caffemodel'

    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    
    app.model = [sess,image_tensor,detection_boxes,
                 detection_scores,detection_classes,num_detections,
                 CATEGORY_INDEX,MINIMUM_CONFIDENCE,model_incres,
                 detector,prediction,porn_model,
                 yolo_model,Service,screen_model,
                 scene_model_365]

def url_to_image2(url):
	"""download the image, convert it to a NumPy array, and then read
	it into OpenCV format"""
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
	"""return the image"""
	return image

def _get_storage_client():
    return storage.Client()

def _safe_filename(filename):
    """
    Generates a safe filename that is unlikely to collide with existing objects
    in Google Cloud Storage.
    ``filename.ext`` is transformed into ``filename-YYYY-MM-DD-HHMMSS.ext``
    """
    filename = secure_filename(filename)
    date = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d-%H%M%S")
    basename, extension = filename.rsplit('.', 1)
    return "{0}-{1}.{2}".format(basename, date, extension)

def upload_file(file_stream, filename):
    """
    Uploads a file to a given Cloud Storage bucket and returns the public url
    to the new object.
    """
    filename = _safe_filename(filename)

    client = _get_storage_client()
    bucket = client.bucket('gchgame.appspot.com')
    blob = bucket.blob('images/' + datetime.utcnow().strftime("%Y-%m-%d")+ '/' + filename)
    
    blob.upload_from_filename(
        file_stream
        )

    url = blob.public_url

    if isinstance(url, six.binary_type):
        url = url.decode('utf-8')

    return url

def upload_image_file(file):
    """
    Upload the user-uploaded file to Google Cloud Storage and retrieve its
    publicly-accessible URL.
    """
    if not file:
        return None
    
    public_url = upload_file(
        file,
        file
    )
    #print("Uploaded file %s as %s.", file.filename, public_url)

    return public_url

def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length
    
def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        
        self.objness = objness
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score

def _sigmoid(x):
	return 1. / (1. + np.exp(-x))

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
	grid_h, grid_w = netout.shape[:2]
	nb_box = 3
	netout = netout.reshape((grid_h, grid_w, nb_box, -1))
	nb_class = netout.shape[-1] - 5
	boxes = []
	netout[..., :2]  = _sigmoid(netout[..., :2])
	netout[..., 4:]  = _sigmoid(netout[..., 4:])
	netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
	netout[..., 5:] *= netout[..., 5:] > obj_thresh
 
	for i in range(grid_h*grid_w):
		row = i / grid_w
		col = i % grid_w
		for b in range(nb_box):
			# 4th element is objectness score
			objectness = netout[int(row)][int(col)][b][4]
			if(objectness.all() <= obj_thresh): continue
			# first 4 elements are x, y, w, and h
			x, y, w, h = netout[int(row)][int(col)][b][:4]
			x = (col + x) / grid_w # center position, unit: image width
			y = (row + y) / grid_h # center position, unit: image height
			w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
			h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
			# last elements are class probabilities
			classes = netout[int(row)][col][b][5:]
			box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
			boxes.append(box)
	return boxes
 
def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    if (float(net_w)/image_w) < (float(net_h)/image_h):
        new_w = net_w
        new_h = (image_h*net_w)/image_w
    else:
        new_h = net_w
        new_w = (image_w*net_h)/image_h
        
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

def _interval_overlap(interval_a, interval_b):
	x1, x2 = interval_a
	x3, x4 = interval_b
	if x3 < x1:
		if x4 < x1:
			return 0
		else:
			return min(x2,x4) - x1
	else:
		if x2 < x3:
			 return 0
		else:
			return min(x2,x4) - x3

def bbox_iou(box1, box2):
	intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
	intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
	intersect = intersect_w * intersect_h
	w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
	w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
	union = w1*h1 + w2*h2 - intersect
	return float(intersect) / union

def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
        
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0: continue

            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0
                    
def draw_boxes(filename, v_boxes, v_labels, v_scores):
	# load the image
	data = pyplot.imread(filename)
	# plot the image
	pyplot.imshow(data)
	# get the context for drawing boxes
	ax = pyplot.gca()
	# plot each box
	for i in range(len(v_boxes)):
		box = v_boxes[i]
		# get coordinates
		y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
		# calculate width and height of the box
		width, height = x2 - x1, y2 - y1
		# create the shape
		rect = Rectangle((x1, y1), width, height, fill=False, color='white')
		# draw the box
		ax.add_patch(rect)
		# draw text and score in top left corner
		label = "%s (%.3f)" % (v_labels[i], v_scores[i])
		pyplot.text(x1, y1, label, color='white')
	# show the plot
	pyplot.show()

def get_boxes(boxes, labels, thresh):
	v_boxes, v_labels, v_scores = list(), list(), list()
	# enumerate all boxes
	for box in boxes:
		# enumerate all possible labels
		for i in range(len(labels)):
			# check if the threshold for this label is high enough
			if box.classes[i] > thresh:
				v_boxes.append(box)
				v_labels.append(labels[i])
				v_scores.append(box.classes[i]*100)
				# don't break, many labels may trigger for one box
	return v_boxes, v_labels, v_scores

def load_image_pixels(filename, shape):
    # load the image to get its shape
    
    response = requests.get(filename)
    image = Image.open(BytesIO(response.content))
    width, height = image.size                        
    image = image.resize(shape)
    
    
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height

def url_to_image(url):
	urllib.request.urlretrieve(url,'01.jpg')
	return '01.jpg'

def load_image_file_0(file, mode='RGB'):
    """
    Loads an image file (.jpg, .png, etc) into a numpy array

    :param file: image file name or file object to load
    :param mode: format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L' (black and white) are supported.
    :return: image contents as numpy array
    """
    im = Image.open(file)
    
    if mode:
        im = im.convert(mode)
    im = np.array(im)
    al, an, ca = im.shape
    if al > 1400 or an > 1400:
        if al > an:
            im = cv2.resize(im,(700,1400))
        else:
            im = cv2.resize(im,(1400,700))
    return im

def load_image_file_270(file, mode='RGB'):
    """
    Loads an image file (.jpg, .png, etc) into a numpy array

    :param file: image file name or file object to load
    :param mode: format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L' (black and white) are supported.
    :return: image contents as numpy array
    """
    im = Image.open(file).transpose(Image.ROTATE_270)
    
    if mode:
        im = im.convert(mode)
    im = np.array(im)
    al, an, ca = im.shape
    if al > 1400 or an > 1400:
        if al > an:
            im = cv2.resize(im,(700,1400))
        else:
            im = cv2.resize(im,(1400,700))
    return im

def load_image_file_180(file, mode='RGB'):
    """
    Loads an image file (.jpg, .png, etc) into a numpy array

    :param file: image file name or file object to load
    :param mode: format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L' (black and white) are supported.
    :return: image contents as numpy array
    """
    im = Image.open(file).transpose(Image.ROTATE_180)
    
    if mode:
        im = im.convert(mode)
    im = np.array(im)
    al, an, ca = im.shape
    if al > 1400 or an > 1400:
        if al > an:
            im = cv2.resize(im,(700,1400))
        else:
            im = cv2.resize(im,(1400,700))
    return im

def load_image_file_90(file, mode='RGB'):
    """
    Loads an image file (.jpg, .png, etc) into a numpy array

    :param file: image file name or file object to load
    :param mode: format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L' (black and white) are supported.
    :return: image contents as numpy array
    """
    im = Image.open(file).transpose(Image.ROTATE_90)
    
    if mode:
        im = im.convert(mode)
    im = np.array(im)
    al, an, ca = im.shape
    if al > 1400 or an > 1400:
        if al > an:
            im = cv2.resize(im,(700,1400))
        else:
            im = cv2.resize(im,(1400,700))
    return im

def orientation_fix_function(image_path):
  orientation_fix_functions_dict = {1:load_image_file_0,2:load_image_file_0,
                3:load_image_file_180,4:load_image_file_180,
                5:load_image_file_270,6:load_image_file_270,
                7:load_image_file_90,8:load_image_file_90}
  try:
    image = url_to_image(image_path)
    with open(image, 'rb') as image_file:
        my_image = exif_image(image_file)
        if my_image.has_exif:
          image = orientation_fix_functions_dict[my_image.orientation.value](image)
          plt.imsave('orientation_fixed_image.jpg',image)
          return upload_image_file('orientation_fixed_image.jpg')
        else:
          return image_path
  except Exception as e:
    return image_path

def check_rotation(path_video_file):
    # this returns meta-data of the video file in form of a dictionary
    try:
        meta_dict = ffmpeg.probe(path_video_file)
    except Exception as e:
        print(e)
        print('No meta-video')
        meta_dict = {'streams':[{'tags':{'rotate':0}}]}
  
    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotateCode = None
    try:
        if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
            rotateCode = cv2.ROTATE_90_CLOCKWISE
        elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
            rotateCode = cv2.ROTATE_180
        elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
            rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE
    except Exception:
      pass
    print(rotateCode)
    return rotateCode
    
def correct_rotation(frame, rotateCode):
    print('corrigiendo')  
    return cv2.rotate(frame, rotateCode) 

def video_to_thumbnail_url(video_path,threshold=50,threshold2=200,thumb_width=400,thumb_height=600,limit=5):
    if video_path == '':
        return ''
    try:
        urllib.request.urlretrieve(video_path,'gotchu_video.mp4')
    except Exception:
        return ''

    try:
        video_path = 'gotchu_video.mp4'
        # check if video requires rotation
        rotateCode = check_rotation(video_path)

        vcap = cv2.VideoCapture(video_path)
        response, image = vcap.read()
        i = 0
        while (image.mean() < threshold or image.mean() > threshold2) and response and i < limit:
            i += 1
            response, image = vcap.read()

        if rotateCode != None:
            image = correct_rotation(image, rotateCode)
            print('video corregido')

        al, an, ca = image.shape
        if al >= an:
            image = cv2.resize(image, (thumb_width, thumb_height), 0, 0, cv2.INTER_LINEAR)
            #overlay = cv2.imread('images/play_r.png')
            foreground = cv2.imread('images/play_r.png') ###NEW (Quitando el fondo difuminado)
            alpha = foreground ###NEW (Quitando el fondo difuminado)
            alpha = cv2.bitwise_not(alpha) ###NEW (Quitando el fondo difuminado)
        if an > al:
            image = cv2.resize(image, (thumb_height, thumb_width), 0, 0, cv2.INTER_LINEAR)
            #overlay = cv2.imread('images/play_ra.png')
            foreground = cv2.imread('images/play_ra.png') ###NEW (Quitando el fondo difuminado)
            alpha = foreground ###NEW (Quitando el fondo difuminado)
            alpha = cv2.bitwise_not(alpha) ###NEW (Quitando el fondo difuminado)

        background = image  ###NEW (Quitando el fondo difuminado)
        foreground = cv2.resize(foreground, (background.shape[1], background.shape[0]), 0, 0, cv2.INTER_LINEAR) ###NEW (Quitando el fondo difuminado)
        alpha = cv2.resize(alpha, (background.shape[1], background.shape[0]), 0, 0, cv2.INTER_LINEAR) ###NEW (Quitando el fondo difuminado)
        foreground = foreground.astype(float) ###NEW (Quitando el fondo difuminado)
        background = background.astype(float) ###NEW (Quitando el fondo difuminado)
        alpha = alpha.astype(float)/255 ###NEW (Quitando el fondo difuminado)
        foreground = cv2.multiply(alpha, foreground) ###NEW (Quitando el fondo difuminado)
        background = cv2.multiply(1.0 - alpha, background) ###NEW (Quitando el fondo difuminado)
        added_image = cv2.add(foreground, background) ###NEW (Quitando el fondo difuminado)

        # background = image
        # overlay = cv2.resize(overlay, (background.shape[1], background.shape[0]), 0, 0, cv2.INTER_LINEAR)
        # added_image = cv2.addWeighted(background,0.6,overlay,0.2,0)

        cv2.imwrite('thumbnail.jpg', added_image)
        return upload_image_file('thumbnail.jpg')
    except Exception as e:
        print(e)
        return ''
  
def detect_objects(image_path,validar,names,labels):
    if validar in names:
        return 'na', 'na'
        sess = app.model[0]
        image_tensor = app.model[1]
        detection_boxes = app.model[2]
        
        detection_scores = app.model[3]
        detection_classes = app.model[4]

        num_detections = app.model[5]
        
        CATEGORY_INDEX = app.model[6]
        MINIMUM_CONFIDENCE = app.model[7]

        size = 600 
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content))
        
        image = image.resize((size,size)) 
        (im_width, im_height) = image.size

        image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})
        
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            CATEGORY_INDEX,
            min_score_thresh=MINIMUM_CONFIDENCE,
            use_normalized_coordinates=True,
            line_thickness=3)
        fig = plt.figure()
        fig.set_size_inches(16, 9)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        
        #plt.imshow(image_np, aspect = 'auto')
        #plt.savefig('{}'.format(image_path), dpi = 62)
        
        #plt.close(fig)
        respns = [CATEGORY_INDEX.get(value) for index, value in enumerate(classes[0]) if scores[0,index] > MINIMUM_CONFIDENCE]
        if respns == []:
            return 'na', 'na' #respns
        else:
            respns = respns[0]['name']
            return respns, respns

    elif validar in labels:
        
        anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
        class_threshold = 0.6
        input_w, input_h = 416, 416

        image, image_w, image_h = load_image_pixels(image_path, (input_w, input_h))
        
        yolo_model = app.model[12]
        yhat = yolo_model.predict(image)

        boxes = list()
        for i in range(len(yhat)):
            
            boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
        

        correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)

        do_nms(boxes, 0.5)

        v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)

        if validar in v_labels:
            return validar, ', '.join(v_labels)
        else:
            return ', '.join(v_labels), ', '.join(v_labels)
    else:
        return validar, validar

def masked_face(image,end=False):
    if end:
        #return####DEV
        image = face_recognition.load_image_file(image)
    #return False####DEV
    faces = face_recognition.face_locations(image,2)

    marks_list = {'images/00.png':[1.6,405,105,160,1,60],'images/01.png':[1.6,405,105,160,1,60],'images/02.png':[1.6,405,105,160,1,60],#indice 5 es la punta izquiera del sombrero
                      'images/03.png':[1.6,405,105,160,1,60],'images/04.png':[1.6,405,105,160,1,60],'images/05.png':[1.6,405,105,160,1,60],
                      'images/10.png':[1.95,500,150,170,1,110],'images/11.png':[1.95,500,150,170,1,110],'images/12.png':[1.95,500,150,170,1,110],
                      'images/13.png':[1.95,500,150,170,1,110],'images/14.png':[1.95,500,150,170,1,110],'images/15.png':[1.95,500,150,170,1,110],
                      'images/20.png':[1.73,500,143,190,1.486,95],'images/21.png':[1.73,500,143,190,1.486,95],'images/22.png':[1.73,500,143,190,1.486,95],#26/06/20 CAMBIÉ 130 POR 143
                      'images/23.png':[1.73,500,143,190,1.486,95],'images/24.png':[1.73,500,143,190,1.486,95],'images/25.png':[1.73,500,143,190,1.486,95]}


    if faces == []:

        if end:
            return
        
        url2json = data['url']
        Service[2] = False
        masked_url[0] = url2json
        return False
        
    for i in range(len(faces)):
        top, right, bottom, left = faces[i]

        k = randint(0,2)  
        l = randint(0,5)  
        name_img = 'images/' + str(k) + str(l) + '.png'

        mask_nsize = floor((right-left)*marks_list[name_img][0])
        mask_nsize_factor = mask_nsize/marks_list[name_img][1]

        mask = cv2.imread(name_img,-1)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2RGBA)
        mask = cv2.resize(mask,(mask_nsize,floor(mask_nsize*marks_list[name_img][4]))) 
        w, h, c = mask.shape

        face = cv2.cvtColor(image,cv2.COLOR_BGR2BGRA)
        x = left
        y = top
        z = floor(mask_nsize_factor*marks_list[name_img][3])
        z1 = floor(mask_nsize_factor*marks_list[name_img][5])
        for i in range(w):
            for j in range(h):
                if mask[i,j][3] != 0 and mask[i,j][3] == 255:
                    try:
                        if y+i-z < 0 or x+j-z1 < 0:
                            continue
                        face[y+i-z,x+j-z1] = mask[i,j]
                    except IndexError as e:
                        pass

    plt.imsave('face.jpg',face)

    url2json = upload_image_file('face.jpg')
    Service[2] = True
    masked_url[0] = url2json
    return True

def imagen_final(image_path):
    if image_path == '':
        return ''

    image_origin = url_to_image(image_path)
    imagen_con_logo = load_image_file_0(image_origin)
    
    alto, ancho, p = imagen_con_logo.shape

    name_logo = 'images/GotchuLogo.png'
    logo = face_recognition.load_image_file(name_logo)
    logo = cv2.resize(logo,(150,150))
    w, h, c = logo.shape

    x = ancho - w
    y = alto - h

    for i in range(w):
        for j in range(h):
            if list(logo[i,j]) != [255,255,255] and list(logo[i,j])[1]<160:
                try:
                    if y+i < 0 or x+j < 0:
                        continue
                    
                    imagen_con_logo[y+i,x+j] = logo[i,j]
                except IndexError as e:
                    pass

    plt.imsave('imagen_con_logo.jpg',imagen_con_logo)
    url2json = upload_image_file('imagen_con_logo.jpg')
    return url2json

def detect_human(image_path,validar,extras):
    if validar == 'persona':
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        resp = urllib.request.urlopen(image_path)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image_o = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = imutils.resize(image_o, width=min(600, image_o.shape[1]))

        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
        padding=(8, 8), scale=1.05)

        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        # draw the final bounding boxes
        #for (xA, yA, xB, yB) in pick:
        #cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

        
        if len(pick) == 0:
            
            detector = app.model[9]
            # load the image, resize it to have a width of 600 pixels (while
            # maintaining the aspect ratio), and then grab the image dimensions
            
            
            
            image = imutils.resize(image_o, width=600)
            (h, w) = image.shape[:2]
            
            # construct a blob from the image
            imageBlob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)
            
            # apply OpenCV's deep learning-based face detector to localize
            # faces in the input image
            detector.setInput(imageBlob)
            detections = detector.forward()
            
            # loop over the detections
            boxes = []
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections
                if confidence > 0.5:
                    # compute the (x, y)-coordinates of the bounding box for the
                    # face
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    boxes.append(box)
            if len(boxes)==0:
                Service[2] = False
            else:
                Service[2] = True
        else:
            Service[2] = True

    if validar == 'cara':

        resp = urllib.request.urlopen(image_path)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image_o = cv2.imdecode(image, cv2.IMREAD_COLOR)

        detector = app.model[9]
        # load the image, resize it to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image dimensions
        
        image = imutils.resize(image_o, width=600)
        (h, w) = image.shape[:2]
        
        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)
        
        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()
        
        # loop over the detections
        boxes = []
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for the
                # face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                boxes.append(box)
        if len(boxes)==0:
            Service[2] = False
        else:
            Service[2] = True

    if validar == 'selfie':
        image_origin = url_to_image(image_path)
        url2json0[0] = upload_image_file(image_origin)
        image = load_image_file_0(image_origin)
        face_landmarks_list = face_recognition.face_landmarks(image)

        marks_list = {'images/00.png':[1.6,405,105,160,1,60],'images/01.png':[1.6,405,105,160,1,60],'images/02.png':[1.6,405,105,160,1,60],#indice 5 es la punta izquiera del sombrero
                      'images/03.png':[1.6,405,105,160,1,60],'images/04.png':[1.6,405,105,160,1,60],'images/05.png':[1.6,405,105,160,1,60],
                      'images/10.png':[1.95,500,150,170,1,110],'images/11.png':[1.95,500,150,170,1,110],'images/12.png':[1.95,500,150,170,1,110],
                      'images/13.png':[1.95,500,150,170,1,110],'images/14.png':[1.95,500,150,170,1,110],'images/15.png':[1.95,500,150,170,1,110],
                      'images/20.png':[1.73,500,143,190,1.486,95],'images/21.png':[1.73,500,143,190,1.486,95],'images/22.png':[1.73,500,143,190,1.486,95],#26/06/20 CAMBIÉ 130 POR 143
                      'images/23.png':[1.73,500,143,190,1.486,95],'images/24.png':[1.73,500,143,190,1.486,95],'images/25.png':[1.73,500,143,190,1.486,95]}


        if face_landmarks_list == []:
            if masked_face(image):
                return
            
            url2json = data['url']
            Service[2] = False
            masked_url[0] = url2json
            return

        face = image

        for i in range(len(face_landmarks_list)):

            x_coords = []
            y_coords = []
            facial_centroids = {}

            k = randint(0,2)
            l = randint(0,5)
            name_img = 'images/' + str(k) + str(l) + '.png'

            for facial_feature in face_landmarks_list[i].keys():
                facial_centroids[facial_feature] = centeroidnp(np.asarray(face_landmarks_list[i][facial_feature]))
                x_coords.extend([x for x, y in face_landmarks_list[i][facial_feature]])
                y_coords.extend([y for x, y in face_landmarks_list[i][facial_feature]])
                
            xychin = [(x, y) for x, y in face_landmarks_list[i]['chin']]
            chin_distance = floor(max(pdist(np.asarray(xychin))))
            eye_distance = euclidean(np.asarray(facial_centroids['left_eye']),np.asarray(facial_centroids['right_eye']))
            mask_nsize = floor(chin_distance*marks_list[name_img][0]) #1.6 es calculado
            mask_nsize_factor = mask_nsize/marks_list[name_img][1] #shape original mask 

            x = min(face_landmarks_list[i]['left_eyebrow'])[0]
            y = min(face_landmarks_list[i]['left_eyebrow'])[1]

            dx = facial_centroids['right_eye'][0]-facial_centroids['left_eye'][0]
            dy = facial_centroids['right_eye'][1]-facial_centroids['left_eye'][1]
            myradians = atan2(dy,dx)
            mydegrees = -degrees(myradians)
            
            mask = cv2.imread(name_img,-1)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2RGBA)
            mask = cv2.resize(mask,(mask_nsize,floor(mask_nsize*marks_list[name_img][4]))) 

            ####Código muy especial e infernal
            angle_rad = np.deg2rad(mydegrees)
            c, s = np.cos(angle_rad), np.sin(angle_rad)
            rot_matrix = np.array([[c, s],
                                    [-s, c]])
            out_bounds = rot_matrix @ [[0],
                                    [mask_nsize*marks_list[name_img][4]]]
            estrella = out_bounds[0][0]
            lens_point_out = rot_matrix @ [[mask_nsize_factor*marks_list[name_img][2]],
                                        [mask_nsize_factor*marks_list[name_img][3]]]
            if mydegrees <= 0:
                x160 = lens_point_out[0][0] - estrella
                y90 = lens_point_out[1][0]
            else:
                x160 = lens_point_out[0][0] 
                y90 = lens_point_out[1][0] + estrella
            ####Fin del código muy especial e infernal

            mask = ndimage.rotate(mask, mydegrees,order=0)
            w, h, c = mask.shape

            face = cv2.cvtColor(face,cv2.COLOR_BGR2BGRA)
            z = floor(y90)
            z1 = floor(x160)

            for i in range(w):
                for j in range(h):
                    if mask[i,j][3] != 0 and mask[i,j][3] == 255:
                        try:
                            if y+i-z < 0 or x+j-z1 < 0:
                                continue
                            face[y+i-z,x+j-z1] = mask[i,j]
                        except IndexError:
                            pass

        plt.imsave('face.jpg',face)

        url2json = upload_image_file('face.jpg')
        Service[2] = True
        masked_url[0] = url2json
        masked_face('face.jpg',end=True)
        return

    if validar == 'na':
        Service[2] = True

def detect_scene(image_path,validar,class_names,class_names_param,x,scene_classes,labels_IO,labels_attribute,W_attribute):
    global features_blobs
    features_blobs = []
    if validar == 'na':
        Service[1] = True

    elif validar in scene_classes or validar in labels_attribute or validar == 'indoor':
        model = app.model[15]
        # load the test image
        urllib.request.urlretrieve(image_path, '00000003.jpg')
        img = Image.open('00000003.jpg')
        input_img = V(transfmr(img).unsqueeze(0))

        # forward pass
        logit = model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.numpy()
        idx = idx.numpy()

        # output the IO prediction
        if validar == 'indoor':
            io_image = np.mean(labels_IO[idx[:10]]) # vote for the indoor or outdoor
            if io_image < 0.5:
                Service[1] = True #--TYPE OF ENVIRONMENT: indoor
            else:
                Service[1] = False #--TYPE OF ENVIRONMENT: outdoor
        
        else:
            # output the prediction of scene category
            scenes_top_list = []
            for i in range(0, 5):
                scenes_top_list.append(scene_classes[idx[i]])

            # output the scene attributes
            responses_attribute = W_attribute.dot(features_blobs[1])
            idx_a = np.argsort(responses_attribute)
            labels_attribute_top_list = [labels_attribute[idx_a[i]] for i in range(-1,-10,-1)]

            if validar in scenes_top_list or validar in labels_attribute_top_list:
                Service[1] = True
            else:
                Service[1] = False

    else:
        model = app.model[8]
        global indx
        global thres
        global ths
        global relat
        
        mot = validar
        prediction = app.model[10]
        indx = class_names.index(mot)
        thres = class_names_param[mot]['threshold']
        ths = class_names_param[mot]['threshold2']
        relat = class_names_param[mot]['related'] 
        
        if validar == 'anaqueles_vacios':
            urllib.request.urlretrieve(image_path, "00000001.jpg")
            img = "00000001.jpg"
            predictions, probabilities = prediction.predictImage(img, result_count=2)
            indx = predictions.index('anaquel')
            pred = [probabilities]
            relat = []
        else:
            pred = model.predict(x)
        T = []
        if pred[0][indx]>thres:
            Service[1] = True
        else:
            if relat != []:
                for r in relat:
                    idx = class_names.index(r)
                
                    if pred[0][idx]>ths:
                        T.append(ths)
                        Service[1] = True               
                        break
            if T == []:
                Service[1] = False

def porn(x):
    porn_model = app.model[11]            
    porn_preds = porn_model.predict(x)
    #Es imagen inapropiada?
    if porn_preds[0][0] > 0.05:
        Service[0] = True #NO#Safe
    else:
        Service[0] = False #SI#Porn

def porn_explicit(image_path,i):
    global Service
        
    response = requests.get(image_path)
    img = Image.open(BytesIO(response.content))
    
    if img.format == 'PNG':
        img = img.resize((299,299))
        img.save('images/0003p.png')

        j = 'images/0003p.png'
        img = cv2.imread(j)
        cv2.imwrite(j[:-3] + 'jpg', img)

        img = Image.open('images/0003p.jpg')
        img = img.resize((299,299))
        img = img.save('images/001p.JPG')  #
        img = ig.load_img('images/001p.JPG', target_size = (299,299))  #
        x = ig.img_to_array(img)
        x = x.reshape((1,) + x.shape)
        x = x/255
        
    else:
        img = img.resize((299,299))
        img = img.save('images/001p.JPG')  #
        img = ig.load_img('images/001p.JPG', target_size = (299,299))  #
        x = ig.img_to_array(img)
        x = x.reshape((1,) + x.shape)
        x = x/255

    porn_model = app.model[11]       
    porn_preds = porn_model.predict(x)
    #print(porn_preds[0][0])
    #Es imagen inapropiada?
    if porn_preds[0][0] > 0.5:
        Service[i] = False #NO#Safe
    else:
        Service[i] = True #SI#Porn

def screen(y):
    screen_model = app.model[14]            
    screen_preds = screen_model.predict(y)
    #Es imagen tomada de una pantalla?
    if screen_preds[0][0] > 0.5:
        Service[3] = False #SI
    else:
        Service[3] = True #NO

def screen_explicit(image_path):
    response = requests.get(image_path)
    img = Image.open(BytesIO(response.content))
                    
    img = img.resize((299,299))
    img = img.save('images/001.JPG')  #
    img = ig.load_img('images/001.JPG', target_size = (299,299))  #
    y = ig.img_to_array(img)
    y = y.reshape((1,) + y.shape)
    y = y/255
    screen_model = app.model[14]            
    screen_preds = screen_model.predict(y)
    #Es imagen tomada de una pantalla?
    if screen_preds[0][0] > 0.5:
        Service[3] = True #SI
    else:
        Service[3] = False #NO

def liveness(image_path):
    if image_path == '':
        return True
    response = requests.get(image_path)
    img = Image.open(BytesIO(response.content))
                    
    img = img.resize((299,299))
    img = img.save('images/005.JPG')  #
    img = ig.load_img('images/005.JPG', target_size = (299,299))  #
    y = ig.img_to_array(img)
    y = y.reshape((1,) + y.shape)
    y = y/255
    screen_model = app.model[14]            
    screen_preds = screen_model.predict(y)
    #Es imagen tomada de una pantalla?
    if screen_preds[0][0] > 0.5:
        return False #SI
    else:
        return True #NO

def face_recog(val_face,image_path):

    
    url = image_path
  
    # my_face_encoding now contains a universal 'encoding' of my facial features that can be compared to any other picture of a face!
    image = url_to_image(url)
    unknown_picture = load_image_file_0(image)
    unknown_face_encodings = face_recognition.face_encodings(unknown_picture)
    
    # Now we can see the two face encodings are of the same person with `compare_faces`!
    found_faces = []
    for face in unknown_face_encodings:
        results = face_recognition.compare_faces(my_faces, face)
        found_faces.extend([nombres[i] for i, e in enumerate(results) if e == True])
    print(found_faces)

    if val_face in found_faces:
        return True
    else:
        return False

def detect_text_uri(uri):
    """Detects text in the file located in Google Cloud Storage or on the Web.
    """
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = uri

    response = client.text_detection(image=image)
    texts = response.text_annotations
    text_list = []
    for text in texts:
      #print(text.description)
      text_list.append(text.description)
    full_text = ''.join(text_list)
    return full_text

def detect_text_local(path):
    """"Detects text in the file."""
    import io
    client = vision.ImageAnnotatorClient()
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    text_list = []
    for text in texts:
      #print(text.description)
      text_list.append(text.description)
    full_text = ''.join(text_list)
    return full_text

def find_fecha_hora(full_text):
    fecha_list = re.findall('\d{4}[/\-\s]\d\d[/\-\s]\d\d',full_text)
    if len(fecha_list) == 0:
      fecha = '1970-01-01'
    else: fecha = fecha_list[0]

    hora_list = re.findall('\d\d[:\.]\d\d',full_text)
    if len(hora_list) == 0:
        hora = '00:00'
    else:
        if int(hora_list[0][3]) > 5:
            hora = hora_list[0][:3] + '5' + hora_list[0][-1]
        else: hora = hora_list[0]
    
    fecha = fecha.replace(' ','-')
    fecha = fecha.replace('/','-')
    hora = hora.replace('.',':')
    return fecha, hora

def fecha_hora_2_timestamp(fecha,hora):
    fecha_str = fecha + ' ' + '00:00'
    hora_str = '1970-01-01' + ' ' + hora
    element_fecha = datetime.strptime(fecha_str,"%Y-%m-%d %H:%M")
    element_hora = datetime.strptime(hora_str,"%Y-%m-%d %H:%M")
    fecha_timestamp = datetime.timestamp(element_fecha)
    hora_timestamp = datetime.timestamp(element_hora)
    return fecha_timestamp, hora_timestamp

def detect_text_uri_demo(uri):
    """Detects text in the file located in Google Cloud Storage or on the Web.
    """
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = uri

    response = client.text_detection(image=image)
    texts = response.text_annotations
    
    texto = texts[0].description
    texto = texto.split('\n')
    texto_final = {}
    for i, t in enumerate(texto):
      texto_final[i] = t
    return texto_final

def find_codigo_factura(full_text):
    codigo = re.findall('(?<=CODIGO FACTURA:).*',full_text)
    if len(codigo) == 0:
      return '0'
    return codigo[0]

def find_no_ticket(full_text):
    codigo = re.findall('(?<=N\..{8})\d*',full_text)
    if len(codigo) == 0:
      return '0'
    codigo = codigo[0].split(' ')
    return codigo[0]

def message_general_service(json_respuesta):
    location = json_respuesta['Location']
    time = json_respuesta['Time']
    porn = json_respuesta['Porn']
    service = json_respuesta['Service']

    if service:
        return '¡Felicidades Agente! Has cumplido tu misión satisfactoriamente'
    elif porn:
        return 'Tu captura tiene contenido explícito, si crees que esto es un error no te preocupes lo solucionaremos a la brevedad y tu captura será aceptada'
    elif not location:
        return 'No te encuentras en el lugar correcto para realizar tu misión, si crees que esto es un error no te preocupes lo solucionaremos a la brevedad y tu captura será aceptada'
    elif not time:
        return 'Esta misión ha expirado o aún no es momento de completarla, si crees que esto es un error no te preocupes lo solucionaremos a la brevedad y tu captura será aceptada'
    else:
        return 'Tu captura no cumple con los requerimientos de la misión, revisa la descripción de la misión. Si crees que esto es un error no te preocupes lo solucionaremos a la brevedad y tu captura será aceptada'

def message_pay_service(json_respuesta):
    location = json_respuesta['Location']
    time = json_respuesta['Time']
    porn = json_respuesta['Porn']
    service = json_respuesta['Service']

    if service:
        return '¡Felicidades Agente! Has cumplido tu misión satisfactoriamente'
    elif porn:
        return 'Tu captura tiene contenido explícito, si crees que esto es un error no te preocupes lo solucionaremos a la brevedad y tu captura será aceptada'
    elif not location:
        return 'No te encuentras en el lugar correcto para realizar tu misión u otro agente ya hizo está misión aquí, si crees que esto es un error no te preocupes lo solucionaremos a la brevedad y tu captura será aceptada'
    elif not time:
        return 'Esta misión ha expirado o aún no es momento de completarla, si crees que esto es un error no te preocupes lo solucionaremos a la brevedad y tu captura será aceptada'
    else:
        return 'Tu captura no cumple con los requerimientos de la misión, revisa la descripción de la misión. Si crees que esto es un error no te preocupes lo solucionaremos a la brevedad y tu captura será aceptada'

def url_qr_2_text(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
    except:
        return 'No se pudo leer nombre de ticket, falla de imagen'
    try:
        code = decode(img)
        str_data = code[0].data
        data_decoded = str_data.decode()
        data64 = re.split('&|#',data_decoded)[-1]
        message_bytes = base64.b64decode(data64)
        message = message_bytes.decode('ascii')
        return message
    except:
        return 'No se pudo leer nombre de ticket'

def enviar_mail(url,headers,payload):
  requests.request("POST", url, headers=headers, data = json.dumps(payload))

def Categorizacion_esta(df):
    df.loc[((df.no_luce_limpia.isnull()) & (df.no_bien_iluminada.isnull()) & (df.No_letreros_precios.isnull())),"Que_le_faltaba_aLa_estacion"]="Tiene los tres elementos"
    df.loc[((df.no_luce_limpia.isnull()) & (df.no_bien_iluminada.isnull()) & (df.No_letreros_precios.notnull())),"Que_le_faltaba_aLa_estacion"]="No tiene precios iluminados"

    df.loc[((df.no_luce_limpia.isnull()) & (df.no_bien_iluminada.notnull()) & (df.No_letreros_precios.isnull())),"Que_le_faltaba_aLa_estacion"]="No está bien iluminada"
    df.loc[((df.no_luce_limpia.isnull()) & (df.no_bien_iluminada.notnull()) & (df.No_letreros_precios.notnull())),"Que_le_faltaba_aLa_estacion"]="No letrero de precios y no bien iluminada"

    df.loc[((df.no_luce_limpia.notnull()) & (df.no_bien_iluminada.isnull()) & (df.No_letreros_precios.isnull())),"Que_le_faltaba_aLa_estacion"]="No está limpia"
    df.loc[((df.no_luce_limpia.notnull()) & (df.no_bien_iluminada.isnull()) & (df.No_letreros_precios.notnull())),"Que_le_faltaba_aLa_estacion"]="No está limpia y no precios iluminados"

    df.loc[((df.no_luce_limpia.notnull()) & (df.no_bien_iluminada.notnull()) & (df.No_letreros_precios.isnull())),"Que_le_faltaba_aLa_estacion"]="No está limpia, no bien iluminada"
    df.loc[((df.no_luce_limpia.notnull()) & (df.no_bien_iluminada.notnull()) & (df.No_letreros_precios.notnull())),"Que_le_faltaba_aLa_estacion"]="No tiene los tres elementos"
    return df

def Categorizacion_que_no_llevan(df):
  df.loc[((df.uniforme_limpio.isnull()) & (df.cubrebocas.isnull()) & (df.lentes_protectores.isnull())),"que_no_llevaba_despachador"]="Llevan lentes,cubrebocas y uniforme limpio"
  df.loc[((df.uniforme_limpio.isnull()) & (df.cubrebocas.isnull()) & (df.lentes_protectores.notnull())),"que_no_llevaba_despachador"]="No_llevan: Lentes protectores"

  df.loc[((df.uniforme_limpio.isnull()) & (df.cubrebocas.notnull()) & (df.lentes_protectores.isnull())),"que_no_llevaba_despachador"]="No llevan: Cubrebocas"
  df.loc[((df.uniforme_limpio.isnull()) & (df.cubrebocas.notnull()) & (df.lentes_protectores.notnull())),"que_no_llevaba_despachador"]="No traen: Cubrebocas y lentes protectores"

  df.loc[((df.uniforme_limpio.notnull()) & (df.cubrebocas.isnull()) & (df.lentes_protectores.isnull())),"que_no_llevaba_despachador"]="No tienen:  Uniforme limpio"
  df.loc[((df.uniforme_limpio.notnull()) & (df.cubrebocas.isnull()) & (df.lentes_protectores.notnull())),"que_no_llevaba_despachador"]="No tienen: Uniforme limpio y lentes protectores"

  df.loc[((df.uniforme_limpio.notnull()) & (df.cubrebocas.notnull()) & (df.lentes_protectores.isnull())),"que_no_llevaba_despachador"]="No tienen: Uniforme limpio y cubrebocas"
  df.loc[((df.uniforme_limpio.notnull()) & (df.cubrebocas.notnull()) & (df.lentes_protectores.notnull())),"que_no_llevaba_despachador"]="No tienen: Ninguno de los tres elementos"
  return df

def Categorizacion_Cantidad_tipo_forma(df):
  df.loc[((df.cantidad_combustible.isnull()) & (df.Tipo_combustible.isnull()) & (df.Forma_pago.isnull())),"Que_no_pregunto"]="Pregunto las tres opciones: Cantidad, tipo y forma"
  df.loc[((df.cantidad_combustible.isnull()) & (df.Tipo_combustible.isnull()) & (df.Forma_pago.notnull())),"Que_no_pregunto"]="No preguntó Forma de pago"
  df.loc[((df.cantidad_combustible.isnull()) & (df.Tipo_combustible.notnull()) & (df.Forma_pago.isnull())),"Que_no_pregunto"]="No preguntó Tipo de combustible"
  df.loc[((df.cantidad_combustible.isnull()) & (df.Tipo_combustible.notnull()) & (df.Forma_pago.notnull())),"Que_no_pregunto"]="No preguntó Tipo de combustible y forma de pago"
  df.loc[((df.cantidad_combustible.notnull()) & (df.Tipo_combustible.isnull()) & (df.Forma_pago.isnull())),"Que_no_pregunto"]="No preguntó Cantidad de combustible"
  df.loc[((df.cantidad_combustible.notnull()) & (df.Tipo_combustible.isnull()) & (df.Forma_pago.notnull())),"Que_no_pregunto"]="No preguntó Cantidad de combustible y forma de pago"
  df.loc[((df.cantidad_combustible.notnull()) & (df.Tipo_combustible.notnull()) & (df.Forma_pago.isnull())),"Que_no_pregunto"]="No preguntó Cantidad y tipo de combustible"
  df.loc[((df.cantidad_combustible.notnull()) & (df.Tipo_combustible.notnull()) & (df.Forma_pago.notnull())),"Que_no_pregunto"]="No preguntó :Cantidad, tipo de combustible y forma de pago"
  return df

def Categorizacion_paarabrisa_prdo(df):
  df.loc[((df.prod_periferico.isnull()) & (df.Limpieza_parabrisas.isnull())),"que_no_te_ofrecio"]="Ofreció las dos opciones, limpieza y productos"
  df.loc[((df.prod_periferico.isnull()) & (df.Limpieza_parabrisas.notnull()) ),"que_no_te_ofrecio"]="No ofrecio Limpieza de parabrisas"
  df.loc[((df.prod_periferico.notnull()) & (df.Limpieza_parabrisas.isnull())),"que_no_te_ofrecio"]="No ofrecio Producto periférico"
  df.loc[((df.prod_periferico.notnull()) & (df.Limpieza_parabrisas.notnull())),"que_no_te_ofrecio"]="No ofrecio Limpieza de parabrisas y Producto periférico"
  return df

@app.route('/', methods=['POST'])
def location_time_validate():
    global data
    global json_respuesta
    global Service
    global obj
    global from_service
    global det
    global validar1
    global validar2
    global validar3
    global validar4
    global validar5
    global validar6
    global detected_obj
    global masked_url
    global url2json0
    global video_path

    data = request.json
    data['url'] = orientation_fix_function(data['url'])
    try:
        video_path = data['Url_Video']
    except KeyError as e:
        video_path = ''
        print(e)
    url2json0 = ['']
    masked_url = [data['url']]
    det = 'na'
    detected_obj = 'na'
    obj = False
    from_service = 'premier'

    
    gettime = request.args.get('time')
    getloc = request.args.get('loc')

    if gettime == None and getloc == None:
        time = True
        loc = True
    elif gettime == 'y' and getloc == 'y':
        time = True
        loc = True
    elif gettime == 'n' and getloc == 'y':
        time = False
        loc = True
    elif gettime == 'y' and getloc == 'n':
        time = True
        loc = False
    elif gettime == 'n' and getloc == 'n':
        time = False
        loc = False

    user_pos = (data['Location_latitude'],data['Location_longitude']) 
    mission_point_pos = (data['Location_mission_latitude'],data['Location_mission_longitude'])
    
    if (geodesic(user_pos, mission_point_pos).meters <= data['Location_mission_radio'] or not loc):
      
        start_date = datetime.strptime(data['Start_Date_mission'], '%Y-%m-%d %H:%M:%S.%f')
        end_date = datetime.strptime(data['End_Date_mission'], '%Y-%m-%d %H:%M:%S.%f')
        user_time = (end_date - start_date).total_seconds()
        mission_target_time = data['Target_time_mission']
        
        if (user_time<=mission_target_time or not time):
            text = data['text']
            target_text = request.args.get('target_text')
            if target_text == None:
                final_text = True
            else:
                target_text_list = target_text.split('_')
                text = text.lower()
                text = unidecode.unidecode(text)
                final_text = len([word for word in target_text_list if(word in text)])==len(target_text_list)
                
            if final_text:
                image_path = data['url']
                if image_path == '' or image_path == 'None' or image_path == None:
                    validar1 = 'na'
                    validar2 = 'na'
                    validar3 = 'na'
                    validar4 = 'na'
                    validar5 = 'na'
                    validar6 = 'na'
                    Service = [False,False,False,True]
                    json_respuesta = {'Location':True,'Time':True,'Service':True,'Porn':False,'Url_themask':'','url_thumbnail':video_to_thumbnail_url(video_path),'msg':''}
                    mission_message = message_general_service(json_respuesta)
                    json_respuesta['msg'] = mission_message
                    return jsonify(json_respuesta)
                response = requests.get(image_path)
                img = Image.open(BytesIO(response.content))
                                
                img = img.resize((299,299))
                x = ig.img_to_array(img)
                x = x.reshape((1,) + x.shape)
                x=x/255

                img = img.save('images/001.JPG')  #
                img = ig.load_img('images/001.JPG', target_size = (299,299))  #
                y = ig.img_to_array(img)
                y = y.reshape((1,) + y.shape)
                y = y/255

                Service = app.model[13]
                T = []
                scene_classes, labels_IO, labels_attribute, W_attribute = load_labels()
                class_names = ['agua_calles','anaqueles_vacios','banqueta','calle_oscura','calles',
                                'flood','forest','incendio','grietas','highway',
                                'overflowed_river','park','beach','smoke','volcano']
                class_names_param = {'agua_calles':{'threshold':0.50,'related':[],'threshold2':0.90},
                                    'anaqueles_vacios':{'threshold':0.95,'related':[],'threshold2':0.90},
                                    'banqueta':{'threshold':0.50,'related':['calles'],'threshold2':0.90},
                                    'calle_oscura':{'threshold':0.98,'related':[],'threshold2':0.90},
                                    'calles':{'threshold':0.50,'related':['banqueta','agua_calles','highway'],'threshold2':0.50},
                                    'flood':{'threshold':0.50,'related':['overflowed_river'],'threshold2':0.50},
                                    'forest':{'threshold':0.80,'related':['park'],'threshold2':0.50},
                                    'incendio':{'threshold':0.80,'related':['smoke'],'threshold2':0.80},
                                    'grietas':{'threshold':0.60,'related':[],'threshold2':0.90},
                                    'highway':{'threshold':0.50,'related':[],'threshold2':0.90},
                                    'overflowed_river':{'threshold':0.60,'related':['flood','highway'],'threshold2':0.90},
                                    'park':{'threshold':0.60,'related':['forest'],'threshold2':0.50},
                                    'beach':{'threshold':0.60,'related':[],'threshold2':0.90},
                                    'smoke':{'threshold':0.50,'related':['incendio','volcano'],'threshold2':0.90},
                                    'volcano':{'threshold':0.85,'related':['incendio','smoke'],'threshold2':0.60}}
                objects = ['basura','colilla','morena','pan','prd','pri','ppolitico']
                ppoliticos = ['morena','pan','prd','pri']
                extras = ['persona','selfie','cara']
                labels = ["person", "bicycle", "car", "motorbike", "aeroplane",
                            "bus", "train", "truck","boat", "traffic light", "fire hydrant",
                            "stop sign", "parking meter", "bench",
                            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
                            "zebra", "giraffe","backpack", "umbrella", "handbag", 
                            "tie", "suitcase", "frisbee", "skis", "snowboard",
                            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
                            "surfboard","tennis racket", "bottle", "wine glass", "cup", 
                            "fork", "knife", "spoon", "bowl", "banana","apple", "sandwich", 
                            "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                            "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet",
                            "tvmonitor", "laptop", "mouse","remote", "keyboard", "cell phone",
                            "microwave", "oven", "toaster", "sink", "refrigerator","book", "clock",
                            "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

                

                validar1 = request.args.get('validar_escena')
                validar2 = request.args.get('validar_objeto')
                validar2 = validar2.replace('.',' ')
                validar3 = request.args.get('validar_extra')
                validar4 = request.args.get('o_validar_escena')
                validar5 = request.args.get('o_validar_objeto')
                validar6 = request.args.get('o_validar_extra')
                liveness_flag = request.args.get('liveness')
                try:
                    validar5 = validar5.replace('.',' ')
                except Exception:
                    validar5 = 'none'
                    validar4 = 'none'
                    validar6 = 'none'
                    pass
                
                #FACE... START
                val_face = request.args.get('val_face')

                if val_face != None:
                    face_rec = face_recog(val_face,image_path)
                    if face_rec == True:
                        Service = [False,False,False,False]
                        from_service = 'face recognition'
                        json_respuesta = {'Location':True,'Time':True,'Service':True,'Porn':False,'Url_themask':'','url_thumbnail':video_to_thumbnail_url(video_path),'msg':''}
                        mission_message = message_general_service(json_respuesta)
                        json_respuesta['msg'] = mission_message
                        return jsonify(json_respuesta)
                    else:
                        Service = [False,False,False,False]
                        from_service = 'face recognition'
                        json_respuesta = {'Location':True,'Time':True,'Service':False,'Porn':False,'Url_themask':'','url_thumbnail':video_to_thumbnail_url(video_path),'msg':''}
                        mission_message = message_general_service(json_respuesta)
                        json_respuesta['msg'] = mission_message
                        return jsonify(json_respuesta)
                else:
                    pass
                #FACE... END

                if validar1 in class_names or validar1 == 'na' or validar1 in scene_classes or validar1 in labels_attribute or validar1 == 'indoor':
                    if validar2 in objects or validar2 in labels or validar2 == 'na':
                        if validar3 in extras or validar3 == 'na':
                            
                            p1 = Process(target=detect_scene(image_path,validar1,class_names,class_names_param,x,scene_classes,labels_IO,labels_attribute,W_attribute))
                            p1.start()
                            p2 = Process(target=detect_human(image_path,validar3,extras))
                            p2.start()
                            p3 = Process(target=porn(y))
                            p3.start()
                            p4 = Process(target=screen(y))
                            p4.start()
                            p1.join()
                            p2.join()
                            p3.join()
                            p4.join()
                            p1.terminate
                            p3.terminate
                            p4.terminate
                            p2.terminate
                            if liveness_flag == 'no':
                                Service[3] = True #NO
                            
                            if False in Service:
                                if validar4 in class_names or validar4 == 'na' or validar4 in scene_classes or validar4 in labels_attribute or validar4 == 'indoor':
                                    if validar5 in objects or validar5 in labels or validar5 == 'na':
                                        if validar6 in extras or validar6 == 'na':
                                            
                                            p1 = Process(target=detect_scene(image_path,validar4,class_names,class_names_param,x,scene_classes,labels_IO,labels_attribute,W_attribute))
                                            p1.start()
                                            p2 = Process(target=detect_human(image_path,validar6,extras))
                                            p2.start()
                                            p3 = Process(target=porn(y))
                                            p3.start()
                                            p4 = Process(target=screen(y))
                                            p4.start()
                                            p1.join()
                                            p2.join()
                                            p3.join()
                                            p4.join()
                                            p4.terminate
                                            p3.terminate
                                            p1.terminate
                                            p2.terminate
                                            if liveness_flag == 'no':
                                                Service[3] = True #NO
                                            
                                            if False in Service:
                                                json_respuesta = {'Location':True,'Time':True,'Service':False,'Porn':not Service[0],'Url_themask':'','url_thumbnail':'','msg':''}
                                                mission_message = message_general_service(json_respuesta)
                                                json_respuesta['msg'] = mission_message
                                                return jsonify(json_respuesta)
                                                
                                            else:
                                                det, detected_obj = detect_objects(image_path,validar5,objects,labels)
                                                obj = det == validar5 or det in ppoliticos
                                                json_respuesta = {'Location':True,'Time':True,'Service':det == validar5 or det in ppoliticos,'Porn':False,'Url_themask':imagen_final(masked_url[0]),'url_thumbnail':video_to_thumbnail_url(video_path),'msg':''}
                                                mission_message = message_general_service(json_respuesta)
                                                json_respuesta['msg'] = mission_message
                                                return jsonify(json_respuesta)

                                        else:
                                            json_respuesta = {'Location':True,'Time':True,'Service':False,'Porn':not Service[0],'Url_themask':'','url_thumbnail':'','msg':''}
                                            mission_message = message_general_service(json_respuesta)
                                            json_respuesta['msg'] = mission_message
                                            return jsonify(json_respuesta)

                                    else:
                                        json_respuesta = {'Location':True,'Time':True,'Service':False,'Porn':not Service[0],'Url_themask':'','url_thumbnail':'','msg':''}
                                        mission_message = message_general_service(json_respuesta)
                                        json_respuesta['msg'] = mission_message
                                        return jsonify(json_respuesta)

                                else:
                                    json_respuesta = {'Location':True,'Time':True,'Service':False,'Porn':not Service[0],'Url_themask':'','url_thumbnail':'','msg':''}
                                    mission_message = message_general_service(json_respuesta)
                                    json_respuesta['msg'] = mission_message
                                    return jsonify(json_respuesta)

                                
                            else:
                                det, detected_obj = detect_objects(image_path,validar2,objects,labels)
                                if det == validar2 or det in ppoliticos:
                                    obj = True
                                    json_respuesta = {'Location':True,'Time':True,'Service':True,'Porn':False,'Url_themask':imagen_final(masked_url[0]),'url_thumbnail':video_to_thumbnail_url(video_path),'msg':''}
                                    mission_message = message_general_service(json_respuesta)
                                    json_respuesta['msg'] = mission_message
                                    return jsonify(json_respuesta)
                                else:
                                    if validar4 in class_names or validar4 == 'na' or validar4 in scene_classes or validar4 in labels_attribute or validar4 == 'indoor':
                                        if validar5 in objects or validar5 in labels or validar5 == 'na':
                                            if validar6 in extras or validar6 == 'na':
                                                
                                                p1 = Process(target=detect_scene(image_path,validar4,class_names,class_names_param,x,scene_classes,labels_IO,labels_attribute,W_attribute))
                                                p1.start()
                                                p2 = Process(target=detect_human(image_path,validar6,extras))
                                                p2.start()
                                                p3 = Process(target=porn(y))
                                                p3.start()
                                                p4 = Process(target=screen(y))
                                                p4.start()
                                                p1.join()
                                                p2.join()
                                                p3.join()
                                                p4.join()
                                                p4.terminate
                                                p3.terminate
                                                p1.terminate
                                                p2.terminate
                                                if liveness_flag == 'no':
                                                    Service[3] = True #NO
                                                
                                                if False in Service:
                                                    json_respuesta = {'Location':True,'Time':True,'Service':False,'Porn':not Service[0],'Url_themask':'','url_thumbnail':'','msg':''}
                                                    mission_message = message_general_service(json_respuesta)
                                                    json_respuesta['msg'] = mission_message
                                                    return jsonify(json_respuesta)
                                                
                                                else:
                                                    det, detected_obj = detect_objects(image_path,validar5,objects,labels)
                                                    obj = det == validar5 or det in ppoliticos
                                                    json_respuesta = {'Location':True,'Time':True,'Service':det == validar5 or det in ppoliticos,'Porn':False,'Url_themask':imagen_final(masked_url[0]),'url_thumbnail':video_to_thumbnail_url(video_path),'msg':''}
                                                    mission_message = message_general_service(json_respuesta)
                                                    json_respuesta['msg'] = mission_message
                                                    return jsonify(json_respuesta)

                                            else:
                                                json_respuesta = {'Location':True,'Time':True,'Service':False,'Porn':False,'Url_themask':'','url_thumbnail':'','msg':''}
                                                mission_message = message_general_service(json_respuesta)
                                                json_respuesta['msg'] = mission_message
                                                return jsonify(json_respuesta)

                                        else:
                                            json_respuesta = {'Location':True,'Time':True,'Service':False,'Porn':False,'Url_themask':'','url_thumbnail':'','msg':''}
                                            mission_message = message_general_service(json_respuesta)
                                            json_respuesta['msg'] = mission_message
                                            return jsonify(json_respuesta)

                                    else:
                                        json_respuesta = {'Location':True,'Time':True,'Service':False,'Porn':False,'Url_themask':'','url_thumbnail':'','msg':''}
                                        mission_message = message_general_service(json_respuesta)
                                        json_respuesta['msg'] = mission_message
                                        return jsonify(json_respuesta)
        
                        else:
                            json_respuesta = {'Location':True,'Time':True,'Service':False,'Porn':False,'Url_themask':'','url_thumbnail':'','msg':''}
                            mission_message = message_general_service(json_respuesta)
                            json_respuesta['msg'] = mission_message
                            return jsonify(json_respuesta)                            
                            
                    else:
                        json_respuesta = {'Location':True,'Time':True,'Service':False,'Porn':False,'Url_themask':'','url_thumbnail':'','msg':''}
                        mission_message = message_general_service(json_respuesta)
                        json_respuesta['msg'] = mission_message
                        return jsonify(json_respuesta)

                else:
                    json_respuesta = {'Location':True,'Time':True,'Service':False,'Porn':False,'Url_themask':'','url_thumbnail':'','msg':''}
                    mission_message = message_general_service(json_respuesta)
                    json_respuesta['msg'] = mission_message
                    return jsonify(json_respuesta)
            
            else:
                json_respuesta = {'Location':True,'Time':True,'Service':False,'Porn':False,'Url_themask':'','url_thumbnail':'','msg':''}
                mission_message = message_general_service(json_respuesta)
                json_respuesta['msg'] = mission_message
                return jsonify(json_respuesta)
                             
        else:
            json_respuesta = {'Location':True,'Time':False,'Service':False,'Porn':False,'Url_themask':'','url_thumbnail':'','msg':''}
            mission_message = message_general_service(json_respuesta)
            json_respuesta['msg'] = mission_message
            return jsonify(json_respuesta)
       
    else:
        json_respuesta = {'Location':False,'Time':True,'Service':False,'Porn':False,'Url_themask':'','url_thumbnail':'','msg':''}
        mission_message = message_general_service(json_respuesta)
        json_respuesta['msg'] = mission_message
        return jsonify(json_respuesta)

@app.route('/explicit', methods=['POST'])
def contenido_explicito():
    global data
    global json_respuesta
    global Service

    global obj
    global from_service
    global det
    global validar1
    global validar2
    global validar3
    global validar4
    global validar5
    global validar6
    global detected_obj
    global masked_url
    global url2json0
    global video_path

    data = request.json
    data['url'] = orientation_fix_function(data['url'])
    re_explicit = request.args.get('re')
    liveness_flag = request.args.get('liveness')
    if re_explicit != 'true':
        data['url2'] = orientation_fix_function(data['url2'])
    validar1 = 'none'
    validar2 = 'none'
    validar3 = 'none'
    validar4 = 'none'
    validar5 = 'none'
    validar6 = 'none'
    obj = True
    det = 'na'
    detected_obj = 'na'
    url2json0 = ['']
    masked_url = [data['url']]

    Service = [False,False,False,False]
    from_service = 'Explicit'


    gettime = request.args.get('time')
    getloc = request.args.get('loc')

    if gettime == None and getloc == None:
        time = True
        loc = True
    elif gettime == 'y' and getloc == 'y':
        time = True
        loc = True
    elif gettime == 'n' and getloc == 'y':
        time = False
        loc = True
    elif gettime == 'y' and getloc == 'n':
        time = True
        loc = False
    elif gettime == 'n' and getloc == 'n':
        time = False
        loc = False

    if re_explicit == 'true':
        from_service = 'Re:Explicit'
        try:
            video_path = data['Url_Video']
        except KeyError as e:
            video_path = ''
            print(e)
        latitud_usuario = data['Location_latitude']
        longitud_usuario = data['Location_longitude']
        latitud_mision = data['Location_mission_latitude']
        longitud_mision = data['Location_mission_longitude']
        radio = data['Location_mission_radio']
        fecha_inicio = data['Start_Date_mission']
        fecha_final = data['End_Date_mission']
        duracion = data['Target_time_mission']
    else:
        video_path = ''
        latitud_usuario = 0
        longitud_usuario = 0
        latitud_mision = 0
        longitud_mision = 0
        radio = 100
        fecha_inicio = "2019-08-10 19:19:08.293319"
        fecha_final = "2019-08-10 20:04:08.293365"
        duracion = 9000

    forb_words = ['matar','asesinar','violar','acuchillar','secuestrar',
                  'linchar','chingar','chingate','joder','jodete','coger',
                  'follar','puto','puta','malnacido', 'golpear',
                  'pito','polla','pendeja','pendejo','pinche','mierda',
                  'concha','chingatumadre','descuartizar','mamada','sexo','pene',
                  'nepe','mamadita','cojo','cogida','pack','nudes','apuñala']
    extras = ['persona','selfie','cara']
    text = data['text']
    
    target_text_list = forb_words
    text = text.lower()
    text = unidecode.unidecode(text)
    final_text = len([word for word in target_text_list if(word in text)])>=1    

    if not final_text:

        user_pos = (latitud_usuario,longitud_usuario) 
        mission_point_pos = (latitud_mision,longitud_mision)
        
        if (geodesic(user_pos, mission_point_pos).meters <= radio or not loc):
        
            start_date = datetime.strptime(fecha_inicio, '%Y-%m-%d %H:%M:%S.%f')
            end_date = datetime.strptime(fecha_final, '%Y-%m-%d %H:%M:%S.%f')
            user_time = (end_date - start_date).total_seconds()
            mission_target_time = duracion
            
            if (user_time <= mission_target_time or not time):

                if data['url'] != '':
                    image_path = data['url']
                    p1 = Process(target=porn_explicit(image_path,0))
                    p1.start()
                    if re_explicit == 'true':
                        p2 = Process(target=screen_explicit(image_path))
                        p2.start()
                        p3 = Process(target=detect_human(image_path,'selfie',extras))
                        Service[2] = False
                        p3.start()
                        p2.join()
                        p3.join()
                        if liveness_flag == 'no':
                            Service[3] = False
                    p1.join()
                    p1.terminate
                    if re_explicit == 'true':
                        p2.terminate
                        p3.terminate
                                        
                if  not re_explicit == 'true' and data['url2'] != '':
                    image_path2 = data['url2']
                    p1 = Process(target=porn_explicit(image_path2,1))
                    p1.start()
                    p1.join()
                    p1.terminate
                        
                if True in Service:
                    json_respuesta = {'Location':True,'Time':True,'Service':False,'Porn':Service[0] or Service[1],'Url_themask':'','url_thumbnail':'','msg':''}
                    mission_message = message_general_service(json_respuesta)
                    json_respuesta['msg'] = mission_message
                    return jsonify(json_respuesta)
                else:
                    json_respuesta = {'Location':True,'Time':True,'Service':True,'Porn':Service[0] or Service[1],'Url_themask':imagen_final(masked_url[0]),'url_thumbnail':video_to_thumbnail_url(video_path),'msg':''}
                    mission_message = message_general_service(json_respuesta)
                    json_respuesta['msg'] = mission_message
                    return jsonify(json_respuesta)
            else:
                json_respuesta = {'Location':True,'Time':False,'Service':False,'Porn':Service[0] or Service[1],'Url_themask':'','url_thumbnail':'','msg':''}
                mission_message = message_general_service(json_respuesta)
                json_respuesta['msg'] = mission_message
                return jsonify(json_respuesta)
        else:
            json_respuesta = {'Location':False,'Time':False,'Service':False,'Porn':Service[0] or Service[1],'Url_themask':'','url_thumbnail':'','msg':''}
            mission_message = message_general_service(json_respuesta)
            json_respuesta['msg'] = mission_message
            return jsonify(json_respuesta)
    else:
        json_respuesta = {'Location':True,'Time':True,'Service':False,'Porn':Service[0] or Service[1],'Url_themask':'','url_thumbnail':'','msg':''}
        mission_message = message_general_service(json_respuesta)
        json_respuesta['msg'] = mission_message
        return jsonify(json_respuesta)

@app.route('/taifelds', methods=['POST'])
def taifelds_service():
    global data
    global json_respuesta
    global bandera
    global from_service
    global video_path

    from_service = 'Taifelds'

    data = request.json
    bandera = request.args.get('re_data')
    video_path = ''

    if bandera != 'yes':
        data['url'] = orientation_fix_function(data['url'])

    if bandera == 'yes':
        try:
            with engine_misions.connect() as connection:
                results = connection.execute(db.select([missions_taifelds]).where(missions_taifelds.c.Flag == 'Pending')).fetchall()
        except Exception as e:
            print(e)
            try:
                with engine_misions.connect() as connection:
                    results = connection.execute(db.select([missions_taifelds]).where(missions_taifelds.c.Flag == 'Pending')).fetchall()
            except Exception as e:
                print(e)
                return jsonify({'Service':False})
        if len(results) == 0:
            return jsonify({'Service':False})
        df = pd.DataFrame(results)
        df.columns = results[0].keys()
        lat , lng, address, ids = df['Latitude'].tolist(), df['Longitude'].tolist(), df['Address'].tolist(), df['Id'].tolist()

        id_tienda_salida = data['Id_Store']
        if id_tienda_salida not in ids:
            return jsonify({'Service':False})

        status = data['Status']
        user_id = data['User_Id']
        user_lat = data['User_Latitude']
        user_lng = data['User_Longitude']
        url_p = data['Url_Photo']
        url_v = data['Url_Video']
        video_path = data['Url_Video']
        miss_id = data['Mission_Id']

        try:
            with engine_misions.connect() as connection:
                connection.execute(missions_taifelds.update().where(missions_taifelds.c.Id == id_tienda_salida).values(Flag = status,
                    Date = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
                    User_Id = user_id, User_Latitude = user_lat, User_Longitude = user_lng, Url_Photo = url_p,
                    Url_Video = url_v, Mission_Id = miss_id))
        except Exception as e:
            print(e)
            try:
                with engine_misions.connect() as connection:
                    connection.execute(missions_taifelds.update().where(missions_taifelds.c.Id == id_tienda_salida).values(Flag = status,
                        Date = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
                        User_Id = user_id, User_Latitude = user_lat, User_Longitude = user_lng, Url_Photo = url_p,
                        Url_Video = url_v, Mission_Id = miss_id))
            except Exception as e:
                print(e)
                return jsonify({'Service':False})

        
        try:
            with engine_misions.connect() as connection:
                results_all = connection.execute(db.select([missions_taifelds]).where(missions_taifelds.c.Flag == 'Yes')).fetchall()
        except Exception as e:
            print(e)
            try:
                with engine_misions.connect() as connection:
                    results_all = connection.execute(db.select([missions_taifelds]).where(missions_taifelds.c.Flag == 'Yes')).fetchall()
            except Exception as e:
                print(e)
                return jsonify({'Service':False})

        if len(results_all) == 0:
            return jsonify({'Service':True})
        df_all = pd.DataFrame(results_all)
        df_all.columns = results_all[0].keys()
        lat , lng, address, name_soriana, url_fotos, url_videos, fechas_captura  = df_all['Latitude'].tolist(), df_all['Longitude'].tolist(), df_all['Address'].tolist(), df_all['Name'].tolist(), df_all['Url_Photo'].tolist(), df_all['Url_Video'].tolist(), df_all['Date'].tolist()
        nombre_de_mision, ids_all = df_all['Store'].tolist(), df_all['Id'].tolist()
        df_all_dict = {'Id':ids_all,'Nombre de Misión':nombre_de_mision,'Nombre de Tienda':name_soriana,'Dirección':address,'Fecha de Captura':fechas_captura,'Latitud':lat,'Longitud':lng,'Foto':url_fotos,'Video':url_videos}
        df_mrT = pd.DataFrame(df_all_dict)
        df_mrT.sort_values('Fecha de Captura',ascending=False,inplace=True)
        
        try:
            gc = pygsheets.authorize(service_file='images/gchgame-ea9d60803e55.json')

            sh = gc.open_by_url("https://docs.google.com/spreadsheets/d/1tZTqFdEtLzUV2CaYqQSpV5PYEr0rElzv_oZ9D7RvABQ/edit?usp=sharing")
            wks = sh.sheet1
            wks.clear()
            wks.set_dataframe(df_mrT,(1,1),extend = True)

        except Exception as e:
            print('Falló Sheets')
            return jsonify({'Service':False})

        try:
            index_tienda_salida_id = ids_all.index(id_tienda_salida)
            url = "https://us-central1-gchgame.cloudfunctions.net/mail-sender-to-mrT"
            body_taifelds_to_MrT = f"Hay una nueva misión Taifelds cumplida; con dirección: "

            payload = {'id_tienda':id_tienda_salida,'direccion':address[index_tienda_salida_id],'message':body_taifelds_to_MrT,
                        'foto':url_p,'video':url_v,'service':from_service,'subject':'Taifelds - MISION CUMPLIDA'}
            headers = {'Content-Type': 'application/json'}

            response = requests.request("POST", url, headers=headers, data = json.dumps(payload))

        except Exception as e:
            print('Falló Mail')
            pass

        
        return jsonify({'Service':True})

    elif bandera != None:
        print('Error de request')
        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':'Ocurrió un error inesperado con esta misión, por favor repórtalo a Soporte Gotchu!'}
        return jsonify(json_respuesta)
    else:
        pass

    try:
        with engine_misions.connect() as connection:
            results = connection.execute(db.select([missions_taifelds]).where(missions_taifelds.c.Flag != 'Yes')).fetchall()
    except Exception as e:
        print(e)
        try:
            with engine_misions.connect() as connection:
                results = connection.execute(db.select([missions_taifelds]).where(missions_taifelds.c.Flag != 'Yes')).fetchall()
        except Exception as e:
            print(e)
            json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':'Ocurrió un error inesperado, vuelve a intentarlo por favor'}
            mission_message = message_pay_service(json_respuesta)
            json_respuesta['msg'] = mission_message
            return jsonify(json_respuesta)

    if len(results) == 0:
        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':'Lo sentimos, ya no hay ubicaciones disponibles para esta misión'}
        return jsonify(json_respuesta)
    df = pd.DataFrame(results)
    df.columns = results[0].keys()
    lat , lng, address, ids = df['Latitude'].tolist(), df['Longitude'].tolist(), df['Address'].tolist(), df['Id'].tolist()
    user_pos = (data['Location_latitude'],data['Location_longitude']) 
    
    image_path = data['url']
    try:
        video_path = data['Url_Video']
    except KeyError as e:
        video_path = ''
        print(e)
    distancias = []
    for t, g in zip(lat,lng):
        distancias.append(geodesic(user_pos,(t,g)).meters)
    if min(distancias) < 300:
        
        start_date = datetime.strptime(data['Start_Date_mission'], '%Y-%m-%d %H:%M:%S.%f')
        end_date = datetime.strptime(data['End_Date_mission'], '%Y-%m-%d %H:%M:%S.%f')
        user_time = (end_date - start_date).total_seconds()
        mission_target_time = data['Target_time_mission']

        if (user_time<=mission_target_time):
            if liveness(image_path):
                j = np.argmin(distancias)
                direc = address[j]
                id_tienda = ids[j]
                data['Location_mission_latitude'] = lat[j]
                data['Location_mission_longitude'] = lng[j]
                try:
                    with engine_misions.connect() as connection:
                        connection.execute(missions_taifelds.update().where(missions_taifelds.c.Address == direc).values(Flag = 'Pending'))
                except Exception as e:
                    print(e)
                    try:
                        with engine_misions.connect() as connection:
                            connection.execute(missions_taifelds.update().where(missions_taifelds.c.Address == direc).values(Flag = 'Pending'))
                    except Exception as e:
                        print(e)
                        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':'Ocurrió un error inesperado, vuelve a intentarlo por favor'}
                        return jsonify(json_respuesta)
                
                try:

                    url = "https://us-central1-gchgame.cloudfunctions.net/mail-sender"

                    payload = {'id_tienda':id_tienda,'message':body_taifelds,'service':from_service,'subject':'Taifelds - NUEVA MISION'}
                    headers = {'Content-Type': 'application/json'}

                    response = requests.request("POST", url, headers=headers, data = json.dumps(payload))

                except Exception as e:
                    print(e)
                    pass
                json_respuesta = {'Location':True,'Time':True,'Service':True,'Live':True,'Porn':False,'Id':id_tienda,'Url_themask':imagen_final(image_path),'url_thumbnail':video_to_thumbnail_url(video_path),'msg':'','hideCapture':True}
                mission_message = message_pay_service(json_respuesta)
                json_respuesta['msg'] = mission_message
                json_respuesta['msg'] = '¡Felicidades Agente! Hemos recibido tus evidencias, pronto serán calificadas, sigue ganando con Gotchu!'
                return jsonify(json_respuesta)

            else:
                json_respuesta = {'Location':True,'Time':True,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':''}
                mission_message = message_pay_service(json_respuesta)
                json_respuesta['msg'] = mission_message
                return jsonify(json_respuesta)
        else:
            json_respuesta = {'Location':True,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':''}
            mission_message = message_pay_service(json_respuesta)
            json_respuesta['msg'] = mission_message
            return jsonify(json_respuesta)
    else:
        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':''}
        mission_message = message_pay_service(json_respuesta)
        json_respuesta['msg'] = mission_message
        return jsonify(json_respuesta)

@app.route('/taifelds2', methods=['POST'])
def taifelds_service2():
    global data
    global json_respuesta
    global bandera
    global from_service
    global video_path

    from_service = 'Taifelds'

    data = request.json
    bandera = request.args.get('re_data')
    video_path = ''

    if bandera != 'yes':
        data['url'] = orientation_fix_function(data['url'])

    if bandera == 'yes':
        try:
            with engine_misions.connect() as connection:
                results = connection.execute(db.select([missions_taifelds2]).where(missions_taifelds2.c.Flag == 'Pending')).fetchall()
        except Exception as e:
            print(e)
            try:
                with engine_misions.connect() as connection:
                    results = connection.execute(db.select([missions_taifelds2]).where(missions_taifelds2.c.Flag == 'Pending')).fetchall()
            except Exception as e:
                print(e)
                return jsonify({'Service':False})
        if len(results) == 0:
            return jsonify({'Service':False})
        df = pd.DataFrame(results)
        df.columns = results[0].keys()
        lat , lng, address, ids = df['Latitude'].tolist(), df['Longitude'].tolist(), df['Address'].tolist(), df['Id'].tolist()

        id_tienda_salida = data['Id_Store']
        if id_tienda_salida not in ids:
            return jsonify({'Service':False})

        status = data['Status']
        user_id = data['User_Id']
        user_lat = data['User_Latitude']
        user_lng = data['User_Longitude']
        try:
            url_p = data['Url_Photo']
        except:
            url_p = ''
        url_v = data['Url_Video']
        video_path = data['Url_Video']
        miss_id = data['Mission_Id']

        try:
            with engine_misions.connect() as connection:
                connection.execute(missions_taifelds2.update().where(missions_taifelds2.c.Id == id_tienda_salida).values(Flag = status,
                    Date = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
                    User_Id = user_id, User_Latitude = user_lat, User_Longitude = user_lng, Url_Photo = url_p,
                    Url_Video = url_v, Mission_Id = miss_id))
        except Exception as e:
            print(e)
            try:
                with engine_misions.connect() as connection:
                    connection.execute(missions_taifelds2.update().where(missions_taifelds2.c.Id == id_tienda_salida).values(Flag = status,
                        Date = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
                        User_Id = user_id, User_Latitude = user_lat, User_Longitude = user_lng, Url_Photo = url_p,
                        Url_Video = url_v, Mission_Id = miss_id))
            except Exception as e:
                print(e)
                return jsonify({'Service':False})

        
        return jsonify({'Service':True})

    elif bandera != None:
        print('Error de request')
        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':'Ocurrió un error inesperado con esta misión, por favor repórtalo a Soporte Gotchu!'}
        return jsonify(json_respuesta)
    else:
        pass

    try:
        with engine_misions.connect() as connection:
            results = connection.execute(db.select([missions_taifelds2]).where(missions_taifelds2.c.Flag != 'Yes')).fetchall()
    except Exception as e:
        print(e)
        try:
            with engine_misions.connect() as connection:
                results = connection.execute(db.select([missions_taifelds2]).where(missions_taifelds2.c.Flag != 'Yes')).fetchall()
        except Exception as e:
            print(e)
            json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':'Ocurrió un error inesperado, vuelve a intentarlo por favor'}
            return jsonify(json_respuesta)

    if len(results) == 0:
        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':'Lo sentimos, ya no hay ubicaciones disponibles para esta misión'}
        return jsonify(json_respuesta)
    df = pd.DataFrame(results)
    df.columns = results[0].keys()
    lat , lng, address, ids = df['Latitude'].tolist(), df['Longitude'].tolist(), df['Address'].tolist(), df['Id'].tolist()
    user_pos = (data['Location_latitude'],data['Location_longitude']) 
    
    image_path = data['url']
    try:
        video_path = data['Url_Video']
    except KeyError as e:
        video_path = ''
        print(e)
    distancias = []
    for t, g in zip(lat,lng):
        distancias.append(geodesic(user_pos,(t,g)).meters)
    if min(distancias) < 200:
        
        start_date = datetime.strptime(data['Start_Date_mission'], '%Y-%m-%d %H:%M:%S.%f')
        end_date = datetime.strptime(data['End_Date_mission'], '%Y-%m-%d %H:%M:%S.%f')
        user_time = (end_date - start_date).total_seconds()
        mission_target_time = data['Target_time_mission']

        if (user_time<=mission_target_time):
            if liveness(image_path):
                j = np.argmin(distancias)
                direc = address[j]
                id_tienda = ids[j]
                data['Location_mission_latitude'] = lat[j]
                data['Location_mission_longitude'] = lng[j]
                try:
                    with engine_misions.connect() as connection:
                        connection.execute(missions_taifelds2.update().where(missions_taifelds2.c.Address == direc).values(Flag = 'Pending'))
                except Exception as e:
                    print(e)
                    try:
                        with engine_misions.connect() as connection:
                            connection.execute(missions_taifelds2.update().where(missions_taifelds2.c.Address == direc).values(Flag = 'Pending'))
                    except Exception as e:
                        print(e)
                        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':'Ocurrió un error inesperado, vuelve a intentarlo por favor'}
                        return jsonify(json_respuesta)
                
                try:

                    url = "https://us-central1-gchgame.cloudfunctions.net/mail-sender"

                    payload = {'id_tienda':id_tienda,'message':body_taifelds,'service':from_service,'subject':'Taifelds - NUEVA MISION'}
                    headers = {'Content-Type': 'application/json'}

                    response = requests.request("POST", url, headers=headers, data = json.dumps(payload))

                except Exception as e:
                    print(e)
                    pass
                json_respuesta = {'Location':True,'Time':True,'Service':True,'Live':True,'Porn':False,'Id':id_tienda,'Url_themask':imagen_final(image_path),'url_thumbnail':video_to_thumbnail_url(video_path),'msg':'','hideCapture':True}
                mission_message = message_pay_service(json_respuesta)
                json_respuesta['msg'] = mission_message
                json_respuesta['msg'] = '¡Felicidades Agente! Hemos recibido tus evidencias, pronto serán calificadas, sigue ganando con Gotchu!'
                return jsonify(json_respuesta)

            else:
                json_respuesta = {'Location':True,'Time':True,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':''}
                mission_message = message_pay_service(json_respuesta)
                json_respuesta['msg'] = mission_message
                return jsonify(json_respuesta)
        else:
            json_respuesta = {'Location':True,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':''}
            mission_message = message_pay_service(json_respuesta)
            json_respuesta['msg'] = mission_message
            return jsonify(json_respuesta)
    else:
        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':''}
        mission_message = message_pay_service(json_respuesta)
        json_respuesta['msg'] = mission_message
        return jsonify(json_respuesta)

@app.route('/hidrosina', methods=['POST'])
def hidrosina_service():
    global data
    global json_respuesta
    global bandera
    global from_service
    global video_path

    from_service = 'Hidrosina'

    data = request.json
    bandera = request.args.get('re_data')
    video_path = ''
    data['url'] = orientation_fix_function(data['url'])

    try:
        with engine_misions.connect() as connection:
            results = connection.execute(db.select([missions_hidrosina]).where(missions_hidrosina.c.Flag != 'Yes')).fetchall() ########################TEMP#QUITADO
            results_yes = connection.execute(db.select([missions_hidrosina]).where(missions_hidrosina.c.Flag == 'Yes')).fetchall()
            results_auditoria = connection.execute(db.select([missions_hidrosina_auditoria_strikes])).fetchall()
            results_servicio = connection.execute(db.select([missions_hidrosina_servicio_strikes])).fetchall()
    except Exception as e:
        print(e)
        try:
            with engine_misions.connect() as connection:
                results = connection.execute(db.select([missions_hidrosina]).where(missions_hidrosina.c.Flag != 'Yes')).fetchall() ########################TEMP#QUITADO
                results_yes = connection.execute(db.select([missions_hidrosina]).where(missions_hidrosina.c.Flag == 'Yes')).fetchall()
                results_auditoria = connection.execute(db.select([missions_hidrosina_auditoria_strikes])).fetchall()
                results_servicio = connection.execute(db.select([missions_hidrosina_servicio_strikes])).fetchall()
        except Exception as e:
            print(e)
            json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':'Error de conexión, inténtalo nuevamente'}
            return jsonify(json_respuesta)

    if len(results) == 0:
        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':'No hay más ubicaciones disponibles para esta misión, gracias por participar'}
        return jsonify(json_respuesta)

    df = pd.DataFrame(results)
    df.columns = results[0].keys()
    lat , lng, address, ids, names_hd, zonas = df['Latitude'].tolist(), df['Longitude'].tolist(), df['Address'].tolist(), df['Id'].tolist(), df['Name'].tolist(), df['Zona'].tolist()
    hora_inf, hora_sup = df['Horario_Inf_Timestamp'], df['Horario_Sup_Timestamp']
    user_pos = (data['Location_latitude'],data['Location_longitude'])

    df_auditoria = pd.DataFrame(results_auditoria)
    df_auditoria.columns = results_auditoria[0].keys()
    hd_auditoria, strikes_auditoria = df_auditoria['HD'].tolist(), df_auditoria['Numero_de_calificaciones_bajas_en_Auditoria'].tolist()

    df_servicio = pd.DataFrame(results_servicio)
    df_servicio.columns = results_servicio[0].keys()
    hd_servicio, strikes_servicio = df_servicio['Name'].tolist(), df_servicio['Numero_de_calificaciones_bajas_en_Servicio'].tolist()

    df_yes = pd.DataFrame(results_yes)
    df_yes.columns = results_yes[0].keys()
    codigos = df_yes['Codigo_factura'].tolist()
    tickets = df_yes['No_Ticket'].tolist()
    image_path = data['url']
    try:
        video_path = data['Url_Video']
    except KeyError as e:
        video_path = ''
        print(e)

    # full_text = detect_text_uri(image_path)

    ticket_image_local_path = url_to_image(image_path)
    full_text = detect_text_local(ticket_image_local_path)

    if full_text == '':
        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':1,'Url_themask':'','url_thumbnail':'','msg':'Tu captura ha sido seleccionada para ser revisada por el equipo de Gotchu! Pronto será calificada. Sigue ganando con Gotchu!'}
        return jsonify(json_respuesta)
        
    codigo = find_codigo_factura(full_text)
    no_ticket = find_no_ticket(full_text)
    fecha, hora = find_fecha_hora(full_text)

    # fecha = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d")###DEV
    # hora = (datetime.utcnow() - timedelta(hours=6)).strftime("%H:%M")###DEV

    try:
        fecha_timestamp, hora_timestamp = fecha_hora_2_timestamp(fecha,hora)
    except Exception as e:
        print(e,fecha,hora)
        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':'Tu foto no cumple con los requerimientos o está mal enfocada, vuelve a intentarlo, si crees que esto es un error comunícate con soporte Gotchu! al teléfono 55 7161 7864'}
        return jsonify(json_respuesta)

    if fecha == '1970-01-01' or hora == '00:00':
        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':'Tu foto no cumple con los requerimientos o está mal enfocada, vuelve a intentarlo, si crees que esto es un error comunícate con soporte Gotchu! al teléfono 55 7161 7864'}
        return jsonify(json_respuesta)

    # if hora_timestamp < 21600:
    #     hora_timestamp += 86400

    distancias = []
    for t, g in zip(lat,lng):
        distancias.append(geodesic(user_pos,(t,g)).meters)

    indices_horarios = [i for i, v in enumerate(distancias) if v < 200]
    horarios_vs_real_index = [i for i in indices_horarios if hora_inf[i] < hora_timestamp < hora_sup[i]]
    
    date_time_fecha = datetime.fromtimestamp(fecha_timestamp)
    fecha_hoy_str = date_time_fecha.strftime("%Y-%m-%d")
    fecha_hoy_str_real = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d")

    if fecha_hoy_str_real != fecha_hoy_str:
        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':'Tu foto no cumple con los requerimientos o está mal enfocada, vuelve a intentarlo, si crees que esto es un error comunícate con soporte Gotchu! al teléfono 55 7161 7864'}
        return jsonify(json_respuesta)

    if len(horarios_vs_real_index) > 0:
        
        start_date = datetime.strptime(data['Start_Date_mission'], '%Y-%m-%d %H:%M:%S.%f')
        end_date = datetime.strptime(data['End_Date_mission'], '%Y-%m-%d %H:%M:%S.%f')
        user_time = (end_date - start_date).total_seconds()
        mission_target_time = data['Target_time_mission']

        if (user_time<=mission_target_time):
            if True: #no_ticket not in tickets:##########DEV
                j = horarios_vs_real_index[0]
                id_tienda = ids[j]
                name_hd = names_hd[j]
                j_auditoria = hd_auditoria.index(name_hd)
                numero_de_strikes = strikes_auditoria[j_auditoria]
                zona = zonas[j]
                data['Location_mission_latitude'] = lat[j]
                data['Location_mission_longitude'] = lng[j]
                address_hidro = address[j]
                user_id = data['id']
                user_lat = user_pos[0]
                user_lng = user_pos[1]
                miss_id = data['id_mission']
                qr_decoded = url_qr_2_text(image_path)
                alerta_sanitaria = 'Ninguna'
                alerta_precios_apagados = 'No'
                alerta_ticket_incorrecto = 'No'
                score = 0
                try:
                    hash_typeform = data['hash_typeform']
                except KeyError as e:
                    hash_typeform = ''
                    print('no hay variable hash_typeform')
                # if flags[j] == 'No': #################################TEMP


                try:

                    gc = pygsheets.authorize(service_file='images/gchgame-ea9d60803e55.json')
                    sh = gc.open_by_url("https://docs.google.com/spreadsheets/d/1EGnt8H7Bwxs_EUCB4hiavwcnkHdMlj0Y4Wr0aD6Wqz0/edit?usp=sharing")
                    wks = sh.sheet1
                    df_typeform = wks.get_as_df()

                    df_typeform['uniforme limpio'] = ['uniforme limpio' if type(x) != float and 'limpio' in x else np.nan for x in df_typeform['¿Qué no tenía el despachador?']]
                    df_typeform['cubrebocas'] = ['cubrebocas' if type(x) != float and 'cubrebocas' in x else np.nan for x in df_typeform['¿Qué no tenía el despachador?']]
                    df_typeform['lentes protectores'] = ['lentes protectores' if type(x) != float and 'lentes' in x else np.nan for x in df_typeform['¿Qué no tenía el despachador?']]
                    df_typeform['uniforme limpio'] = np.where(df_typeform['uniforme limpio'].isnull(),1,0)
                    df_typeform['cubrebocas'] = np.where(df_typeform['cubrebocas'].isnull(),1,0)
                    df_typeform['lentes protectores'] = np.where(df_typeform['lentes protectores'].isnull(),1,0)
                    df_typeform['Los despachadores, ¿traían uniforme limpio, cubrebocas y lentes protectores?'] = (df_typeform['uniforme limpio'] + df_typeform['cubrebocas'] + df_typeform['lentes protectores']) * 1 / 3

                    df_row = df_typeform.loc[df_typeform['capture_id'] == hash_typeform]
                    if len(df_row) == 0:
                        print('Hash no encontrado')
                    else:

                        alerta_cubrebocas = df_row['¿Qué no tenía el despachador?'].to_list()[0]
                        alerta_precios = df_row['¿Qué falta?'].to_list()[0]
                        
                        if 'cubrebocas' in alerta_cubrebocas:
                            alerta_sanitaria = 'cubrebocas'
                            if zona == 'Metro':
                                try:
                                    fecha_incidente = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S")
                                    nombre_despachador = qr_decoded
                                    url = "https://us-central1-gchgame.cloudfunctions.net/mail-sender-alerta-metro"

                                    payload = {'id_tienda':name_hd,'message':body_hidrosina_alerta,'service':from_service,'nombre_despachador':nombre_despachador,'fecha_incidente':fecha_incidente,'Codigo_factura':codigo,'subject':'ALERTA DE SEGURIDAD SANITARIA'}
                                    headers = {'Content-Type': 'application/json'}

                                    mail_process1 = Process(target = enviar_mail, args = (url,headers,payload))
                                    mail_process1.start()

                                except Exception as e:
                                    print(e,'fallo email alerta')
                                    pass
                            else:
                                try:
                                    fecha_incidente = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S")
                                    nombre_despachador = qr_decoded
                                    url = "https://us-central1-gchgame.cloudfunctions.net/mail-sender-alerta-provincia"

                                    payload = {'id_tienda':name_hd,'message':body_hidrosina_alerta,'service':from_service,'nombre_despachador':nombre_despachador,'fecha_incidente':fecha_incidente,'Codigo_factura':codigo,'subject':'ALERTA DE SEGURIDAD SANITARIA'}
                                    headers = {'Content-Type': 'application/json'}

                                    mail_process1 = Process(target = enviar_mail, args = (url,headers,payload))
                                    mail_process1.start()

                                except Exception as e:
                                    print(e,'fallo email alerta')
                                    pass
                        
                        if 'precios' in alerta_precios:
                            alerta_precios_apagados = 'Si'
                            if zona == 'Metro':
                                try:
                                    fecha_incidente = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S")
                                    nombre_despachador = qr_decoded
                                    url = "https://us-central1-gchgame.cloudfunctions.net/mail-sender-alerta-metro"

                                    payload = {'id_tienda':name_hd,'message':body_hidrosina_alerta_precios,'service':from_service,'nombre_despachador':nombre_despachador,'fecha_incidente':fecha_incidente,'Codigo_factura':codigo,'subject':'ALERTA DE LETRERO DE PRECIOS APAGADO'}
                                    headers = {'Content-Type': 'application/json'}

                                    mail_process6 = Process(target = enviar_mail, args = (url,headers,payload))
                                    mail_process6.start()

                                except Exception as e:
                                    print(e,'fallo email alerta')
                                    pass
                            else:
                                try:
                                    fecha_incidente = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S")
                                    nombre_despachador = qr_decoded
                                    url = "https://us-central1-gchgame.cloudfunctions.net/mail-sender-alerta-provincia"

                                    payload = {'id_tienda':name_hd,'message':body_hidrosina_alerta_precios,'service':from_service,'nombre_despachador':nombre_despachador,'fecha_incidente':fecha_incidente,'Codigo_factura':codigo,'subject':'ALERTA DE LETRERO DE PRECIOS APAGADO'}
                                    headers = {'Content-Type': 'application/json'}

                                    mail_process6 = Process(target = enviar_mail, args = (url,headers,payload))
                                    mail_process6.start()

                                except Exception as e:
                                    print(e,'fallo email alerta')
                                    pass
                        
                        if df_row['¿Entregó el ticket correcto?'].values == 'FALSE':
                            alerta_ticket_incorrecto = 'Si'
                            if zona == 'Metro':
                                try:
                                    fecha_incidente = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S")
                                    nombre_despachador = qr_decoded
                                    url = "https://us-central1-gchgame.cloudfunctions.net/mail-sender-alerta-metro"

                                    payload = {'id_tienda':name_hd,'message':body_hidrosina_alerta_ticket,'service':from_service,'nombre_despachador':nombre_despachador,'fecha_incidente':fecha_incidente,'Codigo_factura':codigo,'subject':'ALERTA DE TICKET INCORRECTO'}
                                    headers = {'Content-Type': 'application/json'}

                                    mail_process6 = Process(target = enviar_mail, args = (url,headers,payload))
                                    mail_process6.start()

                                except Exception as e:
                                    print(e,'fallo email alerta')
                                    pass
                            else:
                                try:
                                    fecha_incidente = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S")
                                    nombre_despachador = qr_decoded
                                    url = "https://us-central1-gchgame.cloudfunctions.net/mail-sender-alerta-provincia"

                                    payload = {'id_tienda':name_hd,'message':body_hidrosina_alerta_ticket,'service':from_service,'nombre_despachador':nombre_despachador,'fecha_incidente':fecha_incidente,'Codigo_factura':codigo,'subject':'ALERTA DE TICKET INCORRECTO'}
                                    headers = {'Content-Type': 'application/json'}

                                    mail_process6 = Process(target = enviar_mail, args = (url,headers,payload))
                                    mail_process6.start()

                                except Exception as e:
                                    print(e,'fallo email alerta')

                        if df_row['¿Identificas en la estación un trabajador con bata amarilla?'].values == 'FALSE' and (39600 < hora_timestamp < 54000 or 57600 < hora_timestamp < 63000) and (str(name_hd) not in ['67','69','70','71','72','73','81','107','108']):
                            if zona == 'Metro':
                                try:
                                    fecha_incidente = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S")
                                    nombre_despachador = qr_decoded
                                    url = "https://us-central1-gchgame.cloudfunctions.net/mail-sender-alerta-metro"

                                    payload = {'id_tienda':name_hd,'message':body_hidrosina_alerta_gerente,'service':from_service,'nombre_despachador':'No aplica','fecha_incidente':fecha_incidente,'Codigo_factura':codigo,'subject':'ALERTA DE AUSENCIA DE GERENTE DE ESTACION'}
                                    headers = {'Content-Type': 'application/json'}

                                    mail_process8 = Process(target = enviar_mail, args = (url,headers,payload))
                                    mail_process8.start()

                                except Exception as e:
                                    print(e,'fallo email alerta')
                                    
                            else:
                                try:
                                    fecha_incidente = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S")
                                    nombre_despachador = qr_decoded
                                    url = "https://us-central1-gchgame.cloudfunctions.net/mail-sender-alerta-provincia"

                                    payload = {'id_tienda':name_hd,'message':body_hidrosina_alerta_gerente,'service':from_service,'nombre_despachador':'No aplica','fecha_incidente':fecha_incidente,'Codigo_factura':codigo,'subject':'ALERTA DE AUSENCIA DE GERENTE DE ESTACION'}
                                    headers = {'Content-Type': 'application/json'}

                                    mail_process8 = Process(target = enviar_mail, args = (url,headers,payload))
                                    mail_process8.start()

                                except Exception as e:
                                    print(e,'fallo email alerta')



                        if 'pudo leer' in qr_decoded:

                            try:
                                fecha_incidente = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S")
                                url = "https://us-central1-gchgame.cloudfunctions.net/mail-sender-alerta-qr-ilegible"

                                payload = {'id_tienda':name_hd,'message':body_hidrosina_alerta_qr,'service':from_service,'fecha_incidente':fecha_incidente,'url_ticket':image_path,'subject':'ALERTA DE QR ILEGIBLE'}
                                headers = {'Content-Type': 'application/json'}

                                mail_process7 = Process(target = enviar_mail, args = (url,headers,payload))
                                mail_process7.start()

                            except Exception as e:
                                print(e,'fallo email alerta')
                                pass


                        df_row_examen = df_row[['La estación, ¿lucía limpia, bien iluminada y con el letrero de precios encendido?',
                                                '¿Había despachadores indicando el lugar disponible para cargar combustible?',
                                                'Los despachadores, ¿traían uniforme limpio, cubrebocas y lentes protectores?',
                                                'El despachador ¿dio la bienvenida?',
                                                '¿Preguntó la cantidad, tipo de combustible a cargar y forma de pago?',
                                                '¿Mostró que la bomba estuviera en ceros antes de iniciar la carga?',
                                                '¿Ofreció algún producto periférico y limpieza de parabrisas?',
                                                '¿Tenía puesto el gafete en un lugar visible?',
                                                '¿Fue amable y cordial al atenderle?',
                                                '¿Entregó el ticket correcto?']]
                        df_row_examen = df_row_examen.replace(['FALSE','TRUE'],[False,True]) * 1
                        puntajes = pd.Series([35,25,40,20,15,10,15,10,20,10])
                        puntajes_auditoria = pd.Series([35,25,40,0,0,0,0,0,0,0])
                        puntajes_servicio = pd.Series([0,0,0,20,15,10,15,10,20,10])
                        calificacion = df_row_examen.values @ puntajes
                        calificacion_auditoria = df_row_examen.values @ puntajes_auditoria
                        calificacion_servicio = df_row_examen.values @ puntajes_servicio

                        if df_row['¿Los despachadores estaban ocupados atendiendo clientes?'].values == 'TRUE':
                            calificacion += 25

                        score = int(calificacion / 2)
                        calificacion = score
                        score_auditoria = int(calificacion_auditoria)
                        score_servicio = int(calificacion_servicio)
                        

                        if calificacion == 100:
                            try:
                                url = "https://us-central1-gchgame.cloudfunctions.net/mail-sender-alerta"

                                payload = {'id_tienda':name_hd,'message':body_hidrosina_alerta_100,'service':from_service,'subject':'ALERTA DE CALIFICACION POCO CONFIABLE'}
                                headers = {'Content-Type': 'application/json'}

                                mail_process2 = Process(target = enviar_mail, args = (url,headers,payload))
                                mail_process2.start()
                            except Exception as e:
                                print(e,'fallo email alerta')
                                pass
                        if calificacion < 50:
                            try:
                                url = "https://us-central1-gchgame.cloudfunctions.net/mail-sender-alerta"

                                payload = {'id_tienda':name_hd,'message':body_hidrosina_alerta_calificacion_baja,'service':from_service,'subject':'ALERTA DE CALIFICACION BAJA'}
                                headers = {'Content-Type': 'application/json'}

                                mail_process3 = Process(target = enviar_mail, args = (url,headers,payload))
                                mail_process3.start()
                            except Exception as e:
                                print(e,'fallo email alerta')
                                pass
                        if score_auditoria < 90:
                            if numero_de_strikes >= 1:
                                if zona == 'Metro':
                                    try:
                                        fecha_incidente = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S")
                                        nombre_despachador = qr_decoded
                                        url = "https://us-central1-gchgame.cloudfunctions.net/mail-sender-alerta-metro"

                                        payload = {'id_tienda':name_hd,'message':body_hidrosina_alerta_auditoria_baja,'service':from_service,'nombre_despachador':'no aplica','fecha_incidente':fecha_incidente,'Codigo_factura':'no aplica','subject':'ALERTA DE CALIFICACION BAJA Y RECURRENTE EN AUDITORIA'}
                                        headers = {'Content-Type': 'application/json'}

                                        mail_process31 = Process(target = enviar_mail, args = (url,headers,payload))
                                        mail_process31.start()
                                    except Exception as e:
                                        print(e,'fallo email alerta')
                                        pass
                                else:
                                    try:
                                        fecha_incidente = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S")
                                        nombre_despachador = qr_decoded
                                        url = "https://us-central1-gchgame.cloudfunctions.net/mail-sender-alerta-provincia"


                                        payload = {'id_tienda':name_hd,'message':body_hidrosina_alerta_auditoria_baja,'service':from_service,'nombre_despachador':'no aplica','fecha_incidente':fecha_incidente,'Codigo_factura':'no aplica','subject':'ALERTA DE CALIFICACION BAJA Y RECURRENTE EN AUDITORIA'}
                                        headers = {'Content-Type': 'application/json'}

                                        mail_process31 = Process(target = enviar_mail, args = (url,headers,payload))
                                        mail_process31.start()
                                    except Exception as e:
                                        print(e,'fallo email alerta')
                                        pass

                            try:
                                with engine_misions.connect() as connection:
                                    connection.execute(missions_hidrosina_auditoria_strikes.update().where(missions_hidrosina_auditoria_strikes.c.HD == int(name_hd)).values(
                                        Numero_de_calificaciones_bajas_en_Auditoria = int(numero_de_strikes) + 1
                                        ))
                            except Exception as e:
                                print(e)
                                try:
                                    with engine_misions.connect() as connection:
                                        connection.execute(missions_hidrosina_auditoria_strikes.update().where(missions_hidrosina_auditoria_strikes.c.HD == int(name_hd)).values(
                                            Numero_de_calificaciones_bajas_en_Auditoria = int(numero_de_strikes) + 1
                                            ))
                                except Exception as e:
                                    print(e)

                        if score_servicio < 80:
                            if qr_decoded in hd_servicio:
                                j_servicio = hd_servicio.index(qr_decoded)
                                numero_de_strikes_servicio = strikes_servicio[j_servicio]
                                if numero_de_strikes_servicio >= 1 and qr_decoded != 'No se pudo leer nombre de ticket':
                                    if zona == 'Metro':
                                        try:
                                            fecha_incidente = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S")
                                            nombre_despachador = qr_decoded
                                            url = "https://us-central1-gchgame.cloudfunctions.net/mail-sender-alerta-metro"

                                            payload = {'id_tienda':name_hd,'message':body_hidrosina_alerta_servicio_bajo,'service':from_service,'nombre_despachador':nombre_despachador,'fecha_incidente':fecha_incidente,'Codigo_factura':codigo,'subject':'ALERTA DE CALIFICACION BAJA Y RECURRENTE EN SERVICIO'}
                                            headers = {'Content-Type': 'application/json'}

                                            mail_process32 = Process(target = enviar_mail, args = (url,headers,payload))
                                            mail_process32.start()
                                        except Exception as e:
                                            print(e,'fallo email alerta')
                                            pass
                                    else:
                                        try:
                                            fecha_incidente = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S")
                                            nombre_despachador = qr_decoded
                                            url = "https://us-central1-gchgame.cloudfunctions.net/mail-sender-alerta-provincia"

                                            payload = {'id_tienda':name_hd,'message':body_hidrosina_alerta_servicio_bajo,'service':from_service,'nombre_despachador':nombre_despachador,'fecha_incidente':fecha_incidente,'Codigo_factura':codigo,'subject':'ALERTA DE CALIFICACION BAJA Y RECURRENTE EN SERVICIO'}
                                            headers = {'Content-Type': 'application/json'}

                                            mail_process32 = Process(target = enviar_mail, args = (url,headers,payload))
                                            mail_process32.start()
                                        except Exception as e:
                                            print(e,'fallo email alerta')
                                            pass


                                try:
                                    with engine_misions.connect() as connection:
                                        connection.execute(missions_hidrosina_servicio_strikes.update().where(missions_hidrosina_servicio_strikes.c.Name == qr_decoded).values(
                                            Numero_de_calificaciones_bajas_en_Servicio = int(numero_de_strikes_servicio) + 1
                                            ))
                                except Exception as e:
                                    print(e)
                                    try:
                                        with engine_misions.connect() as connection:
                                            connection.execute(missions_hidrosina_servicio_strikes.update().where(missions_hidrosina_servicio_strikes.c.Name == qr_decoded).values(
                                                Numero_de_calificaciones_bajas_en_Servicio = int(numero_de_strikes_servicio) + 1
                                                ))
                                    except Exception as e:
                                        print(e)
                            else:
                                try:
                                    with engine_misions.connect() as connection:
                                        connection.execute(missions_hidrosina_servicio_strikes.insert().values(Name = qr_decoded, Numero_de_calificaciones_bajas_en_Servicio = 1))
                                except:
                                    try:
                                        with engine_misions.connect() as connection:
                                            connection.execute(missions_hidrosina_servicio_strikes.insert().values(Name = qr_decoded, Numero_de_calificaciones_bajas_en_Servicio = 1))
                                    except Exception as e:
                                        print(e)
                                



                except Exception as e:
                    print('Falló Sheets',e)


                try:
                    with engine_misions.connect() as connection:
                        connection.execute(missions_hidrosina.update().where(missions_hidrosina.c.Id == id_tienda).values(Flag = 'Yes',
                            Date = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"), Fecha_Hora = fecha + ' ' + hora,
                            User_Id = user_id, User_Latitude = user_lat, User_Longitude = user_lng, Url_Photo = image_path,
                            Url_Video = video_path, Mission_Id = miss_id, Codigo_factura = codigo, Hash_typeform = hash_typeform, Score = calificacion,
                            Score_auditoria = score_auditoria, Score_servicio = score_servicio,
                            No_Ticket = no_ticket, QR_Decoded = qr_decoded, Alerta = alerta_sanitaria, Alerta_precios = alerta_precios_apagados,
                            Alerta_ticket = alerta_ticket_incorrecto))
                        if (datetime.utcnow() - timedelta(hours=6)).day < 15:
                            connection.execute(missions_hidrosina.update().where(missions_hidrosina.c.Address == address_hidro).values(Flag = 'Yes'))
                except Exception as e:
                    print(e)
                    try:
                        with engine_misions.connect() as connection:
                            connection.execute(missions_hidrosina.update().where(missions_hidrosina.c.Id == id_tienda).values(Flag = 'Yes',
                                Date = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"), Fecha_Hora = fecha + ' ' + hora,
                                User_Id = user_id, User_Latitude = user_lat, User_Longitude = user_lng, Url_Photo = image_path,
                                Url_Video = video_path, Mission_Id = miss_id, Codigo_factura = codigo, Hash_typeform = hash_typeform, Score = calificacion,
                                Score_auditoria = score_auditoria, Score_servicio = score_servicio,
                                No_Ticket = no_ticket, QR_Decoded = qr_decoded, Alerta = alerta_sanitaria, Alerta_precios = alerta_precios_apagados,
                                Alerta_ticket = alerta_ticket_incorrecto))
                        if (datetime.utcnow() - timedelta(hours=6)).day < 15:
                            connection.execute(missions_hidrosina.update().where(missions_hidrosina.c.Address == address_hidro).values(Flag = 'Yes'))
                    except Exception as e:
                        print(e)
                        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':'Error de conexión, inténtalo nuevamente'}
                        return jsonify(json_respuesta)
                

                try:

                    url = "https://us-central1-gchgame.cloudfunctions.net/mail-sender"

                    payload = {'id_tienda':id_tienda,'message':body_hidrosina,'service':from_service,'subject':'Hidrosina - NUEVA MISION'}
                    headers = {'Content-Type': 'application/json'}

                    mail_process4 = Process(target = enviar_mail, args = (url,headers,payload))
                    mail_process4.start()

                except Exception as e:
                    print(e)
                    pass

                json_respuesta = {'Location':True,'Time':True,'Service':True,'Live':True,'Porn':False,'Id':0,'Url_themask':image_path,'url_thumbnail':video_to_thumbnail_url(video_path),'msg':'¡Misión completada! Felicitaciones Agente'}
                return jsonify(json_respuesta)

            else:
                json_respuesta = {'Location':True,'Time':True,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':'Tu número de ticket no es correcto o ya se ha registrado'}
                return jsonify(json_respuesta)
        else:
            json_respuesta = {'Location':True,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':'Esta misión ha expirado'}
            return jsonify(json_respuesta)
    else:
        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':'No te encuentras en una ubicación u horario disponible o esta estación ha sido completada en este periodo, si crees que esto es un error comunícate con soporte Gotchu! al teléfono 55 7161 7864'}
        return jsonify(json_respuesta)

@app.route('/hidrosina-tabla-summary', methods=['POST'])
def hidrosina_summary():
    global from_service
    from_service = 'hidro_summary'
    try:
        with engine_misions.connect() as connection:
            results = connection.execute(db.select([missions_hidrosina])).fetchall()          
    except Exception as e:
        print(e)
        try:
            with engine_misions.connect() as connection:
                results = connection.execute(db.select([missions_hidrosina])).fetchall()
        except Exception as e:
            print(e)
            return jsonify({'Service':False,'Step':1})

    df_hidro = pd.DataFrame(results)
    df_hidro.columns = results[0].keys()
    gc = pygsheets.authorize(service_file='images/gchgame-ea9d60803e55.json')
    sh = gc.open_by_url("https://docs.google.com/spreadsheets/d/1EGnt8H7Bwxs_EUCB4hiavwcnkHdMlj0Y4Wr0aD6Wqz0/edit?usp=sharing")
    wks = sh.sheet1
    df_typeform = wks.get_as_df()
    df_typeform = df_typeform.drop_duplicates('capture_id')
    df_joined = pd.merge(df_hidro, df_typeform, left_on='Hash_typeform', right_on='capture_id')
    df_hidrosina = df_joined

    ##Mod preprocesamiento
    df_hidrosina = df_hidrosina.replace(['TRUE','FALSE'],[1,0])
    df_hidrosina['No luce limpia'] = ['No luce limpia' if type(x) != float and 'limpia' in x else np.nan for x in df_hidrosina['¿Qué falta?']]
    df_hidrosina['No esta bien iluminada'] = ['No esta bien iluminada' if type(x) != float and 'iluminada' in x else np.nan for x in df_hidrosina['¿Qué falta?']]
    df_hidrosina['No está el letrero de precios encendido'] = ['No está el letrero de precios encendido' if type(x) != float and 'precios' in x else np.nan for x in df_hidrosina['¿Qué falta?']]

    df_hidrosina['¿Algún despachador te señaló dónde cargar?'] = df_hidrosina['¿Había despachadores indicando el lugar disponible para cargar combustible?']
    df_hidrosina['Los despachadores, ¿traían uniforme limpio, cubrebocas y lentes?'] = df_hidrosina['Los despachadores, ¿traían uniforme limpio, cubrebocas y lentes protectores?']
    df_hidrosina['Encuentra a un trabajador con bata amarilla en la estación. ¿Lo localizaste?'] = df_hidrosina['¿Identificas en la estación un trabajador con bata amarilla?']

    df_hidrosina['uniforme limpio'] = ['uniforme limpio' if type(x) != float and 'limpio' in x else np.nan for x in df_hidrosina['¿Qué no tenía el despachador?']]
    df_hidrosina['cubrebocas'] = ['cubrebocas' if type(x) != float and 'cubrebocas' in x else np.nan for x in df_hidrosina['¿Qué no tenía el despachador?']]
    df_hidrosina['lentes protectores'] = ['lentes protectores' if type(x) != float and 'lentes' in x else np.nan for x in df_hidrosina['¿Qué no tenía el despachador?']]

    df_hidrosina['Cantidad de combustible'] = ['Cantidad de combustible' if type(x) != float and 'Cantidad' in x else np.nan for x in df_hidrosina['¿Qué no te preguntó el despachador?']]
    df_hidrosina['Tipo de combustible'] = ['Tipo de combustible' if type(x) != float and 'Tipo' in x else np.nan for x in df_hidrosina['¿Qué no te preguntó el despachador?']]
    df_hidrosina['Forma de pago'] = ['Forma de pago' if type(x) != float and 'pago' in x else np.nan for x in df_hidrosina['¿Qué no te preguntó el despachador?']]

    df_hidrosina['Producto periférico'] = ['Producto periférico' if type(x) != float and 'Producto' in x else np.nan for x in df_hidrosina['¿Qué no te ofreció?']]
    df_hidrosina['Limpieza de parabrisas'] = ['Limpieza de parabrisas' if type(x) != float and 'Limpieza' in x else np.nan for x in df_hidrosina['¿Qué no te ofreció?']]

    df_hidrosina['Trato amable'] = ['Trato amable' if type(x) != float and 'Trato' in x else np.nan for x in df_hidrosina['¿Qué calificativos describen mejor al despachador?']]
    df_hidrosina['Limpio'] = ['Limpio' if type(x) != float and 'Limpio' in x else np.nan for x in df_hidrosina['¿Qué calificativos describen mejor al despachador?']]
    df_hidrosina['Excelente actitud'] = ['Excelente actitud' if type(x) != float and 'Excelente' in x else np.nan for x in df_hidrosina['¿Qué calificativos describen mejor al despachador?']]
    df_hidrosina['Grosero'] = ['Grosero' if type(x) != float and 'Grosero' in x else np.nan for x in df_hidrosina['¿Qué calificativos describen mejor al despachador?']]
    df_hidrosina['Comunicación clara'] = ['Comunicación clara' if type(x) != float and 'clara' in x else np.nan for x in df_hidrosina['¿Qué calificativos describen mejor al despachador?']]
    df_hidrosina['Atención rápida'] = ['Atención rápida' if type(x) != float and 'Atenc' in x else np.nan for x in df_hidrosina['¿Qué calificativos describen mejor al despachador?']]
    df_hidrosina['Mala comunicación'] = ['Mala comunicación' if type(x) != float and 'Mala' in x else np.nan for x in df_hidrosina['¿Qué calificativos describen mejor al despachador?']]
    df_hidrosina['Sucio'] = ['Sucio' if type(x) != float and 'Sucio' in x else np.nan for x in df_hidrosina['¿Qué calificativos describen mejor al despachador?']]
    df_hidrosina['Atención lenta'] = ['Atención lenta' if type(x) != float and 'lenta' in x else np.nan for x in df_hidrosina['¿Qué calificativos describen mejor al despachador?']]

    df_hidrosina['Excelente'] = ['Excelente' if type(x) != float and 'Excelente' in x else np.nan for x in df_hidrosina['Selecciona las etiquetas que describan mejor a la estación']]
    df_hidrosina['Cumple protocolo COVID'] = ['Cumple protocolo COVID' if type(x) != float and 'Cumple protocolo COVID' in x else np.nan for x in df_hidrosina['Selecciona las etiquetas que describan mejor a la estación']]
    df_hidrosina['Regular'] = ['Regular' if type(x) != float and 'Regular' in x else np.nan for x in df_hidrosina['Selecciona las etiquetas que describan mejor a la estación']]
    df_hidrosina['No cumple protocolo COVID'] = ['No cumple protocolo COVID' if type(x) != float and 'No cumple protocolo COVID' in x else np.nan for x in df_hidrosina['Selecciona las etiquetas que describan mejor a la estación']]
    df_hidrosina['Decepcionante'] = ['Decepcionante' if type(x) != float and 'Decepcionante' in x else np.nan for x in df_hidrosina['Selecciona las etiquetas que describan mejor a la estación']]

    estado=[]
    zona=[]

    for j in df_hidrosina['Address']:
        e=j.split(re.findall(r'[0-9]+', j)[-1])[-1].strip(" ")
        e=e.strip('(R)').strip(" ")
        estado.append(e)

    df_hidrosina['estado']=estado
    df_hidrosina['zona']=np.where(df_hidrosina['Zona']=='Metro','Megalópolis','Provincia')
    zonas=df_hidrosina[['Name','estado','zona']].drop_duplicates()

    ##Estación
    df_hidrosina['uniforme limpio'] = np.where(df_hidrosina['uniforme limpio'].isnull(),1,0)
    df_hidrosina['cubrebocas'] = np.where(df_hidrosina['cubrebocas'].isnull(),1,0)
    df_hidrosina['lentes protectores'] = np.where(df_hidrosina['lentes protectores'].isnull(),1,0)

    df_hidrosina['score_limpia'] = df_hidrosina['La estación, ¿lucía limpia, bien iluminada y con el letrero de precios encendido?'].apply(lambda x: 35 if x == 1 else 0)
    df_hidrosina['score_bandereando'] = df_hidrosina['¿Algún despachador te señaló dónde cargar?'].apply(lambda x: 25 if x == 1 else 0)

    df_hidrosina.loc[(df_hidrosina.score_bandereando == 0) & 
                    (df_hidrosina['¿Los despachadores estaban ocupados atendiendo clientes?'] == 1), 'score_bandereando'] = 25

    df_hidrosina['score_uniforme'] = (df_hidrosina['uniforme limpio'] + df_hidrosina['cubrebocas'] + df_hidrosina['lentes protectores']) * 40 / 3
    # df_hidrosina['score_uniforme'] = df_hidrosina['Los despachadores, ¿traían uniforme limpio, cubrebocas y lentes protectores?'].apply(lambda x: 40 if x == 1 else 0)

    #Despachador y Servicio

    df_hidrosina['score_bienvenida'] = df_hidrosina['El despachador ¿dio la bienvenida?'].apply(lambda x: 20 if x == 1 else 0)
    df_hidrosina['score_cantidad'] = df_hidrosina['¿Preguntó la cantidad, tipo de combustible a cargar y forma de pago?'].apply(lambda x: 15 if x == 1 else 0)
    df_hidrosina['score_ceros'] = df_hidrosina['¿Mostró que la bomba estuviera en ceros antes de iniciar la carga?'].apply(lambda x: 10 if x == 1 else 0)
    df_hidrosina['score_pextra'] = df_hidrosina['¿Ofreció algún producto periférico y limpieza de parabrisas?'].apply(lambda x: 15 if x == 1 else 0)
    df_hidrosina['score_ticket'] = df_hidrosina['¿Entregó el ticket correcto?'].apply(lambda x: 10 if x == 1 else 0)
    df_hidrosina['score_gafete'] = df_hidrosina['¿Tenía puesto el gafete en un lugar visible?'].apply(lambda x: 10 if x == 1 else 0)
    df_hidrosina['score_amable'] = df_hidrosina['¿Fue amable y cordial al atenderle?'].apply(lambda x: 20 if x == 1 else 0)

    df_hidrosina['score_estacion']=df_hidrosina['score_limpia']+df_hidrosina['score_bandereando']+df_hidrosina['score_uniforme']
    df_hidrosina['score_servicio']=df_hidrosina['score_bienvenida']+df_hidrosina['score_cantidad']+df_hidrosina['score_ceros']+df_hidrosina['score_pextra']+df_hidrosina['score_ticket']+df_hidrosina['score_gafete']+df_hidrosina['score_amable']
    df_hidrosina['score_total']=df_hidrosina['score_estacion']+df_hidrosina['score_servicio']

    df_hidrosina_score=df_hidrosina.groupby(['Name','Latitude','Longitude'])['score_estacion','score_servicio','score_total'].mean().reset_index()
    df_hidrosina_count=df_hidrosina.groupby(['Name','Latitude','Longitude'])['Id'].count().reset_index()
    df_hidrosina_dist=df_hidrosina.groupby(['Name','Latitude','Longitude'])[ 'Excelente',
                                                            'Cumple protocolo COVID',
                                                            'Regular',
                                                            'No cumple protocolo COVID',
                                                            'Decepcionante',
                                                            'Trato amable',
                                                            'Limpio',
                                                            'Excelente actitud',
                                                            'Grosero',
                                                            'Comunicación clara',
                                                            'Atención rápida',
                                                            'Mala comunicación',
                                                            'Sucio',
                                                            'Atención lenta'].count().reset_index()

    df_hidrosina_likee=pd.crosstab(df_hidrosina['Name'], df_hidrosina['¿Like a la estación?']).reset_index()
    df_hidrosina_likee=df_hidrosina_likee.rename(columns={0: "no_like_estacion", 1: "like_estacion"})
    df_hidrosina_liked=pd.crosstab(df_hidrosina['Name'], df_hidrosina['¿Like al despachador por su servicio?']).reset_index()
    df_hidrosina_liked=df_hidrosina_liked.rename(columns={0: "no_like_servicio", 1: "like_servicio"})
    df_hidrosina_tiempo=pd.crosstab(df_hidrosina['Name'], df_hidrosina['¿Cuánto tiempo estuviste formado para recibir el servicio?']).reset_index()
    try: dummie = df_hidrosina_tiempo['Más de 10 minutos']
    except: df_hidrosina_tiempo['Más de 10 minutos'] = 0

    df_hidrosina_summary=pd.merge(df_hidrosina_score,df_hidrosina_count,on=["Name",'Latitude','Longitude'],how="left")
    df_hidrosina_summary=pd.merge(df_hidrosina_summary,df_hidrosina_dist,on=["Name",'Latitude','Longitude'],how="left")
    df_hidrosina_summary=pd.merge(df_hidrosina_summary,df_hidrosina_likee,on=["Name"],how="left")
    df_hidrosina_summary=pd.merge(df_hidrosina_summary,df_hidrosina_liked,on=["Name"],how="left")
    df_hidrosina_summary=pd.merge(df_hidrosina_summary,df_hidrosina_tiempo,on=["Name"],how="left")
    df_hidrosina_summary=df_hidrosina_summary.rename(columns={'Id': "total_encuestas"})
    df_hidrosina_summary=pd.merge(df_hidrosina_summary,zonas,on=["Name"],how="left")
    df_hidrosina_summary.sort_values(['zona','estado','score_total'],inplace=True)

    try:
        with engine_misions.connect() as connection:
            connection.execute(tabla_summary.delete().where(tabla_summary.c.Month == datetime.now().month))
    except:
        try:
            with engine_misions.connect() as connection:
                connection.execute(tabla_summary.delete().where(tabla_summary.c.Month == datetime.now().month))
        except:
            return jsonify({'Service':False,'Step':2})

    for row in df_hidrosina_summary.itertuples():
        try:
            with engine_misions.connect() as connection:
                connection.execute(tabla_summary.insert().values(Date = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"), Month = datetime.now().month,
                                                                Nombre_Estacion = row.Name, Id_Estacion = row.Name,
                                                                Latitud = row.Latitude, Longitud = row.Longitude,
                                                                Score_Estacion = row.score_estacion, Score_Servicio = row.score_servicio, Score_Total = row.score_total, Total_Encuentas = row.total_encuestas,
                                                                Excelente = row.Excelente, Cumple_protocolo_COVID = row._9, Regular = row.Regular,
                                                                No_cumple_protocolo_COVID = row._11, Decepcionante = row.Decepcionante, Trato_amable = row._13, Limpio = row.Limpio,
                                                                Excelente_actitud = row._15, Grosero = row.Grosero, Comunicacion_clara = row._17, Atencion_rapida = row._18,
                                                                Mala_comunicacion = row._19, Sucio = row.Sucio, Atencion_lenta = row._21, No_Like_Estacion = row.no_like_estacion,
                                                                Like_Estacion = row.like_estacion, No_Like_Servicio = row.no_like_estacion, Like_Servicio = row.like_servicio,
                                                                Espera_1_a_5_minutos = row._26, Espera_6_a_10_minutos = row._27,
                                                                Espera_mas_de_10_minutos = row._28, Estado = row.estado, Zona = row.zona))
        except Exception as e:
            print(e)
            try:
                with engine_misions.connect() as connection:
                    connection.execute(tabla_summary.insert().values(Date = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"), Nombre_Estacion = row.Name, Id_Estacion = row.Name,
                                                                    Latitud = row.Latitude, Longitud = row.Longitude,
                                                                    Score_Estacion = row.score_estacion, Score_Servicio = row.score_servicio, Score_Total = row.score_total, Total_Encuentas = row.total_encuestas,
                                                                    Excelente = row.Excelente, Cumple_protocolo_COVID = row._9, Regular = row.Regular,
                                                                    No_cumple_protocolo_COVID = row._11, Decepcionante = row.Decepcionante, Trato_amable = row._13, Limpio = row.Limpio,
                                                                    Excelente_actitud = row._15, Grosero = row.Grosero, Comunicacion_clara = row._17, Atencion_rapida = row._18,
                                                                    Mala_comunicacion = row._19, Sucio = row.Sucio, Atencion_lenta = row._21, No_Like_Estacion = row.no_like_estacion,
                                                                    Like_Estacion = row.like_estacion, No_Like_Servicio = row.no_like_estacion, Like_Servicio = row.like_servicio,
                                                                    Espera_1_a_5_minutos = row._26, Espera_6_a_10_minutos = row._27,
                                                                    Espera_mas_de_10_minutos = row._28, Estado = row.estado, Zona = row.zona))
            except Exception as e:
                print(e)
                return jsonify({'Service':False,'Step':3})
    
    return jsonify({'Service':True,'Step':4})

@app.route('/hidrosina-tabla-totales', methods=['POST'])
def hidrosina_totales():
    global from_service
    from_service = 'hidro_totatales'

    try:
        with engine_misions.connect() as connection:
            results = connection.execute(db.select([missions_hidrosina])).fetchall()          
    except Exception as e:
        print(e)
        try:
            with engine_misions.connect() as connection:
                results = connection.execute(db.select([missions_hidrosina])).fetchall()
        except Exception as e:
            print(e)
            return jsonify({'Service':False,'Step':1})
    
    df_hidro = pd.DataFrame(results)
    df_hidro.columns = results[0].keys()

    gc = pygsheets.authorize(service_file='images/gchgame-ea9d60803e55.json')
    sh = gc.open_by_url("https://docs.google.com/spreadsheets/d/1EGnt8H7Bwxs_EUCB4hiavwcnkHdMlj0Y4Wr0aD6Wqz0/edit?usp=sharing")
    wks = sh.sheet1
    df_typeform = wks.get_as_df()
    df_typeform = df_typeform.drop_duplicates('capture_id')
    df_joined = pd.merge(df_hidro, df_typeform, left_on='Hash_typeform', right_on='capture_id')
    df = df_joined

    ##Mod preprocesamiento
    df = df.replace(['TRUE','FALSE'],[1,0])
    df['No luce limpia'] = ['No luce limpia' if type(x) != float and 'limpia' in x else np.nan for x in df['¿Qué falta?']]
    df['No esta bien iluminada'] = ['No esta bien iluminada' if type(x) != float and 'iluminada' in x else np.nan for x in df['¿Qué falta?']]
    df['No está el letrero de precios encendido'] = ['No está el letrero de precios encendido' if type(x) != float and 'precios' in x else np.nan for x in df['¿Qué falta?']]

    df['¿Algún despachador te señaló dónde cargar?'] = df['¿Había despachadores indicando el lugar disponible para cargar combustible?']
    df['Los despachadores, ¿traían uniforme limpio, cubrebocas y lentes?'] = df['Los despachadores, ¿traían uniforme limpio, cubrebocas y lentes protectores?']
    df['Encuentra a un trabajador con bata amarilla en la estación. ¿Lo localizaste?'] = df['¿Identificas en la estación un trabajador con bata amarilla?']

    df['uniforme limpio'] = ['uniforme limpio' if type(x) != float and 'limpio' in x else np.nan for x in df['¿Qué no tenía el despachador?']]
    df['cubrebocas'] = ['cubrebocas' if type(x) != float and 'cubrebocas' in x else np.nan for x in df['¿Qué no tenía el despachador?']]
    df['lentes protectores'] = ['lentes protectores' if type(x) != float and 'lentes' in x else np.nan for x in df['¿Qué no tenía el despachador?']]

    df['Cantidad de combustible'] = ['Cantidad de combustible' if type(x) != float and 'Cantidad' in x else np.nan for x in df['¿Qué no te preguntó el despachador?']]
    df['Tipo de combustible'] = ['Tipo de combustible' if type(x) != float and 'Tipo' in x else np.nan for x in df['¿Qué no te preguntó el despachador?']]
    df['Forma de pago'] = ['Forma de pago' if type(x) != float and 'pago' in x else np.nan for x in df['¿Qué no te preguntó el despachador?']]

    df['Producto periférico'] = ['Producto periférico' if type(x) != float and 'Producto' in x else np.nan for x in df['¿Qué no te ofreció?']]
    df['Limpieza de parabrisas'] = ['Limpieza de parabrisas' if type(x) != float and 'Limpieza' in x else np.nan for x in df['¿Qué no te ofreció?']]


    df['conteo']=1

    estado=[]
    zona=[]

    for j in df['Address']:
        e=j.split(re.findall(r'[0-9]+', j)[-1])[-1].strip(" ")
        e=e.strip('(R)').strip(" ")
        estado.append(e)
    
    df['estado']=estado
    df['zona']=np.where(df['Zona']=='Metro','Megalópolis','Provincia')

    #Pregunta 1
    conteos_estado_estacion=pd.crosstab(df['Name'],df['La estación, ¿lucía limpia, bien iluminada y con el letrero de precios encendido?']).reset_index().rename(columns={1: "SiFreq_Estacion_clean_luz_PreciosEncendidos",0: "NoFreq_Estacion_clean_luz_PreciosEncendidos"})

    dk = df.rename(columns = {'No luce limpia':'no_luce_limpia','No esta bien iluminada': 'no_bien_iluminada',
                                'No está el letrero de precios encendido':'No_letreros_precios',
                                'La estación, ¿lucía limpia, bien iluminada y con el letrero de precios encendido?':'pregunta'})

    Ns=Categorizacion_esta(dk)
            
    queLe_falta_a_estacion=pd.crosstab(Ns['Name'], Ns['Que_le_faltaba_aLa_estacion']).reset_index()

    #PRegunta 2
    Despachadores_banderando=pd.crosstab(df['Name'],df['¿Algún despachador te señaló dónde cargar?']).reset_index().rename(columns={1: "SiFreq_Desp_bandereando",0: "NoFreq_Desp_bandereando"})

    #Pregunta 3
    protocolo_despachador=pd.crosstab(df['Name'],df['Los despachadores, ¿traían uniforme limpio, cubrebocas y lentes?']).reset_index().rename(columns={1: "SiFreq_Protocolo_desp",0: "NoFreq_Protocolo_desp"})

    d2k = df.rename(columns = {'uniforme limpio': 'uniforme_limpio','lentes protectores': 'lentes_protectores',
                          ' 	Los despachadores, ¿traían uniforme limpio, cubrebocas y lentes protectores?':'pregunta'})

    Ns=Categorizacion_que_no_llevan(d2k)

    que_no_lleva_despachador=pd.crosstab(Ns['Name'], Ns['que_no_llevaba_despachador']).reset_index()

    #Pregunta 4
    trabajador_bata=pd.crosstab(df['Name'],df['Encuentra a un trabajador con bata amarilla en la estación. ¿Lo localizaste?']).reset_index().rename(columns={1: "SiFreq_trabajador_bata",0: "NoFreq_trabajador_bata"})

    ####    PREGUNTAS RELACIONADAS CON EL SERVICIO 

    #Pregunta 1:
    da_bienvenida_el_despachador=pd.crosstab(df['Name'],df['El despachador ¿dio la bienvenida?']).reset_index().rename(columns={1: "Si_el_desp_da_bienvenida",0: "No_el_desp_da_bienvenida"})

    #Pregunta 2:

    pregunto_combustible_cantidad=pd.crosstab(df['Name'],df['¿Preguntó la cantidad, tipo de combustible a cargar y forma de pago?']).reset_index().rename(columns={1: "Si_Forma_pago_com_pago",0: "No_Forma_pago_com_pago"})
    pregunto_combustible_cantidad_p=pd.crosstab(df['Name'],df['¿Preguntó la cantidad, tipo de combustible a cargar y forma de pago?'],normalize='index').reset_index().rename(columns={1: "Si_Forma_pago_com_pago",0: "No_Forma_pago_com_pago"})

    d2kk = df.rename(columns = {'Cantidad de combustible': 'cantidad_combustible','Tipo de combustible': 'Tipo_combustible',
                                'Forma de pago': 'Forma_pago','¿Preguntó la cantidad, tipo de combustible a cargar y forma de pago?':'pregunta'})

    Ns=Categorizacion_Cantidad_tipo_forma(d2kk)
    Quefalta_que_pregunta=pd.crosstab(Ns['Name'], Ns['Que_no_pregunto']).reset_index()    

    #Pregunta 3
    mostro_bomba_en_cero=pd.crosstab(df['Name'], df['¿Mostró que la bomba estuviera en ceros antes de iniciar la carga?']).reset_index().rename(columns={1: "Si_Mostro_bomba_cero",0: "No_Mostro_bomba_cero"})


    #Pregunta 4
    ofrecio_productos_extras=pd.crosstab(df['Name'], df['¿Ofreció algún producto periférico y limpieza de parabrisas?']).reset_index().rename(columns={1: "Si_Ofrece_Prd_Extras",0: "No_Ofrece_Prd_Extras"})
    ofrecio_productos_extras_p=pd.crosstab(df['Name'], df['¿Ofreció algún producto periférico y limpieza de parabrisas?'],normalize='index').reset_index().rename(columns={1: "Si_Ofrece_Prd_Extras",0: "No_Ofrece_Prd_Extras"})

    d4 = df.rename(columns = {'Producto periférico': 'prod_periferico','Limpieza de parabrisas': 'Limpieza_parabrisas', 
                                '¿Ofreció algún producto periférico y limpieza de parabrisas?': 'pregunta'})

    Rres=Categorizacion_paarabrisa_prdo(d4)
    Que_no_ofrecio=pd.crosstab(Rres['Name'], Rres['que_no_te_ofrecio']).reset_index()

    #Pregunta 5
    ticket_correcto=pd.crosstab(df['Name'], df['¿Entregó el ticket correcto?']).reset_index().rename(columns={1: "Si_ticket_correcto",0: "No_ticket_correcto"})

    #Pregunta 6
    gafete_visble=pd.crosstab(df['Name'], df['¿Tenía puesto el gafete en un lugar visible?']).reset_index().rename(columns={1: "Si_Gafete_visible",0: "No_Gafete_visible"})

    #Pregunta 7
    amableYCordial=pd.crosstab(df['Name'], df['¿Fue amable y cordial al atenderle?']).reset_index().rename(columns={1: "Si_amableYCordial",0: "No_amableYCordial"})

    #Pregunta 8
    Tieempo_De_Espera=pd.crosstab(df['Name'], df['¿Cuánto tiempo estuviste formado para recibir el servicio?']).reset_index()

    cols=['SiFreq_Estacion_clean_luz_PreciosEncendidos',
            'NoFreq_Estacion_clean_luz_PreciosEncendidos',
            'Tiene los tres elementos',
            'No tiene precios iluminados',
            'No está bien iluminada',
            'No letrero de precios y no bien iluminada',
            'No está limpia',
            'No está limpia y no precios iluminados',
            'No está limpia, no bien iluminada',
            'No tiene los tres elementos',
            'SiFreq_Desp_bandereando',
            'NoFreq_Desp_bandereando',
            'SiFreq_Protocolo_desp',
            'NoFreq_Protocolo_desp',
            'Llevan lentes,cubrebocas y uniforme limpio',
            'No_llevan: Lentes protectores',
            'No llevan: Cubrebocas',
            'No traen: Cubrebocas y lentes protectores',
            'No tienen:  Uniforme limpio',
            'No tienen: Uniforme limpio y lentes protectores',
            'No tienen: Uniforme limpio y cubrebocas',
            'No tienen: Ninguno de los tres elementos',
            'SiFreq_trabajador_bata',
            'NoFreq_trabajador_bata',
            'Si_el_desp_da_bienvenida',
            'No_el_desp_da_bienvenida',
            'Si_Forma_pago_com_pago',
            'No_Forma_pago_com_pago',
            'Pregunto las tres opciones: Cantidad, tipo y forma',
            'No preguntó Forma de pago',
            'No preguntó Tipo de combustible',
            'No preguntó Tipo de combustible y forma de pago',
            'No preguntó Cantidad de combustible',
            'No preguntó Cantidad de combustible y forma de pago',
            'No preguntó Cantidad y tipo de combustible',
            'No preguntó :Cantidad, tipo de combustible y forma de pago',
            'Si_Mostro_bomba_cero',
            'No_Mostro_bomba_cero',
            'Si_Ofrece_Prd_Extras',
            'No_Ofrece_Prd_Extras',
            'Ofreció las dos opciones, limpieza y productos',
            'No ofrecio Limpieza de parabrisas',
            'No ofrecio Producto periférico',
            'No ofrecio Limpieza de parabrisas y Producto periférico',
            'Si_ticket_correcto',
            'No_ticket_correcto',
            'Si_Gafete_visible',
            'No_Gafete_visible',
            'Si_amableYCordial',
            'No_amableYCordial',
            '1 a 5 minutos',
            '6 a 10 minutos',
            'Más de 10 minutos',
            3.0
            ]
    
    #Tabla numeros
    aux=pd.DataFrame(columns = ['Name','estado','zona']+cols) 
    aux['Name'] = df['Name']
    aux['estado'] = df['estado']
    aux['zona'] = df['zona']
    aux.drop_duplicates(subset=['Name','estado','zona'],inplace=True)

    aux=conteos_estado_estacion.set_index('Name').combine_first(aux.set_index('Name')).reset_index()
    aux=queLe_falta_a_estacion.set_index('Name').combine_first(aux.set_index('Name')).reset_index()
    aux=Despachadores_banderando.set_index('Name').combine_first(aux.set_index('Name')).reset_index()
    aux=protocolo_despachador.set_index('Name').combine_first(aux.set_index('Name')).reset_index()
    aux=trabajador_bata.set_index('Name').combine_first(aux.set_index('Name')).reset_index()
    aux=que_no_lleva_despachador.set_index('Name').combine_first(aux.set_index('Name')).reset_index()
    aux=da_bienvenida_el_despachador.set_index('Name').combine_first(aux.set_index('Name')).reset_index()
    aux=pregunto_combustible_cantidad.set_index('Name').combine_first(aux.set_index('Name')).reset_index()
    aux=Quefalta_que_pregunta.set_index('Name').combine_first(aux.set_index('Name')).reset_index()
    aux=mostro_bomba_en_cero.set_index('Name').combine_first(aux.set_index('Name')).reset_index()
    aux=ofrecio_productos_extras.set_index('Name').combine_first(aux.set_index('Name')).reset_index()
    aux=Que_no_ofrecio.set_index('Name').combine_first(aux.set_index('Name')).reset_index()
    aux=ticket_correcto.set_index('Name').combine_first(aux.set_index('Name')).reset_index()
    aux=gafete_visble.set_index('Name').combine_first(aux.set_index('Name')).reset_index()
    aux=amableYCordial.set_index('Name').combine_first(aux.set_index('Name')).reset_index()
    aux=Tieempo_De_Espera.set_index('Name').combine_first(aux.set_index('Name')).reset_index()


    aux.sort_values(['zona','estado'],inplace=True)
    totales = aux[['Name','estado','zona']+cols]
    totales = totales.fillna(0)

    try:
        with engine_misions.connect() as connection:
            connection.execute(tabla_preguntas.delete().where(tabla_preguntas.c.Month == datetime.now().month))
    except:
        try:
            with engine_misions.connect() as connection:
                connection.execute(tabla_preguntas.delete().where(tabla_preguntas.c.Month == datetime.now().month))
        except:
            return jsonify({'Service':False,'Step':2})

    for row in totales.itertuples():
        try:
            with engine_misions.connect() as connection:
                connection.execute(tabla_preguntas.insert().values(Date = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"), Month = datetime.now().month,
                                                                    Name = row.Name, estado = row.estado, zona = row.zona, SiFreq_Estacion_clean_luz_PreciosEncendidos = row.SiFreq_Estacion_clean_luz_PreciosEncendidos,
                                                                    NoFreq_Estacion_clean_luz_PreciosEncendidos = row.NoFreq_Estacion_clean_luz_PreciosEncendidos,
                                                                    Tiene_los_tres_elementos = row._6, No_tiene_precios_iluminados = row._7,
                                                                    No_esta_bien_iluminada = row._8, No_letrero_de_precios_y_no_bien_iluminada = row._9,
                                                                    No_esta_limpia = row._10, No_esta_limpia_y_no_precios_iluminados = row._11,
                                                                    No_esta_limpia_no_bien_iluminada = row._12, No_tiene_los_tres_elementos = row._13,
                                                                    SiFreq_Desp_bandereando = row.SiFreq_Desp_bandereando, NoFreq_Desp_bandereando = row.NoFreq_Desp_bandereando,
                                                                    SiFreq_Protocolo_desp = row.SiFreq_Protocolo_desp, NoFreq_Protocolo_desp = row.NoFreq_Protocolo_desp,
                                                                    Llevan_lentes_cubrebocas_y_uniforme_limpio = row._18,
                                                                    No_llevan_Lentes_protectores = row._19, No_llevan_Cubrebocas = row._20,
                                                                    No_traen_Cubrebocas_y_lentes_protectores = row._21, No_tienen_Uniforme_limpio = row._22,
                                                                    No_tienen_Uniforme_limpio_y_lentes_protectores = row._23,
                                                                    No_tienen_Uniforme_limpio_y_cubrebocas = row._24,
                                                                    No_tienen_Ninguno_de_los_tres_elementos = row._25, SiFreq_trabajador_bata = row.SiFreq_trabajador_bata,
                                                                    NoFreq_trabajador_bata = row.NoFreq_trabajador_bata, Si_el_desp_da_bienvenida = row.Si_el_desp_da_bienvenida,
                                                                    No_el_desp_da_bienvenida = row.No_el_desp_da_bienvenida, Si_Forma_pago_com_pago = row.Si_Forma_pago_com_pago,
                                                                    No_Forma_pago_com_pago = row.No_Forma_pago_com_pago,
                                                                    Pregunto_las_tres_opciones_Cantidad_tipo_y_forma = row._32,
                                                                    No_pregunto_Forma_de_pago = row._33, No_pregunto_Tipo_de_combustible = row._34,
                                                                    No_pregunto_Tipo_de_combustible_y_forma_de_pago = row._35,
                                                                    No_pregunto_Cantidad_de_combustible = row._36,
                                                                    No_pregunto_Cantidad_de_combustible_y_forma_de_pago = row._37,
                                                                    No_pregunto_Cantidad_y_tipo_de_combustible = row._38,
                                                                    No_pregunto_Cantidad_tipo_de_combustible_y_forma_de_pago = row._39,
                                                                    Si_Mostro_bomba_cero = row.Si_Mostro_bomba_cero, No_Mostro_bomba_cero = row.No_Mostro_bomba_cero, Si_Ofrece_Prd_Extras = row.Si_Ofrece_Prd_Extras,
                                                                    No_Ofrece_Prd_Extras = row.No_Ofrece_Prd_Extras, Ofrecio_las_dos_opciones_limpieza_y_productos = row._44,
                                                                    No_ofrecio_Limpieza_de_parabrisas = row._45, No_ofrecio_Producto_periferico = row._46,
                                                                    No_ofrecio_Limpieza_de_parabrisas_y_Producto_periferico = row._47,
                                                                    Si_ticket_correcto = row.Si_ticket_correcto, No_ticket_correcto = row.No_ticket_correcto, Si_Gafete_visible = row.Si_Gafete_visible,
                                                                    No_Gafete_visible = row.No_Gafete_visible, Si_amableYCordial = row.Si_amableYCordial, No_amableYCordial = row.No_amableYCordial,
                                                                    espera_1_a_5_minutos = row._54, espera_6_a_10_minutos = row._55, Mas_de_10_minutos = row._56))
        except Exception as e:
            print(e)
            try:
                with engine_misions.connect() as connection:
                    connection.execute(tabla_preguntas.insert().values(Date = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"), Month = datetime.now().month,
                                                                        Name = row.Name, estado = row.estado, zona = row.zona, SiFreq_Estacion_clean_luz_PreciosEncendidos = row.SiFreq_Estacion_clean_luz_PreciosEncendidos,
                                                                        NoFreq_Estacion_clean_luz_PreciosEncendidos = row.NoFreq_Estacion_clean_luz_PreciosEncendidos,
                                                                        Tiene_los_tres_elementos = row._6, No_tiene_precios_iluminados = row._7,
                                                                        No_esta_bien_iluminada = row._8, No_letrero_de_precios_y_no_bien_iluminada = row._9,
                                                                        No_esta_limpia = row._10, No_esta_limpia_y_no_precios_iluminados = row._11,
                                                                        No_esta_limpia_no_bien_iluminada = row._12, No_tiene_los_tres_elementos = row._13,
                                                                        SiFreq_Desp_bandereando = row.SiFreq_Desp_bandereando, NoFreq_Desp_bandereando = row.NoFreq_Desp_bandereando,
                                                                        SiFreq_Protocolo_desp = row.SiFreq_Protocolo_desp, NoFreq_Protocolo_desp = row.NoFreq_Protocolo_desp,
                                                                        Llevan_lentes_cubrebocas_y_uniforme_limpio = row._18,
                                                                        No_llevan_Lentes_protectores = row._19, No_llevan_Cubrebocas = row._20,
                                                                        No_traen_Cubrebocas_y_lentes_protectores = row._21, No_tienen_Uniforme_limpio = row._22,
                                                                        No_tienen_Uniforme_limpio_y_lentes_protectores = row._23,
                                                                        No_tienen_Uniforme_limpio_y_cubrebocas = row._24,
                                                                        No_tienen_Ninguno_de_los_tres_elementos = row._25, SiFreq_trabajador_bata = row.SiFreq_trabajador_bata,
                                                                        NoFreq_trabajador_bata = row.NoFreq_trabajador_bata, Si_el_desp_da_bienvenida = row.Si_el_desp_da_bienvenida,
                                                                        No_el_desp_da_bienvenida = row.No_el_desp_da_bienvenida, Si_Forma_pago_com_pago = row.Si_Forma_pago_com_pago,
                                                                        No_Forma_pago_com_pago = row.No_Forma_pago_com_pago,
                                                                        Pregunto_las_tres_opciones_Cantidad_tipo_y_forma = row._32,
                                                                        No_pregunto_Forma_de_pago = row._33, No_pregunto_Tipo_de_combustible = row._34,
                                                                        No_pregunto_Tipo_de_combustible_y_forma_de_pago = row._35,
                                                                        No_pregunto_Cantidad_de_combustible = row._36,
                                                                        No_pregunto_Cantidad_de_combustible_y_forma_de_pago = row._37,
                                                                        No_pregunto_Cantidad_y_tipo_de_combustible = row._38,
                                                                        No_pregunto_Cantidad_tipo_de_combustible_y_forma_de_pago = row._39,
                                                                        Si_Mostro_bomba_cero = row.Si_Mostro_bomba_cero, No_Mostro_bomba_cero = row.No_Mostro_bomba_cero, Si_Ofrece_Prd_Extras = row.Si_Ofrece_Prd_Extras,
                                                                        No_Ofrece_Prd_Extras = row.No_Ofrece_Prd_Extras, Ofrecio_las_dos_opciones_limpieza_y_productos = row._44,
                                                                        No_ofrecio_Limpieza_de_parabrisas = row._45, No_ofrecio_Producto_periferico = row._46,
                                                                        No_ofrecio_Limpieza_de_parabrisas_y_Producto_periferico = row._47,
                                                                        Si_ticket_correcto = row.Si_ticket_correcto, No_ticket_correcto = row.No_ticket_correcto, Si_Gafete_visible = row.Si_Gafete_visible,
                                                                        No_Gafete_visible = row.No_Gafete_visible, Si_amableYCordial = row.Si_amableYCordial, No_amableYCordial = row.No_amableYCordial,
                                                                        espera_1_a_5_minutos = row._54, espera_6_a_10_minutos = row._55, Mas_de_10_minutos = row._56))
            except Exception as e:
                print(e)
                return jsonify({'Service':False,'Step':3})

    return jsonify({'Service':True,'Step':4})

@app.route('/hidrosina-map', methods=['GET'])
def hidrosina_map():
    global from_service
    from_service = 'get'
    lat_new_marker = request.args.get('lat')
    lng_new_marker = request.args.get('long')
    try:
        with engine_misions.connect() as connection:
            results = connection.execute(db.select([missions_hidrosina])).fetchall()
    except Exception as e:
        print(e)
        try:
            with engine_misions.connect() as connection:
                results = connection.execute(db.select([missions_hidrosina])).fetchall()
        except Exception as e:
            print(e)
            return render_template('hidrosina_map.html')

    if len(results) == 0:
        m = folium.Map(location=[23.5929292,-102.3634602], zoom_start=6)
        m.save('templates/hidrosina_map.html')
        return m._repr_html_()

    df = pd.DataFrame(results)
    df.columns = results[0].keys()
    df_hidro = df

    m = folium.Map(location=[23.5929292,-102.3634602], zoom_start=6)
    mc = MarkerCluster()

    if lat_new_marker != None and lng_new_marker != None:
        popup_test = 'Hola, atrás de ti'
        tooltip_test = 'Da clic para saber más'
        folium.Marker([lat_new_marker,lng_new_marker],
                                popup=popup_test,
                                tooltip=tooltip_test).add_to(mc)

    tooltip = 'Da click para ver los horarios Disponibles'
    row_count = 0
    popup_list = []
    for row in df_hidro.itertuples():
        row_count += 1

        if row.Flag == 'No':
            popup_list.append(row.Horario_Inf_Timestamp)
            
        if row_count == 2:
            row_count = 0
            if popup_list != []:
                if len(popup_list) == 2:
                    popup = '<strong>Matutino y Vespertino</strong>'
                elif popup_list == [21600]:
                    popup = '<strong>Matutino</strong>'
                else:
                    popup = '<strong>Vespertino</strong>'

                # Create markers
                logo_gas = folium.features.CustomIcon('images/gasolinera.png', icon_size=(25, 25))
                folium.Marker([row.Latitude,row.Longitude],
                                popup=popup,
                                tooltip= 'HD-' + row.Name + ' ' + tooltip,
                                icon = logo_gas).add_to(mc)
                popup_list = []
            else:
                continue
        else:
            continue

    m.add_child(mc)
    LocateControl(auto_start=True,strings={'title': 'Ve tu ubicación actual','popup':'Estás aquí'}).add_to(m)
    m.save('templates/hidrosina_map.html')

    return m._repr_html_()

@app.route('/santory', methods=['POST'])
def santory_service():
    global data
    global json_respuesta
    global bandera
    global from_service

    from_service = 'Santory'

    data = request.json
    bandera = request.args.get('re_data')
    video_path = ''

    if bandera != 'yes':
        data['url'] = orientation_fix_function(data['url'])

    elif bandera != None:
        print('Error de request')
        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':'Ocurrió un error inesperado con esta misión, por favor repórtalo a Soporte Gotchu!'}
        return jsonify(json_respuesta)
    else:
        pass

    try:
        with engine_misions.connect() as connection:
            results = connection.execute(db.select([missions_santory]).where(missions_santory.c.Flag != 'Yes')).fetchall()
    except Exception as e:
        print(e)
        try:
            with engine_misions.connect() as connection:
                results = connection.execute(db.select([missions_santory]).where(missions_santory.c.Flag != 'Yes')).fetchall()
        except Exception as e:
            print(e)
            json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':'Ocurrió un error inesperado, vuelve a intentarlo por favor'}
            mission_message = message_pay_service(json_respuesta)
            json_respuesta['msg'] = mission_message
            return jsonify(json_respuesta)

    if len(results) == 0:
        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':'Lo sentimos, ya no hay ubicaciones disponibles para esta misión'}
        return jsonify(json_respuesta)

    df = pd.DataFrame(results)
    df.columns = results[0].keys()
    lat , lng, address, ids = df['Latitude'].tolist(), df['Longitude'].tolist(), df['Address'].tolist(), df['Id'].tolist()
    user_pos = (data['Location_latitude'],data['Location_longitude']) 
    
    image_path = data['url']

    distancias = []
    for t, g in zip(lat,lng):
        distancias.append(geodesic(user_pos,(t,g)).meters)
    if min(distancias) < 200:
        
        start_date = datetime.strptime(data['Start_Date_mission'], '%Y-%m-%d %H:%M:%S.%f')
        end_date = datetime.strptime(data['End_Date_mission'], '%Y-%m-%d %H:%M:%S.%f')
        user_time = (end_date - start_date).total_seconds()
        mission_target_time = data['Target_time_mission']

        if (user_time<=mission_target_time):
            if liveness(image_path):
                j = np.argmin(distancias)
                direc = address[j]
                id_tienda = ids[j]
                data['Location_mission_latitude'] = lat[j]
                data['Location_mission_longitude'] = lng[j]
                user_id = data['id']
                user_lat = user_pos[0]
                user_lng = user_pos[1]
                user_id = data['id']
                miss_id = data['id_mission']
                hash_typeform = data['hash_typeform']

                try:
                    with engine_misions.connect() as connection:
                        connection.execute(missions_santory.update().where(missions_santory.c.Id == id_tienda).values(Flag = 'Yes',
                            Date = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
                            User_Id = user_id, User_Latitude = user_lat, User_Longitude = user_lng, Url_Photo = image_path,
                            Mission_Id = miss_id, Hash_typeform = hash_typeform))
                except Exception as e:
                    print(e)
                    try:
                        with engine_misions.connect() as connection:
                            connection.execute(missions_santory.update().where(missions_santory.c.Id == id_tienda).values(Flag = 'Yes',
                                Date = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
                                User_Id = user_id, User_Latitude = user_lat, User_Longitude = user_lng, Url_Photo = image_path,
                                Mission_Id = miss_id, Hash_typeform = hash_typeform))
                    except Exception as e:
                        print(e)
                        return jsonify({'Service':False})
                
                try:

                    url = "https://us-central1-gchgame.cloudfunctions.net/mail-sender"

                    payload = {'id_tienda':id_tienda,'message':body_santory,'service':from_service,'subject':'Santory - NUEVA MISION'}
                    headers = {'Content-Type': 'application/json'}

                    response = requests.request("POST", url, headers=headers, data = json.dumps(payload))

                except Exception as e:
                    print(e)
                    pass
                json_respuesta = {'Location':True,'Time':True,'Service':True,'Live':True,'Porn':False,'Id':0,'Url_themask':image_path,'url_thumbnail':'','msg':''}
                json_respuesta['msg'] = '¡Misión completada! Felicitaciones Agente'
                return jsonify(json_respuesta)

            else:
                json_respuesta = {'Location':True,'Time':True,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':''}
                mission_message = message_pay_service(json_respuesta)
                json_respuesta['msg'] = mission_message
                return jsonify(json_respuesta)
        else:
            json_respuesta = {'Location':True,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':''}
            mission_message = message_pay_service(json_respuesta)
            json_respuesta['msg'] = mission_message
            return jsonify(json_respuesta)
    else:
        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':''}
        mission_message = message_pay_service(json_respuesta)
        json_respuesta['msg'] = mission_message
        return jsonify(json_respuesta)

@app.route('/taifelds-disfruta', methods=['POST'])
def taifelds_disfruta_service():
    global data
    global json_respuesta
    global bandera
    global from_service
    global video_path
    global masked_url
    global url2json0
    global Service

    from_service = 'Taifelds-disfruta'

    data = request.json
    bandera = request.args.get('re_data')
    video_path = ''
    extras = ['persona','selfie','cara']

    if bandera != 'yes':
        data['url'] = orientation_fix_function(data['url'])
    
    url2json0 = ['']
    masked_url = [data['url']]
    Service = [False,False,False,False]

    if bandera == 'yes':
        try:
            with engine_misions.connect() as connection:
                results = connection.execute(db.select([missions_taifelds_disfruta]).where(missions_taifelds_disfruta.c.Flag == 'Pending')).fetchall()
        except Exception as e:
            print(e)
            try:
                with engine_misions.connect() as connection:
                    results = connection.execute(db.select([missions_taifelds_disfruta]).where(missions_taifelds_disfruta.c.Flag == 'Pending')).fetchall()
            except Exception as e:
                print(e)
                return jsonify({'Service':False})

        if len(results) == 0:
            return jsonify({'Service':False})

        df = pd.DataFrame(results)
        df.columns = results[0].keys()
        ids = df['Id'].tolist()
        id_registro_salida = data['Id_Store']
        if id_registro_salida not in ids:
            return jsonify({'Service':False})

        status = data['Status']

        try:
            with engine_misions.connect() as connection:
                connection.execute(missions_taifelds_disfruta.update().where(missions_taifelds_disfruta.c.Id == id_registro_salida).values(Flag = status))
        except Exception as e:
            print(e)
            try:
                with engine_misions.connect() as connection:
                    connection.execute(missions_taifelds_disfruta.update().where(missions_taifelds_disfruta.c.Id == id_registro_salida).values(Flag = status))
            except Exception as e:
                print(e)
                return jsonify({'Service':False})

        
        return jsonify({'Service':True})

    elif bandera != None:
        print('Error de request')
        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':''}
        return jsonify(json_respuesta)
    else:
        pass


    try:
        with engine_misions.connect() as connection:
            results = connection.execute(db.select([missions_taifelds_disfruta]).where(missions_taifelds_disfruta.c.Flag == 'Yes')).fetchall()
    except Exception as e:
        print(e)
        try:
            with engine_misions.connect() as connection:
                results = connection.execute(db.select([missions_taifelds_disfruta]).where(missions_taifelds_disfruta.c.Flag == 'Yes')).fetchall()
        except Exception as e:
            print(e)
            json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':''}
            return jsonify(json_respuesta)
    
    if len(results) == 0:
        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':''}
        return jsonify(json_respuesta)
    df = pd.DataFrame(results)
    df.columns = results[0].keys()
    lat, lng, ids, user_ids = df['User_Latitude'].tolist(), df['User_Longitude'].tolist(), df['Id'].tolist(), df['User_Id'].tolist()
    user_id = data['id']
    if user_id in user_ids:
        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':''}
        return jsonify(json_respuesta)
    user_pos = (data['Location_latitude'],data['Location_longitude'])

    image_path = data['url']
    try:
        video_path = data['Url_Video']
    except KeyError as e:
        video_path = ''
        print(e)
    
    distancias = []
    for t, g in zip(lat,lng):
        distancias.append(geodesic(user_pos,(t,g)).meters)
    if min(distancias) > 50:
        
        start_date = datetime.strptime(data['Start_Date_mission'], '%Y-%m-%d %H:%M:%S.%f')
        end_date = datetime.strptime(data['End_Date_mission'], '%Y-%m-%d %H:%M:%S.%f')
        user_time = (end_date - start_date).total_seconds()
        mission_target_time = data['Target_time_mission']

        if (user_time<=mission_target_time):
            if liveness(image_path):
                user_lat = data['Location_latitude']
                user_lng = data['Location_longitude']
                url_p = data['url']
                url_v = data['Url_Video']
                miss_id = data['id_mission']
                texto_td = data['text']
                try:
                    with engine_misions.connect() as connection:
                        connection.execute(missions_taifelds_disfruta.insert().values(
                            Date = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
                            User_Id = user_id,
                            User_Latitude = user_lat,
                            User_Longitude = user_lng,
                            Url_Photo = url_p,
                            Url_Video = url_v,
                            Text = texto_td,
                            Mission_Id = miss_id,
                            Flag = 'Pending'))
                        final_results = connection.execute(db.select([missions_taifelds_disfruta]).where(missions_taifelds_disfruta.c.User_Id == user_id)).fetchall()
                        final_df = pd.DataFrame(final_results)
                        final_df.columns = final_results[0].keys()
                        final_ids = final_df['Id'].tolist()
                        id_registro = final_ids[-1]
                except Exception as e:
                    print(e)
                    try:
                        with engine_misions.connect() as connection:
                            connection.execute(missions_taifelds_disfruta.insert().values(
                                Date = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
                                User_Id = user_id,
                                User_Latitude = user_lat,
                                User_Longitude = user_lng,
                                Url_Photo = url_p,
                                Url_Video = url_v,
                                Text = texto_td,
                                Mission_Id = miss_id,
                                Flag = 'Pending'))
                            final_results = connection.execute(db.select([missions_taifelds_disfruta]).where(missions_taifelds_disfruta.c.User_Id == user_id)).fetchall()
                            final_df = pd.DataFrame(final_results)
                            final_df.columns = final_results[0].keys()
                            final_ids = final_df['Id'].tolist()
                            id_registro = final_ids[-1]
                    except Exception as e:
                        print(e)
                        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':''}
                        return jsonify(json_respuesta)
                
                try:

                    url = "https://us-central1-gchgame.cloudfunctions.net/mail-sender"

                    payload = {'id_tienda':id_registro,'message':body_taifelds_disfruta,'service':from_service,'subject':'Disfruta y Gana - NUEVA MISION'}
                    headers = {'Content-Type': 'application/json'}

                    response = requests.request("POST", url, headers=headers, data = json.dumps(payload))

                except Exception as e:
                    print(e)
                    pass
                
                detect_human(image_path,'selfie',extras)
                json_respuesta = {'Location':True,'Time':True,'Service':True,'Live':True,'Porn':False,'Id':id_registro,'Url_themask':imagen_final(masked_url[0]),'url_thumbnail':video_to_thumbnail_url(video_path),'msg':''}
                return jsonify(json_respuesta)

            else:
                json_respuesta = {'Location':True,'Time':True,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':''}
                return jsonify(json_respuesta)
        else:
            json_respuesta = {'Location':True,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':''}
            return jsonify(json_respuesta)
    else:
        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':''}
        return jsonify(json_respuesta)

@app.route('/covid', methods=['POST'])
def covid_service():
    global data
    global json_respuesta
    global bandera
    global from_service
    global video_path

    from_service = 'Covid'

    data = request.json
    bandera = request.args.get('re_data')
    video_path = ''

    if bandera != 'yes':
        data['url'] = orientation_fix_function(data['url'])

    if bandera == 'yes':
        try:
            with engine_misions.connect() as connection:
                results = connection.execute(db.select([missions_covid]).where(missions_covid.c.Flag == 'Pending')).fetchall()
        except Exception as e:
            print(e)
            try:
                with engine_misions.connect() as connection:
                    results = connection.execute(db.select([missions_covid]).where(missions_covid.c.Flag == 'Pending')).fetchall()
            except Exception as e:
                print(e)
                return jsonify({'Service':False})
        if len(results) == 0:
            return jsonify({'Service':False})
        df = pd.DataFrame(results)
        df.columns = results[0].keys()
        lat , lng, address, ids = df['Latitude'].tolist(), df['Longitude'].tolist(), df['Address'].tolist(), df['Id'].tolist()

        id_tienda_salida = data['Id_Store']
        if id_tienda_salida not in ids:
            return jsonify({'Service':False})

        status = data['Status']
        user_id = data['User_Id']
        user_lat = data['User_Latitude']
        user_lng = data['User_Longitude']
        url_p = data['Url_Photo']
        url_v = data['Url_Video']
        video_path = data['Url_Video']
        miss_id = data['Mission_Id']

        try:
            with engine_misions.connect() as connection:
                connection.execute(missions_covid.update().where(missions_covid.c.Id == id_tienda_salida).values(Flag = status,
                    Date = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
                    User_Id = user_id, User_Latitude = user_lat, User_Longitude = user_lng, Url_Photo = url_p,
                    Url_Video = url_v, Mission_Id = miss_id))
        except Exception as e:
            print(e)
            try:
                with engine_misions.connect() as connection:
                    connection.execute(missions_covid.update().where(missions_covid.c.Id == id_tienda_salida).values(Flag = status,
                        Date = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
                        User_Id = user_id, User_Latitude = user_lat, User_Longitude = user_lng, Url_Photo = url_p,
                        Url_Video = url_v, Mission_Id = miss_id))
            except Exception as e:
                print(e)
                return jsonify({'Service':False})

        
        return jsonify({'Service':True})

    elif bandera != None:
        print('Error de request')
        json_respuesta = {'Location':False,'Time':False,'Service':False,'Porn':False,'Id':0,'url_thumbnail':'','msg':''}
        return jsonify(json_respuesta)
    else:
        pass

    try:
        with engine_misions.connect() as connection:
            results = connection.execute(db.select([missions_covid]).where(missions_covid.c.Flag != 'Yes')).fetchall()
    except Exception as e:
        print(e)
        try:
            with engine_misions.connect() as connection:
                results = connection.execute(db.select([missions_covid]).where(missions_covid.c.Flag != 'Yes')).fetchall()
        except Exception as e:
            print(e)
            json_respuesta = {'Location':False,'Time':False,'Service':False,'Porn':False,'Id':0,'url_thumbnail':'','msg':''}
            return jsonify(json_respuesta)
    if len(results) == 0:
        json_respuesta = {'Location':False,'Time':False,'Service':False,'Porn':False,'Id':0,'url_thumbnail':'','msg':''}
        return jsonify(json_respuesta)
    df = pd.DataFrame(results)
    df.columns = results[0].keys()
    lat , lng, address, ids = df['Latitude'].tolist(), df['Longitude'].tolist(), df['Address'].tolist(), df['Id'].tolist()
    user_pos = (data['Location_latitude'],data['Location_longitude']) 
    

    image_path = data['url']
    distancias = []
    for t, g in zip(lat,lng):
        distancias.append(geodesic(user_pos,(t,g)).meters)
    if min(distancias) < 500:
        
        start_date = datetime.strptime(data['Start_Date_mission'], '%Y-%m-%d %H:%M:%S.%f')
        end_date = datetime.strptime(data['End_Date_mission'], '%Y-%m-%d %H:%M:%S.%f')
        user_time = (end_date - start_date).total_seconds()
        mission_target_time = data['Target_time_mission']

        if (user_time<=mission_target_time):
            if liveness(image_path):
                j = np.argmin(distancias)
                direc = address[j]
                id_tienda = ids[j]
                data['Location_mission_latitude'] = lat[j]
                data['Location_mission_longitude'] = lng[j]
                try:
                    with engine_misions.connect() as connection:
                        connection.execute(missions_covid.update().where(missions_covid.c.Address == direc).values(Flag = 'Pending'))
                except Exception as e:
                    print(e)
                    try:
                        with engine_misions.connect() as connection:
                            connection.execute(missions_covid.update().where(missions_covid.c.Address == direc).values(Flag = 'Pending'))
                    except Exception as e:
                        print(e)
                        json_respuesta = {'Location':False,'Time':False,'Service':False,'Porn':False,'Id':0,'url_thumbnail':'','msg':''}
                        return jsonify(json_respuesta)
                
                try:
                    url = "https://us-central1-gchgame.cloudfunctions.net/mail-sender"

                    payload = {'id_tienda':id_tienda,'message':body_covid,'service':from_service,'subject':'COVID - NUEVA MISION'}
                    headers = {'Content-Type': 'application/json'}

                    response = requests.request("POST", url, headers=headers, data = json.dumps(payload))
                except Exception as e:
                    print(e)
                    pass
                json_respuesta = {'Location':True,'Time':True,'Service':True,'Live':True,'Porn':False,'Id':id_tienda,'url_thumbnail':video_to_thumbnail_url(video_path),'msg':''}
                return jsonify(json_respuesta)
            else:
                json_respuesta = {'Location':True,'Time':True,'Service':False,'Live':False,'Porn':False,'Id':0,'url_thumbnail':'','msg':''}
                return jsonify(json_respuesta)
        else:
            json_respuesta = {'Location':True,'Time':False,'Service':False,'Live':True,'Porn':False,'Id':0,'url_thumbnail':'','msg':''}
            return jsonify(json_respuesta)
    else:
        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':True,'Porn':False,'Id':0,'url_thumbnail':'','msg':''}
        return jsonify(json_respuesta)

@app.route('/taifelds-map', methods=['GET'])
def taifelds_map():
    global from_service
    from_service = 'get'
    return render_template('index.html')

@app.route('/taifelds-cul-map', methods=['GET'])
def taifelds_cul_map():
    global from_service
    from_service = 'get'
    return render_template('index_cul.html')

@app.route('/money-locator',methods=['POST'])
def money_locator():
    global from_service
    from_service = 'Money-locator'
    data = request.json
    metodo = data['method'] #Taifelds Hidrosina General
    hidrosina_flag = metodo == 'Hidrosina'
    taifelds_flag = metodo == 'Taifelds'
    general_flag = metodo == 'General'
    user_pos = (data['Location_latitude'],data['Location_longitude'])

    if hidrosina_flag or general_flag:
        try:
            with engine_misions.connect() as connection:
                results_hidrosina = connection.execute(db.select([missions_hidrosina]).where(missions_hidrosina.c.Flag != 'Yes')).fetchall()
        except Exception as e:
            print(e)
            try:
                with engine_misions.connect() as connection:
                    results_hidrosina = connection.execute(db.select([missions_hidrosina]).where(missions_hidrosina.c.Flag != 'Yes')).fetchall()
            except Exception as e:
                print(e)
                json_respuesta = {'Service':False,'msg':'Error de conexión, inténtalo nuevamente','Latitude':0,'Longitude':0}
                return jsonify(json_respuesta)

        if len(results_hidrosina) != 0:
            df = pd.DataFrame(results_hidrosina)
            df.columns = results_hidrosina[0].keys()
            lat_hidrosina, lng_hidrosina = df['Latitude'].tolist(), df['Longitude'].tolist()
        else:
            lat_hidrosina, lng_hidrosina = [], []
    else:
        lat_hidrosina, lng_hidrosina = [], []


    if taifelds_flag or general_flag:
        try:
            with engine_misions.connect() as connection:
                results_taifelds2 = connection.execute(db.select([missions_taifelds2]).where(missions_taifelds2.c.Flag != 'Yes')).fetchall()
                results_taifelds = connection.execute(db.select([missions_taifelds]).where(missions_taifelds.c.Flag != 'Yes')).fetchall()
        except Exception as e:
            print(e)
            try:
                with engine_misions.connect() as connection:
                    results_taifelds2 = connection.execute(db.select([missions_taifelds2]).where(missions_taifelds2.c.Flag != 'Yes')).fetchall()
                    results_taifelds = connection.execute(db.select([missions_taifelds]).where(missions_taifelds.c.Flag != 'Yes')).fetchall()
            except Exception as e:
                print(e)
                json_respuesta = {'Service':False,'msg':'Error de conexión, vuelve a intentarlo por favor','Latitude':0,'Longitude':0}
                return jsonify(json_respuesta)

        if len(results_taifelds2) != 0:
            df = pd.DataFrame(results_taifelds2)
            df.columns = results_taifelds2[0].keys()
            lat_taifelds2, lng_taifelds2 = df['Latitude'].tolist(), df['Longitude'].tolist()
        else:
            lat_taifelds2, lng_taifelds2 = [], []

        if len(results_taifelds) != 0:
            df = pd.DataFrame(results_taifelds)
            df.columns = results_taifelds[0].keys()
            lat_taifelds, lng_taifelds = df['Latitude'].tolist(), df['Longitude'].tolist()
        else:
            lat_taifelds, lng_taifelds = [], []

    else:
        lat_taifelds2, lng_taifelds2 = [], []
        lat_taifelds, lng_taifelds = [], []

    lat_taifelds.extend(lat_taifelds2)
    lat_hidrosina.extend(lat_taifelds)
    lat = lat_hidrosina
    if len(lat) == 0:
        json_respuesta = {'Service':False,'msg':'Lo sentimos, ya no hay ubicaciones disponibles para ganar dinero en esta temporada, tendremos más pronto','Latitude':0,'Longitude':0}
        return jsonify(json_respuesta)
    lng_taifelds.extend(lng_taifelds2)
    lng_hidrosina.extend(lng_taifelds)
    lng = lng_hidrosina

    distancias = []
    for t, g in zip(lat,lng):
        distancias.append(geodesic(user_pos,(t,g)).meters)
    j = np.argmin(distancias)
    lat_out = lat[j]
    lng_out = lng[j]

    json_respuesta = {'Service':True,'msg':'Listo Agente, la siguiente ubicación para ganar dinero te espera...','Latitude':str(lat_out),'Longitude':str(lng_out)}
    return jsonify(json_respuesta)

@app.route('/imagen-to-texto', methods=['POST'])
def img2text():
    global from_service
    from_service = 'Texto'
    data = request.json
    image_path = data['url']
    texto = detect_text_uri_demo(image_path)
    json_respuesta = {'Texto':texto}
    return jsonify(json_respuesta)

@app.route('/imagen-qr-to-texto', methods=['POST'])
def imgqr2text():
    global from_service
    from_service = 'QR'
    data = request.json
    image_path = data['url']
    texto_qr = url_qr_2_text(image_path)
    json_respuesta = {'Texto':texto_qr}
    return jsonify(json_respuesta)

CORS(app)
@app.route('/coords-to-address', methods=['POST'])
def coords2address():
    global from_service
    from_service = 'coords2address'
    data = request.json

    i,j = data['Location_latitude'],data['Location_longitude']
    try:
        direccion_dict = gmaps.reverse_geocode((i,j))
        direccion = direccion_dict[0]['formatted_address']
    except:
        direccion =  'No se pudo obtener direccion'
    json_respuesta = {'Address':direccion}
    return jsonify(json_respuesta)


@app.after_request
def mysql_con(response):

    if (from_service == 'Taifelds' or from_service == 'Hidrosina' or from_service == 'Santory') and json_respuesta['Service'] == False:
        try:

            url = "https://us-central1-gchgame.cloudfunctions.net/mail-sender"

            payload = {'id_tienda':0,'message':body_rechazada,'service':from_service,'subject':'ALERTA - CAPTURA RECHAZADA'}
            headers = {'Content-Type': 'application/json'}

            mail_process5 = Process(target = enviar_mail, args = (url,headers,payload))
            mail_process5.start()

        except Exception as e:
            print(e)
            pass


    #Query a Cloud SQL
    if from_service == 'Taifelds' or from_service == 'Taifelds-disfruta' or from_service == 'Hidrosina' or from_service == 'Santory':

        if bandera == None:
            try:
                with engine.connect() as connection:
                    data_a_cloud_sql = [{'From Service':from_service,'Date':(datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
                                        'User Id':data['id'],'User Name':data['name'],
                                        'Mission Id':data['id_mission'],'Mission Name':data['mission_name'],
                                        'User Latitude':data['Location_latitude'],'User Longitude':data['Location_longitude'],
                                        'Mission Latitude':data['Location_mission_latitude'],'Mission Longitude':data['Location_mission_longitude'],
                                        'Start Date Mission':data['Start_Date_mission'],'End Date Mission':data['End_Date_mission'],
                                        'Target Time':data['Target_time_mission'],'Radio':data['Location_mission_radio'],
                                        'URL':data['url'],'Text':data['text'],
                                        'URL Video':data['Url_Video'],'URL Thumbnail (Video)':json_respuesta['url_thumbnail'],
                                        'Location':json_respuesta['Location'],'Time':json_respuesta['Time'],
                                        'Porn':json_respuesta['Porn'],
                                        'Live':json_respuesta['Live'],
                                        'Service':json_respuesta['Service'],'Message':json_respuesta['msg']}]
                    query_cloud = db.insert(result_data)
                    ResultProxy = connection.execute(query_cloud,data_a_cloud_sql)
            except Exception as e:
                print(e)
                
                try:
                    with engine.connect() as connection:
                        data_a_cloud_sql = [{'From Service':'Taifelds','Date':(datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
                                            'User Id':data['id'],'User Name':data['name'],
                                            'Mission Id':data['id_mission'],'Mission Name':data['mission_name'],
                                            'User Latitude':data['Location_latitude'],'User Longitude':data['Location_longitude'],
                                            'Mission Latitude':data['Location_mission_latitude'],'Mission Longitude':data['Location_mission_longitude'],
                                            'Start Date Mission':data['Start_Date_mission'],'End Date Mission':data['End_Date_mission'],
                                            'Target Time':data['Target_time_mission'],'Radio':data['Location_mission_radio'],
                                            'URL':data['url'],'Text':data['text'],
                                            'URL Video':data['Url_Video'],'URL Thumbnail (Video)':json_respuesta['url_thumbnail'],
                                            'Location':json_respuesta['Location'],'Time':json_respuesta['Time'],
                                            'Porn':json_respuesta['Porn'],
                                            'Live':json_respuesta['Live'],
                                            'Service':json_respuesta['Service'],'Message':json_respuesta['msg']}]
                        query_cloud = db.insert(result_data)
                        ResultProxy = connection.execute(query_cloud,data_a_cloud_sql)
                        print('Se logró')
                except Exception as e:
                    print(e)
                    print('Otra vez...')
                    pass

        else:
            pass
        
        return response

    elif from_service == 'Covid':

        if bandera == None:
            try:
                with engine.connect() as connection:
                    data_a_cloud_sql = [{'From Service':'Covid','Date':(datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
                                        'User Id':data['id'],'User Name':data['name'],
                                        'Mission Id':data['id_mission'],'Mission Name':data['mission_name'],
                                        'User Latitude':data['Location_latitude'],'User Longitude':data['Location_longitude'],
                                        'Mission Latitude':data['Location_mission_latitude'],'Mission Longitude':data['Location_mission_longitude'],
                                        'Start Date Mission':data['Start_Date_mission'],'End Date Mission':data['End_Date_mission'],
                                        'Target Time':data['Target_time_mission'],'Radio':data['Location_mission_radio'],
                                        'URL':data['url'],'Text':data['text'],
                                        'URL Video':data['Url_Video'],'URL Thumbnail (Video)':json_respuesta['url_thumbnail'],
                                        'Location':json_respuesta['Location'],'Time':json_respuesta['Time'],
                                        'Porn':json_respuesta['Porn'],
                                        'Live':json_respuesta['Live'],
                                        'Service':json_respuesta['Service'],'Message':json_respuesta['msg']}]
                    query_cloud = db.insert(result_data)
                    ResultProxy = connection.execute(query_cloud,data_a_cloud_sql)
            except Exception as e:
                print(e)

                try:
                    with engine.connect() as connection:
                        data_a_cloud_sql = [{'From Service':'Covid','Date':(datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
                                            'User Id':data['id'],'User Name':data['name'],
                                            'Mission Id':data['id_mission'],'Mission Name':data['mission_name'],
                                            'User Latitude':data['Location_latitude'],'User Longitude':data['Location_longitude'],
                                            'Mission Latitude':data['Location_mission_latitude'],'Mission Longitude':data['Location_mission_longitude'],
                                            'Start Date Mission':data['Start_Date_mission'],'End Date Mission':data['End_Date_mission'],
                                            'Target Time':data['Target_time_mission'],'Radio':data['Location_mission_radio'],
                                            'URL':data['url'],'Text':data['text'],
                                            'URL Video':data['Url_Video'],'URL Thumbnail (Video)':json_respuesta['url_thumbnail'],
                                            'Location':json_respuesta['Location'],'Time':json_respuesta['Time'],
                                            'Porn':json_respuesta['Porn'],
                                            'Live':json_respuesta['Live'],
                                            'Service':json_respuesta['Service'],'Message':json_respuesta['msg']}]
                        query_cloud = db.insert(result_data)
                        ResultProxy = connection.execute(query_cloud,data_a_cloud_sql)
                        print('Se logró')
                except Exception as e:
                    print(e)
                    print('Otra vez...')
                    pass

        else:
            pass
        
        return response
        
    elif from_service == 'premier' or from_service == 'Re:Explicit' or from_service == 'Explicit' or from_service == 'face recognition':
        try:
            with engine.connect() as connection:
                data_a_cloud_sql = [{'From Service':from_service,'Date':(datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
                                    'User Id':data['id'],'User Name':data['name'],
                                    'Mission Id':data['id_mission'],'Mission Name':data['mission_name'],
                                    'User Latitude':data['Location_latitude'],'User Longitude':data['Location_longitude'],
                                    'Mission Latitude':data['Location_mission_latitude'],'Mission Longitude':data['Location_mission_longitude'],
                                    'Start Date Mission':data['Start_Date_mission'],'End Date Mission':data['End_Date_mission'],
                                    'Target Time':data['Target_time_mission'],'Radio':data['Location_mission_radio'],
                                    'URL':data['url'],'URL Primaria':url2json0[0],'URL Selfie':json_respuesta['Url_themask'],'Text':data['text'],
                                    'URL Video':video_path,'URL Thumbnail (Video)':json_respuesta['url_thumbnail'],
                                    'Target_Scene':validar1 + ' o ' + validar4,'Target_Extra':validar3 + ' o ' + validar6,
                                    'Target_Object':validar2 + ' o ' + validar5,'Detected Object(s)':detected_obj,
                                    'Location':json_respuesta['Location'],'Time':json_respuesta['Time'],
                                    'Porn':json_respuesta['Porn'],'Live':Service[3],'Scene':Service[1],'Extra':Service[2],
                                    'Object':obj,'Service':json_respuesta['Service'],'Message':json_respuesta['msg']}]
                query_cloud = db.insert(result_data)
                ResultProxy = connection.execute(query_cloud,data_a_cloud_sql)
        except Exception as e:
            print(e)

            try:
                with engine.connect() as connection:
                    data_a_cloud_sql = [{'From Service':from_service,'Date':(datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
                                        'User Id':data['id'],'User Name':data['name'],
                                        'Mission Id':data['id_mission'],'Mission Name':data['mission_name'],
                                        'User Latitude':data['Location_latitude'],'User Longitude':data['Location_longitude'],
                                        'Mission Latitude':data['Location_mission_latitude'],'Mission Longitude':data['Location_mission_longitude'],
                                        'Start Date Mission':data['Start_Date_mission'],'End Date Mission':data['End_Date_mission'],
                                        'Target Time':data['Target_time_mission'],'Radio':data['Location_mission_radio'],
                                        'URL':data['url'],'URL Primaria':url2json0[0],'URL Selfie':json_respuesta['Url_themask'],'Text':data['text'],
                                        'URL Video':video_path,'URL Thumbnail (Video)':json_respuesta['url_thumbnail'],
                                        'Target_Scene':validar1 + ' o ' + validar4,'Target_Extra':validar3 + ' o ' + validar6,
                                        'Target_Object':validar2 + ' o ' + validar5,'Detected Object(s)':detected_obj,
                                        'Location':json_respuesta['Location'],'Time':json_respuesta['Time'],
                                        'Porn':json_respuesta['Porn'],'Live':Service[3],'Scene':Service[1],'Extra':Service[2],
                                        'Object':obj,'Service':json_respuesta['Service'],'Message':json_respuesta['msg']}]
                    query_cloud = db.insert(result_data)
                    ResultProxy = connection.execute(query_cloud,data_a_cloud_sql)
                    print('Se logró')
            except Exception as e:
                print(e)
                print('Otra vez...')
                pass

        return response

    else:
        return response
        
if __name__ == '__main__':
    
    
    app.run(host='127.0.0.1', port=8080, debug=False)