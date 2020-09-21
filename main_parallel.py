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
import unidecode
import requests
import imutils
import urllib
import six
import cv2
import re
import os
import sys
import dlib
import json
import ffmpeg
import pandas as pd
import face_recognition
import sqlalchemy as db
import tensorflow as tf
import matplotlib.pyplot as plt
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
    global missions_taifelds2
    global missions_hidrosina
    global missions_taifelds_disfruta
    global missions_covid
    global my_faces
    global nombres
    global url2json0
    global receivers
    global body_covid
    global body_taifelds
    global body_hidrosina
    global body_taifelds_disfruta
    global transfmr

    receivers = ['gotchudl@gmail.com','back.ght.001@gmail.com','medel@cimat.mx']
    body_taifelds = "Hay una nueva misión de Taifelds para validar en https://gchgame.web.app/ con número de tienda: "
    body_taifelds_disfruta = "Hay una nueva misión de Disfruta y Gana para validar en https://gchgame.web.app/ con número de tienda: "
    body_covid = "Hay una nueva misión de Hospital Covid para validar en https://gchgame.web.app/ con número de Id de Hospital: "
    body_hidrosina = "Hay una nueva misión de Hidrosina para validar en https://gchgame.web.app/ con número de Id: "

    metadata = db.MetaData()
    result_data = db.Table('result_data', metadata, autoload=True, autoload_with=engine)
    missions_taifelds = db.Table('taifelds', metadata, autoload=True, autoload_with=engine_misions)
    missions_taifelds2 = db.Table('taifelds2', metadata, autoload=True, autoload_with=engine_misions)
    missions_hidrosina = db.Table('hidrosina', metadata, autoload=True, autoload_with=engine_misions)
    missions_taifelds_disfruta = db.Table('taifelds_disfruta', metadata, autoload=True, autoload_with=engine_misions)
    missions_covid = db.Table('covid', metadata, autoload=True, autoload_with=engine_misions)

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
    date = datetime.utcnow().strftime("%Y-%m-%d-%H%M%S")
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
            overlay = cv2.imread('images/play_r.png')
        if an > al:
            image = cv2.resize(image, (thumb_height, thumb_width), 0, 0, cv2.INTER_LINEAR)
            overlay = cv2.imread('images/play_ra.png')

        background = image
        overlay = cv2.resize(overlay, (background.shape[1], background.shape[0]), 0, 0, cv2.INTER_LINEAR)
        added_image = cv2.addWeighted(background,0.6,overlay,0.2,0)

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
    image = vision.types.Image()
    image.source.image_uri = uri

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
    image = vision.types.Image()
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
                        return jsonify(json_respuesta)
                    else:
                        Service = [False,False,False,False]
                        from_service = 'face recognition'
                        json_respuesta = {'Location':True,'Time':True,'Service':False,'Porn':False,'Url_themask':'','url_thumbnail':video_to_thumbnail_url(video_path),'msg':''}
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
                                            
                                            if False in Service:
                                                json_respuesta = {'Location':True,'Time':True,'Service':False,'Porn':not Service[0],'Url_themask':'','url_thumbnail':'','msg':''}
                                                return jsonify(json_respuesta)
                                                
                                            else:
                                                det, detected_obj = detect_objects(image_path,validar5,objects,labels)
                                                obj = det == validar5 or det in ppoliticos
                                                json_respuesta = {'Location':True,'Time':True,'Service':det == validar5 or det in ppoliticos,'Porn':False,'Url_themask':imagen_final(masked_url[0]),'url_thumbnail':video_to_thumbnail_url(video_path),'msg':''}
                                                return jsonify(json_respuesta)

                                        else:
                                            json_respuesta = {'Location':True,'Time':True,'Service':False,'Porn':not Service[0],'Url_themask':'','url_thumbnail':'','msg':''}
                                            return jsonify(json_respuesta)

                                    else:
                                        json_respuesta = {'Location':True,'Time':True,'Service':False,'Porn':not Service[0],'Url_themask':'','url_thumbnail':'','msg':''}
                                        return jsonify(json_respuesta)

                                else:
                                    json_respuesta = {'Location':True,'Time':True,'Service':False,'Porn':not Service[0],'Url_themask':'','url_thumbnail':'','msg':''}
                                    return jsonify(json_respuesta)

                                
                            else:
                                det, detected_obj = detect_objects(image_path,validar2,objects,labels)
                                if det == validar2 or det in ppoliticos:
                                    obj = True
                                    json_respuesta = {'Location':True,'Time':True,'Service':True,'Porn':False,'Url_themask':imagen_final(masked_url[0]),'url_thumbnail':video_to_thumbnail_url(video_path),'msg':''}
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
                                                
                                                if False in Service:
                                                    json_respuesta = {'Location':True,'Time':True,'Service':False,'Porn':not Service[0],'Url_themask':'','url_thumbnail':'','msg':''}
                                                    return jsonify(json_respuesta)
                                                
                                                else:
                                                    det, detected_obj = detect_objects(image_path,validar5,objects,labels)
                                                    obj = det == validar5 or det in ppoliticos
                                                    json_respuesta = {'Location':True,'Time':True,'Service':det == validar5 or det in ppoliticos,'Porn':False,'Url_themask':imagen_final(masked_url[0]),'url_thumbnail':video_to_thumbnail_url(video_path),'msg':''}
                                                    return jsonify(json_respuesta)

                                            else:
                                                json_respuesta = {'Location':True,'Time':True,'Service':False,'Porn':False,'Url_themask':'','url_thumbnail':'','msg':''}
                                                return jsonify(json_respuesta)

                                        else:
                                            json_respuesta = {'Location':True,'Time':True,'Service':False,'Porn':False,'Url_themask':'','url_thumbnail':'','msg':''}
                                            return jsonify(json_respuesta)

                                    else:
                                        json_respuesta = {'Location':True,'Time':True,'Service':False,'Porn':False,'Url_themask':'','url_thumbnail':'','msg':''}
                                        return jsonify(json_respuesta)
        
                        else:
                            json_respuesta = {'Location':True,'Time':True,'Service':False,'Porn':False,'Url_themask':'','url_thumbnail':'','msg':''}
                            return jsonify(json_respuesta)                            
                            
                    else:
                        json_respuesta = {'Location':True,'Time':True,'Service':False,'Porn':False,'Url_themask':'','url_thumbnail':'','msg':''}
                        return jsonify(json_respuesta)

                else:
                    json_respuesta = {'Location':True,'Time':True,'Service':False,'Porn':False,'Url_themask':'','url_thumbnail':'','msg':''}
                    return jsonify(json_respuesta)
            
            else:
                json_respuesta = {'Location':True,'Time':True,'Service':False,'Porn':False,'Url_themask':'','url_thumbnail':'','msg':''}
                return jsonify(json_respuesta)
                             
        else:
            json_respuesta = {'Location':True,'Time':False,'Service':False,'Porn':False,'Url_themask':'','url_thumbnail':'','msg':''}
            return jsonify(json_respuesta)
       
    else:
        json_respuesta = {'Location':False,'Time':True,'Service':False,'Porn':False,'Url_themask':'','url_thumbnail':'','msg':''}
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
                  'nepe','mamadita','cojo']
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
                    return jsonify(json_respuesta)
                else:
                    json_respuesta = {'Location':True,'Time':True,'Service':True,'Porn':Service[0] or Service[1],'Url_themask':imagen_final(masked_url[0]),'url_thumbnail':video_to_thumbnail_url(video_path),'msg':''}
                    return jsonify(json_respuesta)
            else:
                json_respuesta = {'Location':True,'Time':False,'Service':False,'Porn':Service[0] or Service[1],'Url_themask':'','url_thumbnail':'','msg':''}
                return jsonify(json_respuesta)
        else:
            json_respuesta = {'Location':False,'Time':False,'Service':False,'Porn':Service[0] or Service[1],'Url_themask':'','url_thumbnail':'','msg':''}
            return jsonify(json_respuesta)
    else:
        json_respuesta = {'Location':True,'Time':True,'Service':False,'Porn':Service[0] or Service[1],'Url_themask':'','url_thumbnail':'','msg':''}
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
                    Date = (datetime.utcnow() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
                    User_Id = user_id, User_Latitude = user_lat, User_Longitude = user_lng, Url_Photo = url_p,
                    Url_Video = url_v, Mission_Id = miss_id))
        except Exception as e:
            print(e)
            try:
                with engine_misions.connect() as connection:
                    connection.execute(missions_taifelds.update().where(missions_taifelds.c.Id == id_tienda_salida).values(Flag = status,
                        Date = (datetime.utcnow() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
                        User_Id = user_id, User_Latitude = user_lat, User_Longitude = user_lng, Url_Photo = url_p,
                        Url_Video = url_v, Mission_Id = miss_id))
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
            results = connection.execute(db.select([missions_taifelds]).where(missions_taifelds.c.Flag != 'Yes')).fetchall()
    except Exception as e:
        print(e)
        try:
            with engine_misions.connect() as connection:
                results = connection.execute(db.select([missions_taifelds]).where(missions_taifelds.c.Flag != 'Yes')).fetchall()
        except Exception as e:
            print(e)
            json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':''}
            return jsonify(json_respuesta)

    if len(results) == 0:
        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':''}
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
                        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':''}
                        return jsonify(json_respuesta)
                
                try:

                    url = "https://us-central1-gchgame.cloudfunctions.net/mail-sender"

                    payload = {'id_tienda':id_tienda,'message':body_taifelds,'service':from_service,'subject':'Taifelds - NUEVA MISION'}
                    headers = {'Content-Type': 'application/json'}

                    response = requests.request("POST", url, headers=headers, data = json.dumps(payload))

                except Exception as e:
                    print(e)
                    pass
                json_respuesta = {'Location':True,'Time':True,'Service':True,'Live':True,'Porn':False,'Id':id_tienda,'Url_themask':imagen_final(image_path),'url_thumbnail':video_to_thumbnail_url(video_path),'msg':''}
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
        url_p = data['Url_Photo']
        url_v = data['Url_Video']
        video_path = data['Url_Video']
        miss_id = data['Mission_Id']

        try:
            with engine_misions.connect() as connection:
                connection.execute(missions_taifelds2.update().where(missions_taifelds2.c.Id == id_tienda_salida).values(Flag = status,
                    Date = (datetime.utcnow() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
                    User_Id = user_id, User_Latitude = user_lat, User_Longitude = user_lng, Url_Photo = url_p,
                    Url_Video = url_v, Mission_Id = miss_id))
        except Exception as e:
            print(e)
            try:
                with engine_misions.connect() as connection:
                    connection.execute(missions_taifelds2.update().where(missions_taifelds2.c.Id == id_tienda_salida).values(Flag = status,
                        Date = (datetime.utcnow() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
                        User_Id = user_id, User_Latitude = user_lat, User_Longitude = user_lng, Url_Photo = url_p,
                        Url_Video = url_v, Mission_Id = miss_id))
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
            results = connection.execute(db.select([missions_taifelds2]).where(missions_taifelds2.c.Flag != 'Yes')).fetchall()
    except Exception as e:
        print(e)
        try:
            with engine_misions.connect() as connection:
                results = connection.execute(db.select([missions_taifelds2]).where(missions_taifelds2.c.Flag != 'Yes')).fetchall()
        except Exception as e:
            print(e)
            json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':''}
            return jsonify(json_respuesta)

    if len(results) == 0:
        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':''}
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
                        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':''}
                        return jsonify(json_respuesta)
                
                try:

                    url = "https://us-central1-gchgame.cloudfunctions.net/mail-sender"

                    payload = {'id_tienda':id_tienda,'message':body_taifelds,'service':from_service,'subject':'Taifelds - NUEVA MISION'}
                    headers = {'Content-Type': 'application/json'}

                    response = requests.request("POST", url, headers=headers, data = json.dumps(payload))

                except Exception as e:
                    print(e)
                    pass
                json_respuesta = {'Location':True,'Time':True,'Service':True,'Live':True,'Porn':False,'Id':id_tienda,'Url_themask':imagen_final(image_path),'url_thumbnail':video_to_thumbnail_url(video_path),'msg':''}
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

    if bandera != 'yes':
        data['url'] = orientation_fix_function(data['url'])

    if bandera == 'yes':
        return
        try:
            with engine_misions.connect() as connection:
                results = connection.execute(db.select([missions_hidrosina]).where(missions_hidrosina.c.Flag == 'Pending')).fetchall()
        except Exception as e:
            print(e)
            try:
                with engine_misions.connect() as connection:
                    results = connection.execute(db.select([missions_hidrosina]).where(missions_hidrosina.c.Flag == 'Pending')).fetchall()
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
                connection.execute(missions_hidrosina.update().where(missions_hidrosina.c.Id == id_tienda_salida).values(Flag = status,
                    Date = (datetime.utcnow() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
                    User_Id = user_id, User_Latitude = user_lat, User_Longitude = user_lng, Url_Photo = url_p,
                    Url_Video = url_v, Mission_Id = miss_id))
        except Exception as e:
            print(e)
            try:
                with engine_misions.connect() as connection:
                    connection.execute(missions_hidrosina.update().where(missions_hidrosina.c.Id == id_tienda_salida).values(Flag = status,
                        Date = (datetime.utcnow() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
                        User_Id = user_id, User_Latitude = user_lat, User_Longitude = user_lng, Url_Photo = url_p,
                        Url_Video = url_v, Mission_Id = miss_id))
            except Exception as e:
                print(e)
                return jsonify({'Service':False})

        
        return jsonify({'Service':True})

    elif bandera != None:
        print('Error de request')
        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':'Error en la misión, contacta a Soporte Gotchu!'}
        return jsonify(json_respuesta)
    else:
        pass

    try:
        with engine_misions.connect() as connection:
            results = connection.execute(db.select([missions_hidrosina]).where(missions_hidrosina.c.Flag != 'Yes')).fetchall()
            results_yes = connection.execute(db.select([missions_hidrosina]).where(missions_hidrosina.c.Flag == 'Yes')).fetchall()
    except Exception as e:
        print(e)
        try:
            with engine_misions.connect() as connection:
                results = connection.execute(db.select([missions_hidrosina]).where(missions_hidrosina.c.Flag != 'Yes')).fetchall()
                results_yes = connection.execute(db.select([missions_hidrosina]).where(missions_hidrosina.c.Flag == 'Yes')).fetchall()
        except Exception as e:
            print(e)
            json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':'Error de conexión, inténtalo nuevamente'}
            return jsonify(json_respuesta)

    if len(results) == 0:
        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':'No hay más ubicaciones disponibles para esta misión, gracias por participar'}
        return jsonify(json_respuesta)

    df = pd.DataFrame(results)
    df.columns = results[0].keys()
    lat , lng, address, ids = df['Latitude'].tolist(), df['Longitude'].tolist(), df['Address'].tolist(), df['Id'].tolist()
    hora_inf, hora_sup = df['Horario_Inf_Timestamp'], df['Horario_Sup_Timestamp']
    user_pos = (data['Location_latitude'],data['Location_longitude'])

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

    full_text = detect_text_uri(image_path)
    codigo = find_codigo_factura(full_text)
    no_ticket = find_no_ticket(full_text)
    fecha, hora = find_fecha_hora(full_text)
    try:
        fecha_timestamp, hora_timestamp = fecha_hora_2_timestamp(fecha,hora)
    except Exception as e:
        print(e,fecha,hora)
        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':'Tu foto no cumple con los requerimientos o está mal enfocada, vuelve a intentarlo, si crees que esto es un error comunícate con soporte Gotchu!'}
        return jsonify(json_respuesta)

    if fecha == '1970-01-01' or hora == '00:00':
        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':'Tu foto no cumple con los requerimientos o está mal enfocada, vuelve a intentarlo, si crees que esto es un error comunícate con soporte Gotchu!'}
        return jsonify(json_respuesta)

    if hora_timestamp < 21600:
        hora_timestamp += 86400

    distancias = []
    for t, g in zip(lat,lng):
        distancias.append(geodesic(user_pos,(t,g)).meters)

    indices_horarios = [i for i, v in enumerate(distancias) if v < 100]
    horarios_vs_real_index = [i for i in indices_horarios if hora_inf[i] < hora_timestamp < hora_sup[i]]
    
    date_time_fecha = datetime.fromtimestamp(fecha_timestamp)
    fecha_hoy_str = date_time_fecha.strftime("%Y-%m-%d")
    fecha_hoy_str_real = (datetime.utcnow() - timedelta(hours=5)).strftime("%Y-%m-%d")

    if fecha_hoy_str_real != fecha_hoy_str:
        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':'Tu foto no cumple con los requerimientos o está mal enfocada, vuelve a intentarlo, si crees que esto es un error comunícate con soporte Gotchu!'}
        return jsonify(json_respuesta)

    if len(horarios_vs_real_index) > 0:
        
        start_date = datetime.strptime(data['Start_Date_mission'], '%Y-%m-%d %H:%M:%S.%f')
        end_date = datetime.strptime(data['End_Date_mission'], '%Y-%m-%d %H:%M:%S.%f')
        user_time = (end_date - start_date).total_seconds()
        mission_target_time = data['Target_time_mission']

        if (user_time<=mission_target_time):
            if no_ticket not in tickets:
                j = horarios_vs_real_index[0]
                id_tienda = ids[j]
                data['Location_mission_latitude'] = lat[j]
                data['Location_mission_longitude'] = lng[j]
                user_id = data['id']
                user_lat = user_pos[0]
                user_lng = user_pos[1]
                miss_id = data['id_mission']
                try:
                    hash_typeform = data['hash_typeform']
                except KeyError as e:
                    hash_typeform = ''
                    print('no hay variable hash_typeform')
                try:
                    with engine_misions.connect() as connection:
                        connection.execute(missions_hidrosina.update().where(missions_hidrosina.c.Id == id_tienda).values(Flag = 'Yes',
                            Date = (datetime.utcnow() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"), Fecha_Hora = fecha + ' ' + hora,
                            User_Id = user_id, User_Latitude = user_lat, User_Longitude = user_lng, Url_Photo = image_path,
                            Url_Video = video_path, Mission_Id = miss_id, Codigo_factura = codigo, Hash_typeform = hash_typeform, No_Ticket = no_ticket))
                except Exception as e:
                    print(e)
                    try:
                        with engine_misions.connect() as connection:
                            connection.execute(missions_hidrosina.update().where(missions_hidrosina.c.Id == id_tienda).values(Flag = 'Yes',
                                Date = (datetime.utcnow() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"), Fecha_Hora = fecha + ' ' + hora,
                                User_Id = user_id, User_Latitude = user_lat, User_Longitude = user_lng, Url_Photo = image_path,
                                Url_Video = video_path, Mission_Id = miss_id, Codigo_factura = codigo, Hash_typeform = hash_typeform, No_Ticket = no_ticket))
                    except Exception as e:
                        print(e)
                        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':'Error de conexión, inténtalo nuevamente'}
                        return jsonify(json_respuesta)
                
                try:

                    url = "https://us-central1-gchgame.cloudfunctions.net/mail-sender"

                    payload = {'id_tienda':id_tienda,'message':body_hidrosina,'service':from_service,'subject':'Hidrosina - NUEVA MISION'}
                    headers = {'Content-Type': 'application/json'}

                    response = requests.request("POST", url, headers=headers, data = json.dumps(payload))

                except Exception as e:
                    print(e)
                    pass
                json_respuesta = {'Location':True,'Time':True,'Service':True,'Live':True,'Porn':False,'Id':0,'Url_themask':imagen_final(image_path),'url_thumbnail':video_to_thumbnail_url(video_path),'msg':'¡Misión Hidrosina completada! Felicitaciones Agente'}
                return jsonify(json_respuesta)

            else:
                json_respuesta = {'Location':True,'Time':True,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':'Tu número de ticket no es correcto o ya se ha registrado'}
                return jsonify(json_respuesta)
        else:
            json_respuesta = {'Location':True,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':'Esta misión ha expirado'}
            return jsonify(json_respuesta)
    else:
        json_respuesta = {'Location':False,'Time':False,'Service':False,'Live':False,'Porn':False,'Id':0,'Url_themask':'','url_thumbnail':'','msg':'No te encuentras en una ubicación u horario disponible'}
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
                            Date = (datetime.utcnow() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
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
                                Date = (datetime.utcnow() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
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
                    Date = (datetime.utcnow() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
                    User_Id = user_id, User_Latitude = user_lat, User_Longitude = user_lng, Url_Photo = url_p,
                    Url_Video = url_v, Mission_Id = miss_id))
        except Exception as e:
            print(e)
            try:
                with engine_misions.connect() as connection:
                    connection.execute(missions_covid.update().where(missions_covid.c.Id == id_tienda_salida).values(Flag = status,
                        Date = (datetime.utcnow() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
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

@app.route('/imagen-to-texto', methods=['POST'])
def img2text():
    global from_service
    from_service = 'Texto'
    data = request.json
    image_path = data['url']
    texto = detect_text_uri_demo(image_path)
    json_respuesta = {'Texto':texto}
    return jsonify(json_respuesta)


@app.after_request
def mysql_con(response):
    #Query a Cloud SQL
    if from_service == 'Taifelds' or from_service == 'Taifelds-disfruta' or from_service == 'Hidrosina':

        if bandera == None:
            try:
                with engine.connect() as connection:
                    data_a_cloud_sql = [{'From Service':from_service,'Date':(datetime.utcnow() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
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
                                        'Service':json_respuesta['Service']}]
                    query_cloud = db.insert(result_data)
                    ResultProxy = connection.execute(query_cloud,data_a_cloud_sql)
            except Exception as e:
                print(e)
                
                try:
                    with engine.connect() as connection:
                        data_a_cloud_sql = [{'From Service':'Taifelds','Date':(datetime.utcnow() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
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
                                            'Service':json_respuesta['Service']}]
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
                    data_a_cloud_sql = [{'From Service':'Covid','Date':(datetime.utcnow() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
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
                                        'Service':json_respuesta['Service']}]
                    query_cloud = db.insert(result_data)
                    ResultProxy = connection.execute(query_cloud,data_a_cloud_sql)
            except Exception as e:
                print(e)

                try:
                    with engine.connect() as connection:
                        data_a_cloud_sql = [{'From Service':'Covid','Date':(datetime.utcnow() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
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
                                            'Service':json_respuesta['Service']}]
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
                data_a_cloud_sql = [{'From Service':from_service,'Date':(datetime.utcnow() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
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
                                    'Object':obj,'Service':json_respuesta['Service']}]
                query_cloud = db.insert(result_data)
                ResultProxy = connection.execute(query_cloud,data_a_cloud_sql)
        except Exception as e:
            print(e)

            try:
                with engine.connect() as connection:
                    data_a_cloud_sql = [{'From Service':from_service,'Date':(datetime.utcnow() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
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
                                        'Object':obj,'Service':json_respuesta['Service']}]
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