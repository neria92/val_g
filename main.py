from tensorflow.keras.preprocessing.image import img_to_array
from imageai.Prediction.Custom import CustomImagePrediction
from tensorflow.keras.preprocessing.image import load_img
from imutils.object_detection import non_max_suppression
from tensorflow.keras.preprocessing import image as ig
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from matplotlib.patches import Rectangle
from geopy.distance import geodesic
from datetime import datetime
from flask_cors import CORS
from matplotlib import pyplot
from numpy import expand_dims
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
import logging
import sqlalchemy
import sqlalchemy as db
import tensorflow as tf
import matplotlib.pyplot as plt
from Utils import label_map_util
from Utils import visualization_utils as vis_util


user = os.environ.get("DB_USER")
password = os.environ.get("DB_PASS")
database = os.environ.get("DB_NAME")
cloud_sql_connection_name = os.environ.get("CLOUD_SQL_CONNECTION_NAME")


app = Flask(__name__)

logger = logging.getLogger()

engine = db.create_engine(f'mysql+pymysql://{user}:{password}@/{database}?unix_socket=/cloudsql/{cloud_sql_connection_name}')


CORS(app)

@app.before_first_request
def loadmodel():
    global connection
    global result_data

    connection = engine.connect()
    metadata = db.MetaData()

    #base de datos de texto
    result_data = db.Table('result_data', metadata,
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
                db.Column('Service',db.BOOLEAN),
                )
    metadata.create_all(engine) #Creates Table

    MINIMUM_CONFIDENCE = 0.4

    PATH_TO_LABELS = 'label_map.pbtxt'

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=sys.maxsize, use_display_name=True)
    CATEGORY_INDEX = label_map_util.create_category_index(categories)

    PATH_TO_CKPT = 'frozen_inference_graph.pb'

    # Load model into memory
    print('Loading model...')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid: #.io
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
    print('detecting...')
    detection_graph.as_default()
    sess = tf.Session(graph=detection_graph)
    
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    model_incres = load_model('INCResV2_model_premier.h5')
    porn_model = load_model('porn_model_premier.h5')
    yolo_model = load_model('YOLOV3_model.h5')

    prediction = CustomImagePrediction()
    prediction.setModelTypeAsResNet()
    prediction.setModelPath("model_ex-092_acc-0.963542.h5")
    prediction.setJsonPath("model_class.json")
    prediction.loadModel(num_objects=2)
    

    # load our serialized face detector from disk
    protoPath = 'deploy.prototxt'
    modelPath = 'res10_300x300_ssd_iter_140000.caffemodel'

    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    app.model = [sess,image_tensor,detection_boxes,
                 detection_scores,detection_classes,num_detections,
                 CATEGORY_INDEX,MINIMUM_CONFIDENCE,model_incres,
                 detector,prediction,porn_model,yolo_model]

CORS(app)

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

CORS(app)

def _sigmoid(x):
	return 1. / (1. + np.exp(-x))

CORS(app)

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
 
CORS(app)

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

CORS(app)

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

CORS(app)
 
def bbox_iou(box1, box2):
	intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
	intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
	intersect = intersect_w * intersect_h
	w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
	w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
	union = w1*h1 + w2*h2 - intersect
	return float(intersect) / union

CORS(app)

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
                    
CORS(app)

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

CORS(app)

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

CORS(app)

def load_image_pixels(image_original, shape):
    # load the image to get its shape
    
    
    image = image_original
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



CORS(app)

def detect_objects(image_original,validar,names,labels):
    if validar in names:
        sess = app.model[0]
        image_tensor = app.model[1]
        detection_boxes = app.model[2]
        
        detection_scores = app.model[3]
        detection_classes = app.model[4]

        num_detections = app.model[5]
        
        CATEGORY_INDEX = app.model[6]
        MINIMUM_CONFIDENCE = app.model[7]

        size = 600 
        
        image = image_original
        
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
            return respns
        else:
            respns = respns[0]['name']
            return respns

    elif validar in labels:
        
        anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
        class_threshold = 0.6
        input_w, input_h = 416, 416

        image, image_w, image_h = load_image_pixels(image_original, (input_w, input_h))
        
        yolo_model = app.model[12]
        yhat = yolo_model.predict(image)

        boxes = list()
        for i in range(len(yhat)):
            
            boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
        

        correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)

        do_nms(boxes, 0.5)

        v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)

        if validar in v_labels:
            return validar
        else:
            return 'no_detected'



CORS(app)

def detect_human(image_path):
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
            return False
        else:
            return True
    else:
        return True

CORS(app)
def detect_scene(image_path,validar,class_names,class_names_param,x):
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
    return pred

CORS(app)
  
@app.route('/', methods=['POST','GET'])
def location_time_validate():
    data = request.json
    """user_pos = (data['Location_latitude'],data['Location_longitude']) 
    mission_point_pos = (data['Location_mission_latitude'],data['Location_mission_longitude'])"""
    
    if True: #(geodesic(user_pos, mission_point_pos).meters <= data['Location_mission_radio']):
      
        """start_date = datetime.strptime(data['Start_Date_mission'], '%Y-%m-%d %H:%M:%S.%f')
        end_date = datetime.strptime(data['End_Date_mission'], '%Y-%m-%d %H:%M:%S.%f')
        user_time = (end_date - start_date).total_seconds()
        mission_target_time = data['Target_time_mission']"""
        
        if True: #(user_time<=mission_target_time):
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
                response = requests.get(image_path)
                image_original = Image.open(BytesIO(response.content))
                                
                image_original_r = image_original.resize((299,299))
                x = ig.img_to_array(image_original_r)
                x = x.reshape((1,) + x.shape)
                x=x/255

                porn_model = app.model[11]            
                porn_preds = porn_model.predict(x)
                #Es imagen inapropiada?
                if porn_preds[0][0] > 0.5: #No entonces continua
                                        
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
                    objects = ['basura','colilla','ppolitico']
                    extras = ['persona']
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
                    validar3 = request.args.get('validar_extra')
                    validar4 = request.args.get('o_validar_escena')
                    validar5 = request.args.get('o_validar_objeto')
                    validar6 = request.args.get('o_validar_extra')



                    if validar1 in class_names or validar1 == 'na':
                        if validar2 in objects or validar2 in labels or validar2 == 'na':
                            if validar3 in extras or validar3 == 'na':
                                if validar1 == 'na':
                                    if validar2 == 'na':
                                        if validar3 == 'na':
                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                        else:
                                            #human
                                            human = detect_human(image_path)

                                            if human == False:
                                                if validar4 in class_names or validar4 == 'na':
                                                    if validar5 in objects or validar5 in labels or validar5 == 'na':
                                                        if validar6 in extras or validar6 == 'na':
                                                            if validar4 == 'na':
                                                                if validar5 == 'na':
                                                                    if validar6 == 'na':
                                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                    else:
                                                                        #human
                                                                        human = detect_human(image_path)
                                                                        if human == False:
                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                        else:
                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                else:
                                                                    det = detect_objects(image_original,validar5,objects,labels)
                                                                    if det == validar5:
                                                                        if validar6 == 'na':
                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                        else:
                                                                            #human
                                                                            human = detect_human(image_path)
                                                                            if human == False:
                                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                            else:
                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                            
                                                                    else:
                                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                    
                                                            else:
                                                                pred = detect_scene(image_path,validar4,class_names,class_names_param,x)
                                                                
                                                                T = []

                                                                if pred[0][indx]>thres:
                                                                    if validar5 == 'na':

                                                                        if validar6 == 'na':
                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                        else:
                                                                            #human
                                                                            human = detect_human(image_path)
                                                                            if human == False:
                                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                            else:
                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                
                                                
                                                                    
                                                                    else:
                                                                        det = detect_objects(image_original,validar5,objects,labels)
                                                                        if det == validar5:
                                                                            
                                                                            if validar6 == 'na':
                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                            else:
                                                                                #human
                                                                                human = detect_human(image_path)
                                                                                if human == False:
                                                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                else:
                                                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                    
                                                                        else:
                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                    
                                                                else:
                                                                    if relat != []:
                                                                        for r in relat:
                                                                            idx = class_names.index(r)
                                                                        
                                                                            if pred[0][idx]>ths:
                                                                                T.append(ths)
                                                                                
                                                                                if validar5 == 'na':

                                                                                    if validar6 == 'na':
                                                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                    else:
                                                                                        #human
                                                                                        human = detect_human(image_path)
                                                                                        if human == False:
                                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                        else:
                                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})

                                                                                else:
                                                                                    det = detect_objects(image_original,validar5,objects,labels)
                                                                                    if det == validar5:
                                                                                        
                                                                                        if validar6 == 'na':
                                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                        else:
                                                                                            #human
                                                                                            human = detect_human(image_path)
                                                                                            if human == False:
                                                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                            else:
                                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                                                              
                                                            
                                                                                    else:
                                                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                

                                                                                break
                                                                    if T == []:
                                                                        
                                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})

                                                        else:
                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                    else:
                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})


                                                else:
                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                
                                            else:
                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                
                                            
                                    
                                    else:
                                        det = detect_objects(image_original,validar2,objects,labels)
                                        if det == validar2:
                                            if validar3 == 'na':
                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                            else:
                                                #human
                                                human = detect_human(image_path)
                                                                                            
                                                if human == False:
                                                    if validar4 in class_names or validar4 == 'na':
                                                        if validar5 in objects or validar5 in labels or validar5 == 'na':
                                                            if validar6 in extras or validar6 == 'na':
                                                                if validar4 == 'na':
                                                                    if validar5 == 'na':
                                                                        if validar6 == 'na':
                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                        else:
                                                                            #human
                                                                            human = detect_human(image_path)
                                                                            if human == False:
                                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                            else:
                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                                                              
                                                                                                        
                                                                    else:
                                                                        det = detect_objects(image_original,validar5,objects,labels)
                                                                        if det == validar5:
                                                                            if validar6 == 'na':
                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                            else:
                                                                                #human
                                                                                human = detect_human(image_path)
                                                                                if human == False:
                                                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                else:
                                                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                
                                                                        else:
                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                        
                                                                else:
                                                                    pred = detect_scene(image_path,validar4,class_names,class_names_param,x)
                                                                    
                                                                    T = []

                                                                    if pred[0][indx]>thres:
                                                                        if validar5 == 'na':

                                                                            if validar6 == 'na':
                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                            else:
                                                                                #human
                                                                                human = detect_human(image_path)
                                                                                if human == False:
                                                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                else:
                                                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                
                                                                        else:
                                                                            det = detect_objects(image_original,validar5,objects,labels)
                                                                            if det == validar5:
                                                                                
                                                                                if validar6 == 'na':
                                                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                else:
                                                                                    #human
                                                                                    human = detect_human(image_path)
                                                                                    if human == False:
                                                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                    else:
                                                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                

                                                                            else:
                                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                        
                                                                    else:
                                                                        if relat != []:
                                                                            for r in relat:
                                                                                idx = class_names.index(r)
                                                                            
                                                                                if pred[0][idx]>ths:
                                                                                    T.append(ths)
                                                                                    
                                                                                    if validar5 == 'na':

                                                                                        if validar6 == 'na':
                                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                        else:
                                                                                            #human
                                                                                            human = detect_human(image_path)
                                                                                            if human == False:
                                                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                            else:
                                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                    
                                                                                    else:
                                                                                        det = detect_objects(image_original,validar5,objects,labels)
                                                                                        if det == validar5:
                                                                                            
                                                                                            if validar6 == 'na':
                                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                            else:
                                                                                                #human
                                                                                                human = detect_human(image_path)
                                                                                                if human == False:
                                                                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                                else:
                                                                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                                    

                                                                                        else:
                                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                    

                                                                                    break
                                                                        if T == []:
                                                                            
                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})

                                                            else:
                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                        else:
                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})


                                                    else:
                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                    
                                                else:
                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                
                                        else:
                                            if validar4 in class_names or validar4 == 'na':
                                                if validar5 in objects or validar5 in labels or validar5 == 'na':
                                                    if validar6 in extras or validar6 == 'na':
                                                        if validar4 == 'na':
                                                            if validar5 == 'na':
                                                                if validar6 == 'na':
                                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                else:
                                                                    #human
                                                                    human = detect_human(image_path)
                                                                    if human == False:
                                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                    else:
                                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                    
                                                            else:
                                                                det = detect_objects(image_original,validar5,objects,labels)
                                                                if det == validar5:
                                                                    if validar6 == 'na':
                                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                    else:
                                                                        #human
                                                                        human = detect_human(image_path)
                                                                        if human == False:
                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                        else:
                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                    
                                                                        
                                                                else:
                                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                
                                                        else:
                                                            pred = detect_scene(image_path,validar4,class_names,class_names_param,x)
                                                            
                                                            T = []

                                                            if pred[0][indx]>thres:
                                                                if validar5 == 'na':

                                                                    if validar6 == 'na':
                                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                    else:
                                                                        #human
                                                                        human = detect_human(image_path)
                                                                        if human == False:
                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                        else:
                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                            
                                                    
                                                                else:
                                                                    det = detect_objects(image_original,validar5,objects,labels)
                                                                    if det == validar5:
                                                                        
                                                                        if validar6 == 'na':
                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                        else:
                                                                            #human
                                                                            human = detect_human(image_path)
                                                                            if human == False:
                                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                            else:
                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                            

                                                                    else:
                                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                
                                                            else:
                                                                if relat != []:
                                                                    for r in relat:
                                                                        idx = class_names.index(r)
                                                                    
                                                                        if pred[0][idx]>ths:
                                                                            T.append(ths)
                                                                            
                                                                            if validar5 == 'na':

                                                                                if validar6 == 'na':
                                                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                else:
                                                                                    #human
                                                                                    human = detect_human(image_path)
                                                                                    if human == False:
                                                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                    else:
                                                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                    
                                                                                    
                                                                            else:
                                                                                det = detect_objects(image_original,validar5,objects,labels)
                                                                                if det == validar5:
                                                                                    
                                                                                    if validar6 == 'na':
                                                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                    else:
                                                                                        #human
                                                                                        human = detect_human(image_path)
                                                                                        if human == False:
                                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                        else:
                                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                    

                                                                                else:
                                                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                            

                                                                            break
                                                                if T == []:
                                                                    
                                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})

                                                    else:
                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                else:
                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})


                                            else:
                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                            
                                        
                                else:
                                    pred = detect_scene(image_path,validar1,class_names,class_names_param,x)
                                    
                                    T = []

                                    if pred[0][indx]>thres:
                                        if validar2 == 'na':

                                            if validar3 == 'na':
                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                            else:
                                                #human
                                                human = detect_human(image_path)
                                                
                                                if human == False:
                                                    if validar4 in class_names or validar4 == 'na':
                                                        if validar5 in objects or validar5 in labels or validar5 == 'na':
                                                            if validar6 in extras or validar6 == 'na':
                                                                if validar4 == 'na':
                                                                    if validar5 == 'na':
                                                                        if validar6 == 'na':
                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                        else:
                                                                            #human
                                                                            human = detect_human(image_path)      
                                                                                
                                                                            if human == False:
                                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                            else:
                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                
                                                                            
                                                                    else:
                                                                        det = detect_objects(image_original,validar5,objects,labels)
                                                                        if det == validar5:
                                                                            if validar6 == 'na':
                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                            else:
                                                                                #human
                                                                                human = detect_human(image_path)      
                                                                                    
                                                                                if human == False:
                                                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                else:
                                                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})   
                                                                                    
                                                                        else:
                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                        
                                                                else:
                                                                    pred = detect_scene(image_path,validar4,class_names,class_names_param,x)
                                                                    
                                                                    T = []

                                                                    if pred[0][indx]>thres:
                                                                        if validar5 == 'na':

                                                                            if validar6 == 'na':
                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                            else:
                                                                                #human
                                                                                human = detect_human(image_path)      
                                                                                    
                                                                                if human == False:
                                                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                else:
                                                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                    
                                                                        else:
                                                                            det = detect_objects(image_original,validar5,objects,labels)
                                                                            if det == validar5:
                                                                                
                                                                                if validar6 == 'na':
                                                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                else:
                                                                                    #human
                                                                                    human = detect_human(image_path)      
                                                                                        
                                                                                    if human == False:
                                                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                    else:
                                                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                    

                                                                            else:
                                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                        
                                                                    else:
                                                                        if relat != []:
                                                                            for r in relat:
                                                                                idx = class_names.index(r)
                                                                            
                                                                                if pred[0][idx]>ths:
                                                                                    T.append(ths)
                                                                                    
                                                                                    if validar5 == 'na':

                                                                                        if validar6 == 'na':
                                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                        else:
                                                                                            #human
                                                                                            human = detect_human(image_path)      
                                                                                                
                                                                                            if human == False:
                                                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                            else:
                                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                    
                                                                                            
                                                                                    else:
                                                                                        det = detect_objects(image_original,validar5,objects,labels)
                                                                                        if det == validar5:
                                                                                            
                                                                                            if validar6 == 'na':
                                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                            else:
                                                                                                #human
                                                                                                human = detect_human(image_path)      
                                                                                                    
                                                                                                if human == False:
                                                                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                                else:
                                                                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                    

                                                                                        else:
                                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                    

                                                                                    break
                                                                        if T == []:
                                                                            
                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})

                                                            else:
                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                        else:
                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})


                                                    else:
                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                    
                                                else:
                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                
                                                
                                        else:
                                            det = detect_objects(image_original,validar2,objects,labels)
                                            if det == validar2:
                                                
                                                if validar3 == 'na':
                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                else:
                                                    #human
                                                    human = detect_human(image_path)
                                                    
                                                    if human == False:
                                                        if validar4 in class_names or validar4 == 'na':
                                                            if validar5 in objects or validar5 in labels or validar5 == 'na':
                                                                if validar6 in extras or validar6 == 'na':
                                                                    if validar4 == 'na':
                                                                        if validar5 == 'na':
                                                                            if validar6 == 'na':
                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                            else:
                                                                                #human
                                                                                human = detect_human(image_path)      
                                                                                    
                                                                                if human == False:
                                                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                else:
                                                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                
                                                                        else:
                                                                            det = detect_objects(image_original,validar5,objects,labels)
                                                                            if det == validar5:
                                                                                if validar6 == 'na':
                                                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                else:
                                                                                    #human
                                                                                    human = detect_human(image_path)      
                                                                                        
                                                                                    if human == False:
                                                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                    else:
                                                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                    
                                                                            else:
                                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                            
                                                                    else:
                                                                        pred = detect_scene(image_path,validar4,class_names,class_names_param,x)
                                                                        
                                                                        T = []

                                                                        if pred[0][indx]>thres:
                                                                            if validar5 == 'na':

                                                                                if validar6 == 'na':
                                                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                else:
                                                                                    #human
                                                                                    human = detect_human(image_path)      
                                                                                        
                                                                                    if human == False:
                                                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                    else:
                                                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                    
                                                                            else:
                                                                                det = detect_objects(image_original,validar5,objects,labels)
                                                                                if det == validar5:
                                                                                    
                                                                                    if validar6 == 'na':
                                                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                    else:
                                                                                        #human
                                                                                        human = detect_human(image_path)      
                                                                                            
                                                                                        if human == False:
                                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                        else:
                                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                    
                                                                                        
                                                                                else:
                                                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                            
                                                                        else:
                                                                            if relat != []:
                                                                                for r in relat:
                                                                                    idx = class_names.index(r)
                                                                                
                                                                                    if pred[0][idx]>ths:
                                                                                        T.append(ths)
                                                                                        
                                                                                        if validar5 == 'na':

                                                                                            if validar6 == 'na':
                                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                            else:
                                                                                                #human
                                                                                                human = detect_human(image_path)      
                                                                                                    
                                                                                                if human == False:
                                                                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                                else:
                                                                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                    
                                                                                        else:
                                                                                            det = detect_objects(image_original,validar5,objects,labels)
                                                                                            if det == validar5:
                                                                                                
                                                                                                if validar6 == 'na':
                                                                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                                else:
                                                                                                    #human
                                                                                                    human = detect_human(image_path)      
                                                                                                        
                                                                                                    if human == False:
                                                                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                                    else:
                                                                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                    

                                                                                            else:
                                                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                        

                                                                                        break
                                                                            if T == []:
                                                                                
                                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})

                                                                else:
                                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                            else:
                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})


                                                        else:
                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                        
                                                    else:
                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})

                                            else:
                                                if validar4 in class_names or validar4 == 'na':
                                                    if validar5 in objects or validar5 in labels or validar5 == 'na':
                                                        if validar6 in extras or validar6 == 'na':
                                                            if validar4 == 'na':
                                                                if validar5 == 'na':
                                                                    if validar6 == 'na':
                                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                    else:
                                                                        #human
                                                                        human = detect_human(image_path)      
                                                                            
                                                                        if human == False:
                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                        else:
                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                            
                                                                else:
                                                                    det = detect_objects(image_original,validar5,objects,labels)
                                                                    if det == validar5:
                                                                        if validar6 == 'na':
                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                        else:
                                                                            #human
                                                                            human = detect_human(image_path)      
                                                                                
                                                                            if human == False:
                                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                            else:
                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                               
                                                                                
                                                                        
                                                                    else:
                                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                    
                                                            else:
                                                                pred = detect_scene(image_path,validar4,class_names,class_names_param,x)
                                                                
                                                                T = []

                                                                if pred[0][indx]>thres:
                                                                    if validar5 == 'na':

                                                                        if validar6 == 'na':
                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                        else:
                                                                            #human
                                                                            human = detect_human(image_path)      
                                                                                
                                                                            if human == False:
                                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                            else:
                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                
                                                                    else:
                                                                        det = detect_objects(image_original,validar5,objects,labels)
                                                                        if det == validar5:
                                                                            
                                                                            if validar6 == 'na':
                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                            else:
                                                                                #human
                                                                                human = detect_human(image_path)      
                                                                                    
                                                                                if human == False:
                                                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                else:
                                                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                    

                                                                        else:
                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                    
                                                                else:
                                                                    if relat != []:
                                                                        for r in relat:
                                                                            idx = class_names.index(r)
                                                                        
                                                                            if pred[0][idx]>ths:
                                                                                T.append(ths)
                                                                                
                                                                                if validar5 == 'na':

                                                                                    if validar6 == 'na':
                                                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                    else:
                                                                                        #human
                                                                                        human = detect_human(image_path)      
                                                                                            
                                                                                        if human == False:
                                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                        else:
                                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                    
                                                                                else:
                                                                                    det = detect_objects(image_original,validar5,objects,labels)
                                                                                    if det == validar5:
                                                                                        
                                                                                        if validar6 == 'na':
                                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                        else:
                                                                                            #human
                                                                                            human = detect_human(image_path)      
                                                                                                
                                                                                            if human == False:
                                                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                            else:
                                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                                

                                                                                    else:
                                                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                

                                                                                break
                                                                    if T == []:
                                                                        
                                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})

                                                        else:
                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                    else:
                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})


                                                else:
                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                
                                        
                                    else:
                                        if relat != []:
                                            for r in relat:
                                                idx = class_names.index(r)
                                            
                                                if pred[0][idx]>ths:
                                                    T.append(ths)
                                                    
                                                    if validar2 == 'na':

                                                        if validar3 == 'na':
                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                        else:
                                                            #human
                                                            human = detect_human(image_path)      
                                                                                            
                                                            if human == False:
                                                                if validar4 in class_names or validar4 == 'na':
                                                                    if validar5 in objects or validar5 in labels or validar5 == 'na':
                                                                        if validar6 in extras or validar6 == 'na':
                                                                            if validar4 == 'na':
                                                                                if validar5 == 'na':
                                                                                    if validar6 == 'na':
                                                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                    else:
                                                                                        #human
                                                                                        human = detect_human(image_path)      
                                                                                            
                                                                                        if human == False:
                                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                        else:
                                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                    
                                                                                else:
                                                                                    det = detect_objects(image_original,validar5,objects,labels)
                                                                                    if det == validar5:
                                                                                        if validar6 == 'na':
                                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                        else:
                                                                                            #human
                                                                                            human = detect_human(image_path)      
                                                                                                
                                                                                            if human == False:
                                                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                            else:
                                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                    
                                                                                    else:
                                                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                    
                                                                            else:
                                                                                pred = detect_scene(image_path,validar4,class_names,class_names_param,x)
                                                                                
                                                                                T = []

                                                                                if pred[0][indx]>thres:
                                                                                    if validar5 == 'na':

                                                                                        if validar6 == 'na':
                                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                        else:
                                                                                            #human
                                                                                            human = detect_human(image_path)      
                                                                                                
                                                                                            if human == False:
                                                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                            else:
                                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                                
                                                                                    else:
                                                                                        det = detect_objects(image_original,validar5,objects,labels)
                                                                                        if det == validar5:
                                                                                            
                                                                                            if validar6 == 'na':
                                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                            else:
                                                                                                #human
                                                                                                human = detect_human(image_path)      
                                                                                                    
                                                                                                if human == False:
                                                                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                                else:
                                                                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                                    
                                                                                        else:
                                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                    
                                                                                else:
                                                                                    if relat != []:
                                                                                        for r in relat:
                                                                                            idx = class_names.index(r)
                                                                                        
                                                                                            if pred[0][idx]>ths:
                                                                                                T.append(ths)
                                                                                                
                                                                                                if validar5 == 'na':

                                                                                                    if validar6 == 'na':
                                                                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                                    else:
                                                                                                        #human
                                                                                                        human = detect_human(image_path)      
                                                                                                            
                                                                                                        if human == False:
                                                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                                        else:
                                                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                                                    
                                                                                                else:
                                                                                                    det = detect_objects(image_original,validar5,objects,labels)
                                                                                                    if det == validar5:
                                                                                                        
                                                                                                        if validar6 == 'na':
                                                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                                        else:
                                                                                                            #human
                                                                                                            human = detect_human(image_path)      
                                                                                                                
                                                                                                            if human == False:
                                                                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                                            else:
                                                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                    

                                                                                                    else:
                                                                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                                

                                                                                                break
                                                                                    if T == []:
                                                                                        
                                                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})

                                                                        else:
                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                    else:
                                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})


                                                                else:
                                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                
                                                            else:
                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                            
                                                            
                                                    else:
                                                        det = detect_objects(image_original,validar2,objects,labels)
                                                        if det == validar2:
                                                                                                                
                                                            if validar3 == 'na':
                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                            else:
                                                                #human
                                                                human = detect_human(image_path)      
                                                            
                                                                if human == False:
                                                                    if validar4 in class_names or validar4 == 'na':
                                                                        if validar5 in objects or validar5 in labels or validar5 == 'na':
                                                                            if validar6 in extras or validar6 == 'na':
                                                                                if validar4 == 'na':
                                                                                    if validar5 == 'na':
                                                                                        if validar6 == 'na':
                                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                        else:
                                                                                            #human
                                                                                            human = detect_human(image_path)      
                                                                                                
                                                                                            if human == False:
                                                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                            else:
                                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                                    
                                                                                    else:
                                                                                        det = detect_objects(image_original,validar5,objects,labels)
                                                                                        if det == validar5:
                                                                                            if validar6 == 'na':
                                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                            else:
                                                                                                #human
                                                                                                human = detect_human(image_path)      
                                                                                                    
                                                                                                if human == False:
                                                                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                                else:
                                                                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                            
                                                                                        else:
                                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                        
                                                                                else:
                                                                                    pred = detect_scene(image_path,validar4,class_names,class_names_param,x)
                                                                                    
                                                                                    T = []

                                                                                    if pred[0][indx]>thres:
                                                                                        if validar5 == 'na':

                                                                                            if validar6 == 'na':
                                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                            else:
                                                                                                #human
                                                                                                human = detect_human(image_path)      
                                                                                                    
                                                                                                if human == False:
                                                                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                                else:
                                                                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                                        
                                                                                        else:
                                                                                            det = detect_objects(image_original,validar5,objects,labels)
                                                                                            if det == validar5:
                                                                                                
                                                                                                if validar6 == 'na':
                                                                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                                else:
                                                                                                    #human
                                                                                                    human = detect_human(image_path)      
                                                                                                        
                                                                                                    if human == False:
                                                                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                                    else:
                                                                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                                                                                                
                                                                                            else:
                                                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                        
                                                                                    else:
                                                                                        if relat != []:
                                                                                            for r in relat:
                                                                                                idx = class_names.index(r)
                                                                                            
                                                                                                if pred[0][idx]>ths:
                                                                                                    T.append(ths)
                                                                                                    
                                                                                                    if validar5 == 'na':

                                                                                                        if validar6 == 'na':
                                                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                                        else:
                                                                                                            #human
                                                                                                            human = detect_human(image_path)      
                                                                                                                
                                                                                                            if human == False:
                                                                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                                            else:
                                                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                                                        
                                                                                                    else:
                                                                                                        det = detect_objects(image_original,validar5,objects,labels)
                                                                                                        if det == validar5:
                                                                                                            
                                                                                                            if validar6 == 'na':
                                                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                                            else:
                                                                                                                #human
                                                                                                                human = detect_human(image_path)      
                                                                                                                    
                                                                                                                if human == False:
                                                                                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                                                else:
                                                                                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                                                

                                                                                                        else:
                                                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                                    

                                                                                                    break
                                                                                        if T == []:
                                                                                            
                                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})

                                                                            else:
                                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                        else:
                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})


                                                                    else:
                                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                    
                                                                else:
                                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                
                                                                
                                                        else:
                                                            if validar4 in class_names or validar4 == 'na':
                                                                if validar5 in objects or validar5 in labels or validar5 == 'na':
                                                                    if validar6 in extras or validar6 == 'na':
                                                                        if validar4 == 'na':
                                                                            if validar5 == 'na':
                                                                                if validar6 == 'na':
                                                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                else:
                                                                                    #human
                                                                                    human = detect_human(image_path)      
                                                                                        
                                                                                    if human == False:
                                                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                    else:
                                                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                            
                                                                            else:
                                                                                det = detect_objects(image_original,validar5,objects,labels)
                                                                                if det == validar5:
                                                                                    if validar6 == 'na':
                                                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                    else:
                                                                                        #human
                                                                                        human = detect_human(image_path)      
                                                                                            
                                                                                        if human == False:
                                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                        else:
                                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                            
                                                                                else:
                                                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                
                                                                        else:
                                                                            pred = detect_scene(image_path,validar4,class_names,class_names_param,x)
                                                                            
                                                                            T = []

                                                                            if pred[0][indx]>thres:
                                                                                if validar5 == 'na':

                                                                                    if validar6 == 'na':
                                                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                    else:
                                                                                        #human
                                                                                        human = detect_human(image_path)      
                                                                                            
                                                                                        if human == False:
                                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                        else:
                                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                                        
                                                                                else:
                                                                                    det = detect_objects(image_original,validar5,objects,labels)
                                                                                    if det == validar5:
                                                                                        
                                                                                        if validar6 == 'na':
                                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                        else:
                                                                                            #human
                                                                                            human = detect_human(image_path)      
                                                                                                
                                                                                            if human == False:
                                                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                            else:
                                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                                

                                                                                    else:
                                                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                
                                                                            else:
                                                                                if relat != []:
                                                                                    for r in relat:
                                                                                        idx = class_names.index(r)
                                                                                    
                                                                                        if pred[0][idx]>ths:
                                                                                            T.append(ths)
                                                                                            
                                                                                            if validar5 == 'na':

                                                                                                if validar6 == 'na':
                                                                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                                else:
                                                                                                    #human
                                                                                                    human = detect_human(image_path)      
                                                                                                        
                                                                                                    if human == False:
                                                                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                                    else:
                                                                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                            
                                                                                            else:
                                                                                                det = detect_objects(image_original,validar5,objects,labels)
                                                                                                if det == validar5:
                                                                                                    
                                                                                                    if validar6 == 'na':
                                                                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                                    else:
                                                                                                        #human
                                                                                                        human = detect_human(image_path)      
                                                                                                            
                                                                                                        if human == False:
                                                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                                        else:
                                                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                                            

                                                                                                else:
                                                                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                            

                                                                                            break
                                                                                if T == []:
                                                                                    
                                                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})

                                                                    else:
                                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                else:
                                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})


                                                            else:
                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                            
                                                    

                                                    break
                                        if T == []:
                                            
                                            if validar4 in class_names or validar4 == 'na':
                                                if validar5 in objects or validar5 in labels or validar5 == 'na':
                                                    if validar6 in extras or validar6 == 'na':
                                                        if validar4 == 'na':
                                                            if validar5 == 'na':
                                                                if validar6 == 'na':
                                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                else:
                                                                    #human
                                                                    human = detect_human(image_path)      
                                                                        
                                                                    if human == False:
                                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                    else:
                                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                              
                                                            else:
                                                                det = detect_objects(image_original,validar5,objects,labels)
                                                                if det == validar5:
                                                                    if validar6 == 'na':
                                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                    else:
                                                                        #human
                                                                        human = detect_human(image_path)      
                                                                            
                                                                        if human == False:
                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                        else:
                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                            
                                                                else:
                                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                
                                                        else:
                                                            pred = detect_scene(image_path,validar4,class_names,class_names_param,x)
                                                            
                                                            T = []

                                                            if pred[0][indx]>thres:
                                                                if validar5 == 'na':

                                                                    if validar6 == 'na':
                                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                    else:
                                                                        #human
                                                                        human = detect_human(image_path)      
                                                                            
                                                                        if human == False:
                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                        else:
                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                            
                                                                else:
                                                                    det = detect_objects(image_original,validar5,objects,labels)
                                                                    if det == validar5:
                                                                        
                                                                        if validar6 == 'na':
                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                        else:
                                                                            #human
                                                                            human = detect_human(image_path)      
                                                                                
                                                                            if human == False:
                                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                            else:
                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                    

                                                                    else:
                                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                
                                                            else:
                                                                if relat != []:
                                                                    for r in relat:
                                                                        idx = class_names.index(r)
                                                                    
                                                                        if pred[0][idx]>ths:
                                                                            T.append(ths)
                                                                            
                                                                            if validar5 == 'na':

                                                                                if validar6 == 'na':
                                                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                else:
                                                                                    #human
                                                                                    human = detect_human(image_path)      
                                                                                        
                                                                                    if human == False:
                                                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                    else:
                                                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                                                                                        
                                                                            else:
                                                                                det = detect_objects(image_original,validar5,objects,labels)
                                                                                if det == validar5:
                                                                                    
                                                                                    if validar6 == 'na':
                                                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                                    else:
                                                                                        #human
                                                                                        human = detect_human(image_path)      
                                                                                            
                                                                                        if human == False:
                                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                                        else:
                                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                            

                                                                                else:
                                                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                            

                                                                            break
                                                                if T == []:
                                                                    
                                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})

                                                    else:
                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                else:
                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})


                                            else:
                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                            

                            else:
                                if validar4 in class_names or validar4 == 'na':
                                    if validar5 in objects or validar5 in labels or validar5 == 'na':
                                        if validar6 in extras or validar6 == 'na':
                                            if validar4 == 'na':
                                                if validar5 == 'na':
                                                    if validar6 == 'na':
                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                    else:
                                                        #human
                                                        human = detect_human(image_path)      
                                                            
                                                        if human == False:
                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                        else:
                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                            
                                                else:
                                                    det = detect_objects(image_original,validar5,objects,labels)
                                                    if det == validar5:
                                                        if validar6 == 'na':
                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                        else:
                                                            #human
                                                            human = detect_human(image_path)      
                                                                
                                                            if human == False:
                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                            else:
                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                            
                                                    else:
                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                    
                                            else:
                                                pred = detect_scene(image_path,validar4,class_names,class_names_param,x)
                                                
                                                T = []

                                                if pred[0][indx]>thres:
                                                    if validar5 == 'na':

                                                        if validar6 == 'na':
                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                        else:
                                                            #human
                                                            human = detect_human(image_path)      
                                                                
                                                            if human == False:
                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                            else:
                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                            
                                                    else:
                                                        det = detect_objects(image_original,validar5,objects,labels)
                                                        if det == validar5:
                                                            
                                                            if validar6 == 'na':
                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                            else:
                                                                #human
                                                                human = detect_human(image_path)      
                                                                    
                                                                if human == False:
                                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                else:
                                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                    
                                                        else:
                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                    
                                                else:
                                                    if relat != []:
                                                        for r in relat:
                                                            idx = class_names.index(r)
                                                        
                                                            if pred[0][idx]>ths:
                                                                T.append(ths)
                                                                
                                                                if validar5 == 'na':

                                                                    if validar6 == 'na':
                                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                    else:
                                                                        #human
                                                                        human = detect_human(image_path)      
                                                                            
                                                                        if human == False:
                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                        else:
                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                            
                                                                else:
                                                                    det = detect_objects(image_original,validar5,objects,labels)
                                                                    if det == validar5:
                                                                        
                                                                        if validar6 == 'na':
                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                        else:
                                                                            #human
                                                                            human = detect_human(image_path)      
                                                                                
                                                                            if human == False:
                                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                            else:
                                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                                                

                                                                    else:
                                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                

                                                                break
                                                    if T == []:
                                                        
                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})

                                        else:
                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                    else:
                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})


                                else:
                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})

                                
                        else:
                            if validar4 in class_names or validar4 == 'na':
                                if validar5 in objects or validar5 in labels or validar5 == 'na':
                                    if validar6 in extras or validar6 == 'na':
                                        if validar4 == 'na':
                                            if validar5 == 'na':
                                                if validar6 == 'na':
                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                else:
                                                    #human
                                                    human = detect_human(image_path)      
                                                        
                                                    if human == False:
                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                    else:
                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                        
                                            else:
                                                det = detect_objects(image_original,validar5,objects,labels)
                                                if det == validar5:
                                                    if validar6 == 'na':
                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                    else:
                                                        #human
                                                        human = detect_human(image_path)      
                                                            
                                                        if human == False:
                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                        else:
                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                            
                                                else:
                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                
                                        else:
                                            pred = detect_scene(image_path,validar4,class_names,class_names_param,x)
                                            
                                            T = []

                                            if pred[0][indx]>thres:
                                                if validar5 == 'na':

                                                    if validar6 == 'na':
                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                    else:
                                                        #human
                                                        human = detect_human(image_path)      
                                                            
                                                        if human == False:
                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                        else:
                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                            
                                                else:
                                                    det = detect_objects(image_original,validar5,objects,labels)
                                                    if det == validar5:
                                                        
                                                        if validar6 == 'na':
                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                        else:
                                                            #human
                                                            human = detect_human(image_path)      
                                                                
                                                            if human == False:
                                                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                            else:
                                                                return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  

                                                    else:
                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                
                                            else:
                                                if relat != []:
                                                    for r in relat:
                                                        idx = class_names.index(r)
                                                    
                                                        if pred[0][idx]>ths:
                                                            T.append(ths)
                                                            
                                                            if validar5 == 'na':

                                                                if validar6 == 'na':
                                                                    return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                else:
                                                                    #human
                                                                    human = detect_human(image_path)      
                                                                        
                                                                    if human == False:
                                                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                    else:
                                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  
                                                            
                                                            else:
                                                                det = detect_objects(image_original,validar5,objects,labels)
                                                                if det == validar5:
                                                                    
                                                                    if validar6 == 'na':
                                                                        return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})
                                                                    else:
                                                                        #human
                                                                        human = detect_human(image_path)      
                                                                            
                                                                        if human == False:
                                                                            return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                                        else:
                                                                            return jsonify({'Location':True,'Time':True,'Service':True,'Porn':False})                                  

                                                                else:
                                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                                            

                                                            break
                                                if T == []:
                                                    
                                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})

                                    else:
                                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                                else:
                                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})


                            else:
                                return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                            

                    

                    else:
                        return jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})
                
                else:
                    return jsonify({'Location':True,'Time':True,'Service':False,'Porn':True})
                


            else:
                jsonify({'Location':True,'Time':True,'Service':False,'Porn':False})

            
        else:
            return jsonify({'Location':True,'Time':False,'Service':False,'Porn':False})
       
    else:
        return jsonify({'Location':False,'Time':True,'Service':False,'Porn':False})


CORS(app)
@app.after_request
def mysql_con(response):
    #Query a Cloud SQL
    json_respuesta = response.get_json()

    data_a_cloud_sql = [{'User Id':data['id'],'User Name':data['name'],
                        'Mission Id':data['id_mission'],'Mission Name':data['mission_name'],
                        'User Latitude':data['Location_latitude'],'User Longitude':data['Location_longitude'],
                        'Mission Latitude':data['Location_mission_latitude'],'Mission Longitude':data['Location_mission_longitude'],
                        'Start Date Mission':data['Start_Date_mission'],'End Date Mission':data['End_Date_mission'],
                        'Target Time':data['Target_time_mission'],'Radio':data['Location_mission_radio'],
                        'URL':data['url'],'Text':data['text'],
                        'Location':json_respuesta['Location'],'Time':json_respuesta['Time'],
                        'Porn':json_respuesta['Porn'],'Service':json_respuesta['Service']}]
    query_cloud = db.insert(result_data)
    ResultProxy = connection.execute(query_cloud,data_a_cloud_sql)
    
    return response

if __name__ == '__main__':
    
    
    app.run(host='127.0.0.1', port=8080, debug=False)