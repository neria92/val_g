import dlib
import face_recognition
from flask import Flask, request, jsonify
from flask_cors import CORS
import urllib
import datetime
import cv2
import imutils
import numpy as np
from google.cloud import storage
from werkzeug.utils import secure_filename
import six
import matplotlib.image as mpimg
from PIL import Image

app = Flask(__name__)

CORS(app)
def url_to_image2(url):
	"""download the image, convert it to a NumPy array, and then read
	it into OpenCV format"""
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
	"""return the image"""
	return image

CORS(app)

def _get_storage_client():
    return storage.Client()

CORS(app)
def _safe_filename(filename):
    """
    Generates a safe filename that is unlikely to collide with existing objects
    in Google Cloud Storage.
    ``filename.ext`` is transformed into ``filename-YYYY-MM-DD-HHMMSS.ext``
    """
    filename = secure_filename(filename)
    date = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H%M%S")
    basename, extension = filename.rsplit('.', 1)
    return "{0}-{1}.{2}".format(basename, date, extension)

CORS(app)
def upload_file(file_stream, filename):
    """
    Uploads a file to a given Cloud Storage bucket and returns the public url
    to the new object.
    """
    filename = _safe_filename(filename)

    client = _get_storage_client()
    bucket = client.bucket('gchgame.appspot.com')
    blob = bucket.blob('images/' + datetime.datetime.utcnow().strftime("%Y-%m-%d")+ '/' + filename)
    
    blob.upload_from_filename(
        file_stream
        )

    url = blob.public_url

    if isinstance(url, six.binary_type):
        url = url.decode('utf-8')

    return url

CORS(app)
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

CORS(app)
@app.before_first_request
def before():
    global my_faces
    global nombres
    my_faces = []
    images = ['images/ManuelNeria.jpeg','images/Alejandro.JPG','images/Alma.JPG','images/Roberto.JPG',
              'images/olegario.jpg','images/jacobo1.jpg','images/bella.jpg','images/albertob.jpg',
              'images/Rodrigo.JPG']
    nombres = ['manuel_neria','alex','alma','roberto','olegario','jacob','bella','alberto_bazbaz','rodrigo']
    for i in images:
        face = face_recognition.load_image_file(i)
        my_faces.append(face_recognition.face_encodings(face)[0])


CORS(app)
def url_to_image(url):
	urllib.request.urlretrieve(url,'01.jpg')
	return '01.jpg'

CORS(app)
@app.route('/', methods=['POST'])
def face_recog():

    val_face = request.args.get('val_face')
    data = request.json
    url = data['url']
  
    # my_face_encoding now contains a universal 'encoding' of my facial features that can be compared to any other picture of a face!
    image = url_to_image(url)
    unknown_picture = face_recognition.load_image_file(image)
    face_locations = face_recognition.face_locations(unknown_picture)
    unknown_face_encodings = face_recognition.face_encodings(unknown_picture,face_locations)
    
    # Now we can see the two face encodings are of the same person with `compare_faces`!
    found_faces = []
    for face in unknown_face_encodings:
        results = face_recognition.compare_faces(my_faces, face)
        found_faces.extend([nombres[i] if e == True else 'Unknown' for i, e in enumerate(results)])
        #found_faces.extend([nombres[i] for i, e in enumerate(results) if e == True])

    #image = url_to_image2(url)
    #image = imutils.resize(image, width=600)
    #image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = mpimg.imread(image)
    for (top, right, bottom, left), name in zip(face_locations, found_faces):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
    cv2.imwrite('02.jpg',image)

    url2json = upload_image_file('02.jpg')

    if val_face in found_faces:
        return jsonify({'En tu imagen, se encontraron las siguientes caras conocidas':', '.join([i for i in found_faces if i !='Unknown']),
                        'La cara solicitada fue':val_face,
                        'Respuesta Final':'Felicidades se encontró la cara buscada en la imagen',
                        'Puede visualizar la imagen en':url2json})
    else:
        return jsonify({'En tu imagen, se encontraron las siguientes caras conocidas':', '.join([i for i in found_faces if i !='Unknown']),
                        'La cara solicitada fue':val_face,
                        'Respuesta Final':'Lo sentimos la cara solicitada no se ha encontado en la imagen',
                        'Puede visualizar la imagen en':url2json})
    
        


if __name__ == '__main__':
    
    app.run(host='127.0.0.1', port=8080, debug=True)