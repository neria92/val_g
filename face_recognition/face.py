import dlib
import face_recognition
from flask import Flask, request, jsonify
from flask_cors import CORS
import urllib
import numpy as np


app = Flask(__name__)


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
    unknown_face_encodings = face_recognition.face_encodings(unknown_picture)
    
    # Now we can see the two face encodings are of the same person with `compare_faces`!
    found_faces = []
    for face in unknown_face_encodings:
        results = face_recognition.compare_faces(my_faces, face)
        found_faces.extend([nombres[i] for i, e in enumerate(results) if e == True])
    print(found_faces)

    if val_face in found_faces:
        return jsonify({'En tu imagen, se encontraron las siguientes caras conocidas':', '.join(found_faces),
                        'La cara solicitada fue':val_face,
                        'Respuesta Final':'Felicidades se encontró la cara buscada en la imagen'})
    else:
        return jsonify({'En tu imagen, se encontraron las siguientes caras conocidas':', '.join(found_faces),
                        'La cara solicitada fue':val_face,
                        'Respuesta Final':'Lo sentimos la cara solicitada no se ha encontado en la imagen'})
    
        


if __name__ == '__main__':
    
    app.run(host='127.0.0.1', port=8080, debug=True)