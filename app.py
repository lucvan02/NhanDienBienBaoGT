from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH ='model.h5'

model = load_model(MODEL_PATH)

def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img
def getClassName(classNo):

    if classNo == 0:
        return 'Gioi han toc do (20km/h)'
    elif classNo == 1:
        return 'Gioi han toc do (30km/h)'
    elif classNo == 2:
        return 'Gioi han toc do (50km/h)'
    elif classNo == 3:
        return 'Gioi han toc do (60km/h)'
    elif classNo == 4:
        return 'Gioi han toc do (70km/h)'
    elif classNo == 5:
        return 'Gioi han toc do (80km/h)'
    elif classNo == 6:
        return 'Het gioi han toc do (80km/h)'
    elif classNo == 7:
        return 'Gioi han toc do (100km/h)'
    elif classNo == 8:
        return 'Gioi han toc do (120km/h)'
    elif classNo == 9:
        return 'Cam vuot'
    elif classNo == 10:
        return 'Cam vuot cho xe co trong luong tren 3.5 tan'
    elif classNo == 11:
        return 'Uu tien tai giao lo ke tiep'
    elif classNo == 12:
        return 'Duong uu tien'
    elif classNo == 13:
        return 'Nhuong duong'
    elif classNo == 14:
        return 'Dung lai'
    elif classNo == 15:
        return 'Cam xe'
    elif classNo == 16:
        return 'Cam xe co trong luong tren 3.5 tan'
    elif classNo == 17:
        return 'Cam di nguoc chieu'
    elif classNo == 18:
        return 'Canh bao nguy hiem'
    elif classNo == 19:
        return 'Curve nguy hiem ben trai'
    elif classNo == 20:
        return 'Curve nguy hiem ben phai'
    elif classNo == 21:
        return 'Curve kep'
    elif classNo == 22:
        return 'Duong gap ghenh'
    elif classNo == 23:
        return 'Duong tron truot'
    elif classNo == 24:
        return 'Duong hep ben phai'
    elif classNo == 25:
        return 'Cong truong dang lam viec'
    elif classNo == 26:
        return 'Den tin hieu giao thong'
    elif classNo == 27:
        return 'Nguoi di bo'
    elif classNo == 28:
        return 'Tre em qua duong'
    elif classNo == 29:
        return 'Qua duong xe dap'
    elif classNo == 30:
        return 'Canh bao bang tuyet/da'
    elif classNo == 31:
        return 'Qua duong dong vat hoang da'
    elif classNo == 32:
        return 'Het moi gioi han toc do va cam vuot'
    elif classNo == 33:
        return 'Re phai phia truoc'
    elif classNo == 34:
        return 'Re trai phia truoc'
    elif classNo == 35:
        return 'Chi di thang'
    elif classNo == 36:
        return 'Di thang hoac re phai'
    elif classNo == 37:
        return 'Di thang hoac re trai'
    elif classNo == 38:
        return 'Giu ben phai'
    elif classNo == 39:
        return 'Giu ben trai'
    elif classNo == 40:
        return 'Bat buoc vong xuyen'
    elif classNo == 41:
        return 'Het cam vuot'
    elif classNo == 42:
        return 'Het cam vuot cho xe co trong luong tren 3.5 tan'
    # if   classNo == 0: return 'Speed Limit 20 km/h'
    # elif classNo == 1: return 'Speed Limit 30 km/h'
    # elif classNo == 2: return 'Speed Limit 50 km/h'
    # elif classNo == 3: return 'Speed Limit 60 km/h'
    # elif classNo == 4: return 'Speed Limit 70 km/h'
    # elif classNo == 5: return 'Speed Limit 80 km/h'
    # elif classNo == 6: return 'End of Speed Limit 80 km/h'
    # elif classNo == 7: return 'Speed Limit 100 km/h'
    # elif classNo == 8: return 'Speed Limit 120 km/h'
    # elif classNo == 9: return 'No passing'
    # elif classNo == 10: return 'No passing for vechiles over 3.5 metric tons'
    # elif classNo == 11: return 'Right-of-way at the next intersection'
    # elif classNo == 12: return 'Priority road'
    # elif classNo == 13: return 'Yield'
    # elif classNo == 14: return 'Stop'
    # elif classNo == 15: return 'No vechiles'
    # elif classNo == 16: return 'Vechiles over 3.5 metric tons prohibited'
    # elif classNo == 17: return 'No entry'
    # elif classNo == 18: return 'General caution'
    # elif classNo == 19: return 'Dangerous curve to the left'
    # elif classNo == 20: return 'Dangerous curve to the right'
    # elif classNo == 21: return 'Double curve'
    # elif classNo == 22: return 'Bumpy road'
    # elif classNo == 23: return 'Slippery road'
    # elif classNo == 24: return 'Road narrows on the right'
    # elif classNo == 25: return 'Road work'
    # elif classNo == 26: return 'Traffic signals'
    # elif classNo == 27: return 'Pedestrians'
    # elif classNo == 28: return 'Children crossing'
    # elif classNo == 29: return 'Bicycles crossing'
    # elif classNo == 30: return 'Beware of ice/snow'
    # elif classNo == 31: return 'Wild animals crossing'
    # elif classNo == 32: return 'End of all speed and passing limits'
    # elif classNo == 33: return 'Turn right ahead'
    # elif classNo == 34: return 'Turn left ahead'
    # elif classNo == 35: return 'Ahead only'
    # elif classNo == 36: return 'Go straight or right'
    # elif classNo == 37: return 'Go straight or left'
    # elif classNo == 38: return 'Keep right'
    # elif classNo == 39: return 'Keep left'
    # elif classNo == 40: return 'Roundabout mandatory'
    # elif classNo == 41: return 'End of no passing'
    # elif classNo == 42: return 'End of no passing by vechiles over 3.5 metric tons'



def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    # PREDICT IMAGE
    predictions = model.predict(img)
    classIndex = np.argmax(predictions, axis=-1)
    # probabilityValue =np.amax(predictions)
    preds = getClassName(classIndex)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001,debug=True)
