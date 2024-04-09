import numpy as np
import cv2
import pickle
import urllib.request
from keras.models import load_model

# CAMERA RESOLUTION
frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.75 
font = cv2.FONT_HERSHEY_SIMPLEX


# # SETUP CAMERA
# cap = cv2.VideoCapture(0)

url = 'http://192.168.137.227/cam.jpg'
# cv2.namedWindow("live Cam Testing", cv2.WINDOW_AUTOSIZE)
# Create a VideoCapture object
# cap = cv2.VideoCapture(url)

# cap.set(3, frameWidth)
# cap.set(4, frameHeight)
# cap.set(10, brightness)

# Load model
model = load_model("model.h5")


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img


def getClassName(classNo):
    if classNo == 0:
        return 'Gioi han toc do (30km/h)'
    elif classNo == 1:
        return 'Giao nhau voi duong uu tien'
    elif classNo == 2:
        return 'Cong trinh dang thi cong'

    

while True:
    # READ IMAGE
    imgResp = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    imgOrignal = cv2.imdecode(imgNp, -1)

    # PROCESS IMAGE
    img = cv2.resize(imgOrignal, (32, 32))
    img = preprocessing(img)
    # cv2.imshow("Processed Image", img)

    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOrignal, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    # PREDICT IMAGE
    predictions = model.predict(img)
    classIndex = np.argmax(predictions)
    probabilityValue = np.amax(predictions)
    cv2.putText(imgOrignal, str(classIndex)+" "+str(getClassName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, str(round(probabilityValue*100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Result", imgOrignal)

    k = cv2.waitKey(1) 
    if k == ord('q'):
        break

cv2.destroyAllWindows()