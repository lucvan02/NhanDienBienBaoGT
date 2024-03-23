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

url = 'http://192.168.1.87/capture.jpg'
cv2.namedWindow("live Cam Testing", cv2.WINDOW_AUTOSIZE)
# Create a VideoCapture object
cap = cv2.VideoCapture(url)

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

# def getClassName(classNo):
#     if classNo == 0:
#         return 'Gioi han toc do (20km/h)'
#     elif classNo == 1:
#         return 'Gioi han toc do (30km/h)'
#     elif classNo == 2:
#         return 'Gioi han toc do (50km/h)'
#     elif classNo == 3:
#         return 'Gioi han toc do (60km/h)'
#     elif classNo == 4:
#         return 'Gioi han toc do (70km/h)'
#     elif classNo == 5:
#         return 'Gioi han toc do (80km/h)'
#     elif classNo == 6:
#         return 'Het gioi han toc do (80km/h)'
#     elif classNo == 7:
#         return 'Gioi han toc do (100km/h)'
#     elif classNo == 8:
#         return 'Gioi han toc do (120km/h)'
#     elif classNo == 9:
#         return 'Cam vuot'
#     elif classNo == 10:
#         return 'Cam vuot cho xe co trong luong tren 3.5 tan'
#     elif classNo == 11:
#         return 'Uu tien tai giao lo ke tiep'
#     elif classNo == 12:
#         return 'Duong uu tien'
#     elif classNo == 13:
#         return 'Nhuong duong'
#     elif classNo == 14:
#         return 'Dung lai'
#     elif classNo == 15:
#         return 'Cam xe co gioi'
#     elif classNo == 16:
#         return 'Cam xe co trong luong tren 3.5 tan'
#     elif classNo == 17:
#         return 'Cam di nguoc chieu'
#     elif classNo == 18:
#         return 'Canh bao nguy hiem'
#     elif classNo == 19:
#         return 'Duong cong nguy hiem ben trai'
#     elif classNo == 20:
#         return 'Duong cong nguy hiem ben phai'
#     elif classNo == 21:
#         return 'Duong cong kep'
#     elif classNo == 22:
#         return 'Duong gap ghenh'
#     elif classNo == 23:
#         return 'Duong tron truot'
#     elif classNo == 24:
#         return 'Duong hep ben phai'
#     elif classNo == 25:
#         return 'Cong truong dang thi cong'
#     elif classNo == 26:
#         return 'Den tin hieu giao thong'
#     elif classNo == 27:
#         return 'Nguoi di bo'
#     elif classNo == 28:
#         return 'Tre em qua duong'
#     elif classNo == 29:
#         return 'Giao nhau voi xe dap'
#     elif classNo == 30:
#         return 'Canh bao bang tuyet/da'
#     elif classNo == 31:
#         return 'Gap dong vat hoang da bang qua'
#     elif classNo == 32:
#         return 'Het moi gioi han toc do va cam vuot'
#     elif classNo == 33:
#         return 'Re phai phia truoc'
#     elif classNo == 34:
#         return 'Re trai phia truoc'
#     elif classNo == 35:
#         return 'Chi di thang'
#     elif classNo == 36:
#         return 'Di thang hoac re phai'
#     elif classNo == 37:
#         return 'Di thang hoac re trai'
#     elif classNo == 38:
#         return 'Giu ben phai'
#     elif classNo == 39:
#         return 'Giu ben trai'
#     elif classNo == 40:
#         return 'Bat buoc vong xuyen'
#     elif classNo == 41:
#         return 'Het cam vuot'
#     elif classNo == 42:
#         return 'Het cam vuot cho xe co trong luong tren 3.5 tan'

    
# while True:
#     # READ IMAGE
#     success, imgOrignal = cap.read()
    
#     # PROCESS IMAGE
#     img = np.asarray(imgOrignal)
#     img = cv2.resize(img, (32, 32))
#     img = preprocessing(img)
#     cv2.imshow("Processed Image", img)
#     img = img.reshape(1, 32, 32, 1)
#     cv2.putText(imgOrignal, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
#     cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
#     # PREDICT IMAGE
#     predictions = model.predict(img)
#     # classIndex = model.predict_classes(img)
#     classIndex = np.argmax(predictions)
#     probabilityValue =np.amax(predictions)
#     cv2.putText(imgOrignal,str(classIndex)+" "+str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
#     cv2.putText(imgOrignal, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
#     cv2.imshow("Result", imgOrignal)

#     k=cv2.waitKey(1) 
#     if k== ord('q'):
#         break

# cv2.destroyAllWindows()
# cap.release()
    

while True:
    # READ IMAGE
    imgResp = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    imgOrignal = cv2.imdecode(imgNp, -1)

    # PROCESS IMAGE
    img = cv2.resize(imgOrignal, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)

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
