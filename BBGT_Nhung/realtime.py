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


# SETUP CAMERA
cap = cv2.VideoCapture(0)

# url = 'http://10.252.3.215/cam-hi.jpg'
# cv2.namedWindow("live Cam Testing", cv2.WINDOW_AUTOSIZE)
# # Create a VideoCapture object
# cap = cv2.VideoCapture(url)

cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# Load model
model = load_model("modelgoc.h5")


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


# def getCalssName(classNo):
#     if classNo == 0:
#         return 'Gioi han toc do (30km/h)'
#     elif classNo == 1:
#         return 'Giao nhau voi duong uu tien'
#     elif classNo == 2:
#         return 'Cong trinh dang thi cong'


def getCalssName(classNo):
    if   classNo == 0: return 'Gioi han toc do (20km/h)'
    elif classNo == 1: return 'Gioi han toc do (30km/h)'
    elif classNo == 2: return 'Gioi han toc do (50km/h)'
    elif classNo == 3: return 'Gioi han toc do (60 km/h)'
    elif classNo == 4: return 'Gioi han toc do (70 km/h)'
    elif classNo == 5: return 'Gioi han toc do (80 km/h)'
    elif classNo == 6: return 'Het gioi han toc do (80 km/h)'
    elif classNo == 7: return 'Gioi han toc do (100 km/h)'
    elif classNo == 8: return 'Gioi han toc do (120 km/h)'
    elif classNo == 9: return 'Cam vuot'
    elif classNo == 10: return 'Cam vuot xe trn 3,5 tan'
    elif classNo == 11: return 'Quyen uu tin tai giao lo tiep theo'
    elif classNo == 12: return 'duong uu tin'
    elif classNo == 13: return 'nang suat'
    elif classNo == 14: return 'Dung lai'
    elif classNo == 15: return 'khong ci phuong tien'
    elif classNo == 16: return 'Cam xe trn 3,5 tan'
    elif classNo == 17: return 'Cam vo'
    elif classNo == 18: return 'Than trong chung'
    elif classNo == 19: return 'Khc cua nguy hiem bn tri'
    elif classNo == 20: return 'Khc cua nguy hiem bn phai '
    elif classNo == 21: return 'duong cong doi'
    elif classNo == 22: return 'Con duong gap ghenh'
    elif classNo == 23: return 'duong tron truot'
    elif classNo == 24: return 'Con duong hep lai bn phai'
    elif classNo == 25: return 'lm duong'
    elif classNo == 26: return 'Tin hieu giao thong'
    elif classNo == 27: return 'nguoi di bo'
    elif classNo == 28: return 'tre em qua duong'
    elif classNo == 29: return 'Xe dap qua duong'
    elif classNo == 30: return 'Coi chung bang/tuyet'
    elif classNo == 31: return 'Dong vat hoang d bang qua'
    elif classNo == 32: return 'Ket thc tat ca cc gioi han toc do v vuot qua'
    elif classNo == 33: return 'Re phai ve phia truoc'
    elif classNo == 34: return 'Re tri phia truoc'
    elif classNo == 35: return 'Chi di thang'
    elif classNo == 36: return 'Di thang hoac phai'
    elif classNo == 37: return 'Di thang hoac tri'
    elif classNo == 38: return 'Di bn phai'
    elif classNo == 39: return 'Giu bn tri'
    elif classNo == 40: return 'Bat buoc di vong xuyen'
    elif classNo == 41: return 'Ket thc khong di qua'
    elif classNo == 42: return 'Het duong cam xe trn 3,5 tan di qua'


# stop = 0

# while True:
#     # READ IMAGE
#     success, imgOrignal = cap.read()

#     # PROCESS IMAGE
#     img = np.asarray(imgOrignal)
#     img = cv2.resize(img, (32, 32))
#     img = preprocessing(img)
#     cv2.imshow("Processed Image", img)
#     img = img.reshape(1, 32, 32, 1)
#     cv2.putText(imgOrignal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
#     cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
#     # Du doan
#     if (stop == 0):
#         predictions = model.predict(img)
#         classIndex = np.argmax(predictions, axis=-1)
#         probabilityValue = np.amax(predictions)
#         cv2.putText(imgOrignal, str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
#         cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255), 2,
#                     cv2.LINE_AA)
#     str_result = ""
#     str_probility = ""
#     if (round(probabilityValue * 100, 2) > 99):
#         str_result = str(getCalssName(classIndex))
#         str_probility = str(round(probabilityValue * 100, 2))
#         cv2.putText(imgOrignal, str_result, (120, 35), font, 0.75,
#                     (0, 0, 255), 2, cv2.LINE_AA)
#         cv2.putText(imgOrignal, str_probility + "%", (180, 75), font, 0.75, (0, 0, 255), 2,
#                     cv2.LINE_AA)
#         print("=============>" + str_result + " " + str_probility)
#         stop = 1
#     cv2.imshow("Result", imgOrignal)
#     k = cv2.waitKey(1)
#     if k == ord('q'):
#         break
#     if k == ord('r'):
#         stop = 0

# cv2.destroyAllWindows()
# cap.release()
    
while True:
    # READ IMAGE
    success, imgOrignal = cap.read()
    
    # PROCESS IMAGE
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOrignal, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # PREDICT IMAGE
    predictions = model.predict(img)
    # classIndex = model.predict_classes(img)
    classIndex = np.argmax(predictions)
    probabilityValue =np.amax(predictions)
    cv2.putText(imgOrignal,str(classIndex)+" "+str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Result", imgOrignal)

    k=cv2.waitKey(1) 
    if k== ord('q'):
        break

cv2.destroyAllWindows()
cap.release()