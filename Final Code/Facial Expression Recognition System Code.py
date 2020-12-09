from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import time

# parameters for loading data and images
#detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
detection_model_path = 'C:\\Users\\ewhac\\Desktop\\face_classification-master\\face_classification-master\\trained_models\\detection_models\\haarcascade_frontalface_default.xml'
emotion_model_path = 'C:\\Users\\ewhac\\Desktop\\face_classification-master\\_mini_XCEPTION.61-0.63.hdf5'
#emotion_model_path = 'models/_mini_XCEPTION.106-0.65.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]


# starting video streaming
cv2.namedWindow('your_face')
camera = cv2.VideoCapture(0)
prevTime = 0
while True:
    frame = camera.read()[1]
    
    curTime = time.time()
    sec = curTime - prevTime
    prevTime = curTime
    fps = 1/(sec)
    print ("Estimated fps{0}".format(fps))
    
    #reading the frame
    frame = imutils.resize(frame,width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    
    canvas = np.zeros((200, 450, 3), dtype="uint8")
    canvas.fill(255)
    can_h = canvas.shape[0]
    can_w = canvas.shape[1]
    frameClone = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
            # Extract the ROI of the face from the grayscale image, resize it to a fixed 48x48 pixels, and then prepare
            # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0) #관심영역 
        
        
        preds = emotion_classifier.predict(roi)[0] #roi 관심영역의 얼굴을 emotion_classifier로 통해 np에 저장. 수치화.
        #emotion_probability = np.max(preds)
        
        label = EMOTIONS[preds.argmax()] #예측한 것 중 가장 높은 수치를 띄는 emotion을 label로 확정. 
        
        if label == 'angry':
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),(0, 0, 255), 2) #red
            canvas[:] = (0, 0, 255)
        elif label == 'disgust':
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),(182, 121, 157), 2) #purple
            canvas[:] = (182, 121, 157)
        elif label == 'scared':
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),(0, 150, 0), 2) #deep green
            canvas[:] = (0, 150, 0)
        elif label == 'happy':
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),(84, 255, 255), 2) #yellow
            canvas[:] = (84, 255, 255)
        elif label == 'sad':
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),(255, 81, 81), 2) #blue
            canvas[:] = (255, 81, 81)
        elif label == 'surprised':
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),(255, 189, 89), 2) #light blue
            canvas[:] = (255, 189, 89)
        elif label == 'neutral':
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),(0, 255, 0), 2) #light green
            canvas[:] = (0, 255, 0)
        else:
            print("error")
        
        textsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 3, 5)[0]
        textX = (can_w - textsize[0])/2
        textY = (can_h + textsize[1])/2
        org = ((int)(textX),(int)(textY))
        cv2.putText(canvas, label, org, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5) #text 가운데 정렬
        
    cv2.imshow('your_face', frameClone)
    cv2.moveWindow('your_face', 750, 250) #창의 위치 고정
    cv2.imshow("text", canvas)
    cv2.moveWindow('text', 300, 250) #창의 위치 고정
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()


