from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from error_map import emotions_detection_err

import numpy as np
import cv2

class emotions_detection():

    def __init__(self):

        #初始化人臉偵測模型
        self.facecasc = cv2.CascadeClassifier("./models/haarcascade_frontalface_default.xml")

        #初始化情緒模型
        self.model = self.init_model()

        #建立情緒對應表
        self.emotion_dict = {
            0: "Angry",
            1: "Disgusted",
            2: "Fearful",
            3: "Happy",
            4: "Neutral",
            5: "Sad",
            6: "Surprised"
        }
        
    def init_model(self):
        #創建模型網路層
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

        #載入訓練好的參數
        model.load_weights("models/model.h5")
        return model

    def detect_emotions(self, frame):
        
        res = emotions_detection_err.SUCCESS
        face_emotions = list()
        frame_modify = frame.copy()

        gray = cv2.cvtColor(frame_modify, cv2.COLOR_BGR2GRAY)
        faces = self.facecasc.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 5)
        
        for (x, y, w, h) in faces:

            #進行情緒偵測
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = np.round(self.model.predict(cropped_img) * 100.0, decimals = 3) #將預測值轉換成百分比，四捨五入至小數點第三位

            face_info = (x, y, w, h)
            emotions_pred =  list(zip(self.emotion_dict.values(), list(prediction[0])))
            face_emotions += [[face_info, emotions_pred]]

            cv2.rectangle(frame_modify, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            cv2.putText(frame_modify, self.emotion_dict[int(np.argmax(prediction))], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        return res, frame_modify, face_emotions

    def detect_emotions_camera(self):
        video_cap = cv2.VideoCapture(0)
        while True:
            ret, frame = video_cap.read()
            if not ret: #影像擷取失敗
                break

            res, frame_modify, face_emotions = self.detect_emotions(frame)
            cv2.imshow('Video', frame_modify)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_cap.release()
        cv2.destroyAllWindows()
