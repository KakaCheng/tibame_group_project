from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from error_map import emotions_detection_err
from sklearn.preprocessing import LabelEncoder

import dlib
import numpy as np
import cv2

class emotions_detection():

    def __init__(self):

        #初始化人臉偵測模型
        self.facedlib = dlib.get_frontal_face_detector()
        #初始化情緒模型
        self.model = self.init_model()

        #建立情緒對應表
        self.emotion_dict = {
            0: "Angry",
            #1: "Disgusted",
            #2: "Fearful",
            3: "Happy",
            4: "Sad",
            # 5: "surprise",
            6: "Neutral"
        }

        #載入標籤
        self.le = LabelEncoder()
        self.le.classes_ = np.load("models/label.npy")
        
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
        model.add(Dense(4, activation='softmax'))

        #載入訓練好的參數
        model.load_weights("models/best_model.h5")
        return model

    def detect_emotions(self, frame):
        
        res = emotions_detection_err.SUCCESS
        frame_modify = frame.copy()

        gray = cv2.cvtColor(frame_modify, cv2.COLOR_BGR2GRAY)
        
        faces, scores, idx = self.facedlib.run(gray, 0)

        face_info = dict()
        i = 0
        for count in faces:
            x1 = count.left()
            y1 = count.top()
            x2 = count.right()
            y2 = count.bottom()

            tmp = dict()
            #進行情緒偵測
            roi_gray = gray[y1:y2, x1:x2]
            if len(roi_gray) == 0:
                break
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = self.model.predict(cropped_img)

            pre_emotion_inv = self.le.inverse_transform(prediction.argmax(axis=1))
            cv2.rectangle(frame_modify, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame_modify, self.emotion_dict[pre_emotion_inv[0]], (x1+20, y1-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            tmp["face_pos"] = np.array([x1, y1, x2, y2]).tolist()
            tmp["emotions_pred"] = self.emotion_dict[pre_emotion_inv[0]]
            face_info[i] = tmp
            i += 1

        return res, frame_modify, face_info

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
