from emotion_detection_method.emotions_detection_module import emotions_detection
from error_map import *
import numpy as np
import json
import cv2
# import base64

class human_face_info_API():

    def __init__(self):
        self.emotions = emotions_detection()

    def get_emotions_detection(self, frame):
        res = human_face_info_API_err.SUCCESS
        res, frame_modify, face_info = self.emotions.detect_emotions(frame)

        if res != emotions_detection_err.SUCCESS:
            return human_face_info_API_err.DETECT_EMOTION_FAIL, [], []

        # self.content_to_json(frame_modify, face_info)
        return human_face_info_API_err.SUCCESS, frame_modify, face_info

    def demo_emotions_detection(self):
        return self.emotions.detect_emotions_camera()
    
    def save_to_json(self, file_path, output_full_path, frame_modify, face_info):
        
        frame_info_dict = dict()
        frame_info_dict.update(face_info)
        frame_info_dict["output_path"] = output_full_path
        # #numpy.array轉byte並以ascii形式儲存
        # frame_info_dict["img"] = base64.b64encode(frame_modify).decode('ascii')
        
        #儲存
        try:
            with open(file_path, "w+") as fp:
                #轉成json格式
                frame_info_json = json.dump(frame_info_dict, fp)
        except:
            return human_face_info_API_err.WRITE_JSON_FILE_FAIL

        try:
            cv2.imwrite(output_full_path, frame_modify)
        except:
            return human_face_info_API_err.WRITE_PNG_FILE_FAIL

        return human_face_info_API_err.SUCCESS




