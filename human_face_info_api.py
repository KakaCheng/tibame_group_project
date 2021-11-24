from emotion_detection_method.emotions_detection_module import emotions_detection
from error_map import *

class human_face_info_API():

    def __init__(self):
        self.emotions = emotions_detection()

    def get_emotions_detection(self, frame):
        res = human_face_info_API_err.SUCCESS
        res, frame_modify, face_emotions = self.emotions.detect_emotions(frame)

        if res != emotions_detection_err.SUCCESS:
            return human_face_info_API_err.DETECT_EMOTION_FAIL, [], []

        return human_face_info_API_err.SUCCESS, frame_modify, face_emotions

    def demo_emotions_detection(self):
        return self.emotions.detect_emotions_camera()


