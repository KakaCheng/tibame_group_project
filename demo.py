from human_face_info_api import *
import cv2


if __name__ == "__main__":
    test = human_face_info_API()
    img = cv2.imread("test.jpg")
    res, img_modify, face_emotions = test.get_emotions_detection(img)

    cv2.imshow("test", img_modify)
    cv2.waitKey(0)
    cv2.destroyAllWindows()