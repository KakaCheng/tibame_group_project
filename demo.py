from human_face_info_api import *
import cv2
import argparse
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "[訊息][demo] 情緒辨識指令")
    parser.add_argument("-fp", "--FilePath", type = str, default = None,
                        help = "傳入欲做情緒辨識的檔案路徑，例如: ./XXX/XXX/")
    parser.add_argument("-fn", "--FileName", type = str, default = None,
                        help = "欲做情緒辨識的檔案名稱，例如: test")
    parser.add_argument("-sp", "--SavePath", type = str, default = "./", 
                        help = "將做完情緒辨識的檔案輸出至欲儲存的路徑下，例如: ./XXX/XXX/")

    args = parser.parse_args()

    if args.FilePath is not None and args.FileName is not None: #檢查路徑和檔案名稱是否未輸入

        #結合路徑與檔案名稱
        img_full_path = args.FilePath + args.FileName + ".png"
        output_full_path = args.SavePath + args.FileName + "_out" + ".png"
        json_full_path = args.SavePath + args.FileName + ".json"

        if os.path.isfile(img_full_path) and os.path.isdir(args.SavePath): #檢查檔案和資料夾是否存在

            #初始化
            human_face_info = human_face_info_API()
            
            #讀圖
            img = cv2.imread(img_full_path)

            #情緒辨識
            res, img_modify, face_emotions = human_face_info.get_emotions_detection(img)
            if res != human_face_info_API_err.SUCCESS:
                print("[錯誤][demo]: res = " + res)
                exit()
            
            #存成json檔
            res = human_face_info.save_to_json(json_full_path, output_full_path, img_modify, face_emotions)

            # cv2.imshow(args.FileName, img_modify)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        else:
            print("[錯誤][demo]: args.FilePath = {}, args.SavePath = {}. 該檔案或資料夾未存在".format(args.img_full_path, args.SavePath))
            exit()
    else:
        print("[錯誤][demo]: args.FilePath = {}, args.FileName = {}.".format(args.FilePath, args.FileName))
        exit()

    # human_face_info = human_face_info_API()
    # human_face_info.demo_emotions_detection()