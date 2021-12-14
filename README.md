# tibame_group_project

## 前置作業
1. 下載此包專案
```
git clone https://github.com/KakaCheng/tibame_group_project.git
```
2. 於根目錄下建立/models資料夾
    - 放置**model.h5**\
(請使用[Kaggle]HumanEmotion_20211207.ipynb進行訓練)

3. 安裝套件
    - 若安裝dlib出現錯誤時，請參考\[2]
```
pip install -r requirements.txt
```

## 演算法
### 訓練模型
1. 使用Kaggle人臉情緒資料集\[3]
![image](https://github.com/KakaCheng/tibame_group_project/blob/41ed239efb17a799b0020629882327090b9df3d1/pic/001.png)

2. 建立模型並對anger, happy, sad, neutral四種情緒進行辨識\[1]
![image](https://github.com/KakaCheng/tibame_group_project/blob/41ed239efb17a799b0020629882327090b9df3d1/pic/002.png)

3. 辨識正確率
    - valid: 67.1%\
![image](https://github.com/KakaCheng/tibame_group_project/blob/41ed239efb17a799b0020629882327090b9df3d1/pic/005.png)

    - private test: 69.3%\
![image](https://github.com/KakaCheng/tibame_group_project/blob/41ed239efb17a799b0020629882327090b9df3d1/pic/003.PNG)


### 使用模型
1. 使用dlib(HoG + Linear SVM)抓取當前圖片的人臉位置
2. 使用已訓練好的模型
3. 輸出辨識後結果
![image](https://github.com/KakaCheng/tibame_group_project/blob/41ed239efb17a799b0020629882327090b9df3d1/pic/006.png)



## 參考資料 reference
\[1]: https://github.com/atulapra/Emotion-detection \
\[2]: https://www.youtube.com/watch?v=AUJKdehF2ZA \
\[3]: https://www.kaggle.com/deadskull7/fer2013
