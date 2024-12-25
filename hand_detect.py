"""
修改自：https://github.com/ANANTH-SWAMY/NUMBER-DETECTION-WITH-MEDIAPIPE/blob/main/number_detection.py
變成只偵測 0, 1, 5支手指，並顯示於左下角 (p.s. 忽略大拇指)
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用 oneDNN 操作
import cv2
import mediapipe

cap = cv2.VideoCapture(0)  # 開啟攝像頭

# Mediapipe 手部偵測模型
medhands = mediapipe.solutions.hands
hands = medhands.Hands(max_num_hands=1, min_detection_confidence=0.7)  # 設定最大檢測手數量及最低信心度
draw = mediapipe.solutions.drawing_utils  # 用於繪製手部關鍵點

def NumberofFingers(cap):
    success, img = cap.read()  # 讀取影像
    img = cv2.flip(img, 1)  # 翻轉影像，因為攝像頭會有鏡像效果
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 將影像從 BGR 轉為 RGB 顏色空間
    res = hands.process(imgrgb)  # 進行手部偵測

    nohandimg = img.copy()  # 複製一份影像，用於顯示原圖
    lmlist = []  # 儲存所有的關鍵點座標
    tipids = [4, 8, 12, 16, 20]  # 手指尖的關鍵點 ID
    
    # 在畫面中畫一個矩形框，顯示手指數量的位置
    cv2.rectangle(img, (20, 350), (90, 440), (0, 255, 204), cv2.FILLED)
    cv2.rectangle(img, (20, 350), (90, 440), (0, 0, 0), 5)
    
    if res.multi_hand_landmarks:  # 如果檢測到手部
        for handlms in res.multi_hand_landmarks:  # 對每隻手進行處理
            for id, lm in enumerate(handlms.landmark):  # 對每個關鍵點進行處理
                h, w, c = img.shape  # 取得影像的高度、寬度和通道數
                cx, cy = int(lm.x * w), int(lm.y * h)  # 計算每個關鍵點在影像中的座標
                lmlist.append([id, cx, cy])  # 儲存每個關鍵點的 ID 和座標
                
                if len(lmlist) != 0 and len(lmlist) == 21:  # 確保所有的關鍵點已經被處理
                    fingerlist = []  # 儲存每根手指的狀態（伸出或收回）
                    '''
                    # # 處理大拇指，並考慮翻轉情況
                    # if lmlist[12][1] > lmlist[20][1]:
                    #     if lmlist[tipids[0]][1] > lmlist[tipids[0] - 1][1]:
                    #         fingerlist.append(1)  # 大拇指伸出
                    #     else:
                    #         fingerlist.append(0)  # 大拇指收回
                    # else:
                    #     if lmlist[tipids[0]][1] < lmlist[tipids[0] - 1][1]:
                    #         fingerlist.append(1)  # 大拇指伸出
                    #     else:
                    #         fingerlist.append(0)  # 大拇指收回
                    '''
                    # 處理其他手指
                    for id in range(1, 5):
                        if lmlist[tipids[id]][2] < lmlist[tipids[id] - 2][2]:  # 如果指尖高於上一個關鍵點，則表示手指伸出
                            fingerlist.append(1)
                        else:
                            fingerlist.append(0)
                    
                    if len(fingerlist) != 0:
                        fingercount = fingerlist.count(1)  # 計算伸出的手指數量
                    if fingercount == 1 and fingerlist[0] == 1: # 畫畫模式
                        fingercount = 1
                    elif fingercount < 4: # 除了食指以外的手指伸出或手完全張開，都視為 0
                        fingercount = 0
                    else: # 結束畫畫模式
                        fingercount = 5
                    
                    # 顯示手指數量
                    cv2.putText(img, str(fingercount), (25, 430), cv2.FONT_HERSHEY_PLAIN, 6, (0, 0, 0), 5)
                    # 畫出手部的關鍵點及其連線
                    draw.draw_landmarks(img, handlms, medhands.HAND_CONNECTIONS, 
                                    draw.DrawingSpec(color=(0, 255, 204), thickness=2, circle_radius=2),
                                    draw.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=3))
                    # 回傳原圖、只有手部的圖、手指數量、食指座標
                    return img, nohandimg, fingercount, [lmlist[8][1], lmlist[8][2]] 
    return img, nohandimg, 0, [0, 0]
