import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用 oneDNN 操作
import cv2
import numpy as np
from Hand_Detect import NumberofFingers
import tensorflow as tf
import test_predict_draw as predict

cap = cv2.VideoCapture(0) #數值是0代表讀取電腦的攝影機鏡頭
_, image = cap.read() #讀取攝影機的畫面
model = tf.keras.models.load_model("QuickDraw_v2.keras") #載入模型

drawing = False #紀錄變數畫畫開始變數
startdrawing = False
positions_x=[] #紀錄食指位置x[8]變數
positions_y=[] #紀錄食指位置y[8]變數
pred_class = None
count = 0
cv2.namedWindow("paint") #畫布視窗
paintWindow = cv2.imread('black.jpg') #黑色畫布
cv2.imshow("paint", paintWindow) #顯示畫布

while cap.isOpened(): #每一楨影像都可以視為一張照片 一次while迴圈 就是處理一楨影像(一張照片)
    image, handimage, finger, indexfinger = NumberofFingers(cap)  # 執行函數
    
    if(finger==1): # 食指->開始畫畫模式，為了避免前面亂畫，先等20楨
        if count < 20 and startdrawing == False:
            count += 1
        else:
            count = 0
            drawing=True
            startdrawing = True
    elif (finger==5 and startdrawing == True):#結束畫畫模式
        #crop
        xmin=int(min(positions_x))-10
        xmax=int(max(positions_x))+10
        ymin=int(min(positions_y))-10
        ymax=int(max(positions_y))+10
        print(xmin,xmax,ymin,ymax)
        paintWindow = paintWindow*255.0 / paintWindow.max()
        print(paintWindow.shape[0],paintWindow.shape[1])
        if(xmin<0):
            xmin=0
        if(ymin<0):
            ymin=0
        if(xmax>600):
            xmax=600
        if(ymax>600):
            ymax=600
        paintWindow = paintWindow[ymin:ymax, xmin:xmax]
        cv2.imwrite("newestpaint.png", paintWindow) #save
        pred_class, confidence = predict.predict_drawing(model, "newestpaint.png") #predict
        if confidence > 50:
            print(f"預測結果: {pred_class}")
        else:
            print("無法辨識，請再試一次><")

        #clear&reload
        drawing=False
        startdrawing = False
        positions_x=[] 
        positions_y=[] 
        paintWindow =cv2.imread('black.jpg')
        image, handimage, finger, indexfinger = NumberofFingers(cap)  # 執行函數
    else: # 食指伸出->開始畫畫模式，為了避免前面亂畫，先等20楨
        drawing = False

    if drawing == True and finger==1: #畫畫模式
        positions_x.append(indexfinger[0])
        positions_y.append(indexfinger[1])

    if len(positions_x)!=0:
        for i in range (0,len(positions_x)):
            cv2.rectangle(image,(int(positions_x[i]),int(positions_y[i])),(int(positions_x[i]+3),int(positions_y[i]+3)),(0, 0, 0),-1)
            cv2.rectangle(paintWindow,(int(positions_x[i]),int(positions_y[i])),(int(positions_x[i]+3),int(positions_y[i]+3)),(255, 255, 255),-1)
            #print(mediapipeHand.keypointsx[8],mediapipeHand.keypointsy[8])
        cv2.line(paintWindow, (int(positions_x[i]),int(positions_y[i])), (int(positions_x[i-1]),int(positions_y[i-1])), (255, 255, 255),5)
        cv2.line(image, (int(positions_x[i]),int(positions_y[i])), (int(positions_x[i-1]),int(positions_y[i-1])), (255, 0, 0),3)

    # if pred_class is not None: # 預測結果顯示區
    #     cv2.putText(image, "You are drawing " + pred_class, (150, 25),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
    #顯示視訊畫面
    cv2.imshow('MediaPipe Hands',image)
    cv2.imshow("paint", paintWindow) #顯示畫布


    #按q可以關閉視窗
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
