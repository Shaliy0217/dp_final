# 現實畫圖小遊戲
## 目前已確認不須用到 -> 切換為
doodle train: 原專案用於訓練 Quick Draw圖片的  -> TrainQuickDraw.py
QuickDraw_v2.h5 -> QuickDraw_v2.keras

Hand_CNN.h5 -> 改用 hand_detect.py 直接判斷
HandGestureXXX... 因直接使用mediapipe，不須再做訓練與額外判斷

## 檔案用途介紹:
Download_data.py: 下載 Quick Draw 每種類各500張，但一堆看起來就分類錯的:( 
TrainQuickDraw.py: 訓練模型 QuickDraw_v2.keras，圖片原大小28*28、池化卷積3次、卷積核為5*5、共5種圖形
test_predict_drawing.py: predict_drawing函式可輸入一張

Hand_Detect.py: 偵測有幾隻手指舉著，簡化成食指舉著(畫畫)、手張開(結束畫畫)、拳頭(暫停畫畫)


game.py: 遊戲運行主程式
to_model.py: 使用訓練好的模型來判斷圖片
demo_ui.py: 遊戲的ui介面