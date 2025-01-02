from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# 設定工作目錄為程式實際所在資料夾
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 載入已訓練的模型
model = load_model("QuickDraw_v2.h5")

# 定義圖片處理和預測函數
def preprocess_image(image_path):
    # 讀取圖片並轉為灰度模式
    img = Image.open(image_path).convert('L')
    # 調整圖片大小為28x28
    img = img.resize((28, 28))
    # 將圖片數據歸一化至0到1之間
    img_array = np.array(img) / 255.0
    # 調整維度為(1, 28, 28, 1)，以符合模型的輸入要求
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

def predict_image(image_path):
    # 預處理圖片
    img_array = preprocess_image(image_path)
    # 使用模型進行預測
    prediction = model.predict(img_array)
    # 找到預測結果的最大概率對應的類別
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class[0]

# 回傳預測結果
def test_image(image_path):
    predicted_class = predict_image(image_path)
    return predicted_class
