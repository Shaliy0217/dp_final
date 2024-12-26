import numpy as np
from PIL import Image
import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用 oneDNN 操作
import tensorflow as tf

# 類別標籤
categories = ['circle', 'envelope', 'triangle', 'star', 'square']

def preprocess_image(image_path):
    # 載入並預處理圖片
    img = Image.open(image_path).convert('L')  # 轉換為灰階
    img = img.resize((28, 28))  # 調整大小為 28x28
    img_array = np.array(img)
    img_array = img_array / 255.0  # 正規化
    img_array = img_array.reshape(1, 28, 28, 1)  # 重塑為模型輸入格式
    return img_array

def predict_drawing(model, image_path):
    # 載入模型
    
    # 預處理圖片
    processed_image = preprocess_image(image_path)
    
    # 預測
    prediction = model.predict(processed_image)
    predicted_class = categories[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    return predicted_class, confidence

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方式: python predict.py <圖片路徑>")
        sys.exit(1)
        
    model_path = "QuickDraw_v2.keras"
    image_path = sys.argv[1]
    
    try:
        model = tf.keras.models.load_model(model_path)
        predicted_class, confidence = predict_drawing(model_path, image_path)
        print(f"預測結果: {predicted_class}")
        print(f"信心度: {confidence:.2f}%")
    except Exception as e:
        print(f"錯誤: {str(e)}")