import numpy as np
import pandas as pd
import os
import requests
from PIL import Image
import matplotlib.pyplot as plt

# 創建資料存放目錄
if not os.path.exists('quickdraw_data'):
    os.makedirs('quickdraw_data')

# 定義要下載的類別
categories = ['circle', 'envelope', 'triangle', 'star', 'square']
# categories = ['circle', 'envelope', 'triangle', 'star', 'square']


def download_data(category):
    base_url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category}.npy"
    file_path = f"quickdraw_data/{category}.npy"
    
    # 如果文件不存在才下載
    if not os.path.exists(file_path):
        print(f"下載 {category} 資料...")
        response = requests.get(base_url)
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"{category} 下載完成")
    else:
        print(f"{category} 資料已存在")

# 下載所有類別的資料
# for category in categories:
#     download_data(category)

def load_and_process_data(categories, samples_per_category=300):
    all_images = []
    all_labels = []
    
    for idx, category in enumerate(categories):
        # 載入數據
        print(f"處理 {category} 類別...")
        data = np.load(f'quickdraw_data/{category}.npy')
        
        # 隨機選擇300張圖片
        selected_indices = np.random.choice(len(data), samples_per_category, replace=False)
        selected_images = data[selected_indices]
        
        # 重塑圖片形狀從 (784,) 到 (28, 28)
        selected_images = selected_images.reshape(-1, 28, 28)
        
        # 添加到列表中
        all_images.append(selected_images)
        all_labels.extend([idx] * samples_per_category)
        
        # 顯示第一張圖片作為檢查
        # plt.imshow(selected_images[0], cmap='gray')
        # plt.title(f'Sample of {category}')
        # plt.show()
    
    return np.concatenate(all_images), np.array(all_labels)

def save_processed_data(X, y, categories, base_path='processed_data'):
    # 創建主資料夾
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # 為每個類別創建子資料夾
    for category in categories:
        category_path = os.path.join(base_path, category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)

    # 儲存每張圖片
    for i, (image, label) in enumerate(zip(X, y)):
        category = categories[label]
        file_path = os.path.join(base_path, category, f'{category}_{i}.png')
        
        # 將圖片轉換為 PIL Image 並儲存
        img = Image.fromarray(image.astype(np.uint8))
        img.save(file_path)

    print(f"已儲存 {len(X)} 張圖片到 {base_path} 資料夾")

# 儲存處理後的數據

# 執行函數
X, y = load_and_process_data(categories)
print(f"數據形狀: {X.shape}")
print(f"標籤形狀: {y.shape}")

save_processed_data(X, y, categories)