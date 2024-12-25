from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用 oneDNN 操作
from PIL import Image

def load_data(data_dir='processed_data'):
    X = []
    y = []
    categories = ['circle', 'envelope', 'triangle', 'star', 'square']
    
    for idx, category in enumerate(categories):
        category_path = os.path.join(data_dir, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = Image.open(img_path).convert('L')
            img_array = np.array(img) / 255.0
            X.append(img_array)
            y.append(idx)
    
    X = np.array(X).reshape(-1, 28, 28, 1)
    y = to_categorical(y, 5)
    return X, y

def keras_model(image_x, image_y):
    num_of_classes = 5  # 修改為5類
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(image_x,image_y,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(num_of_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "QuickDraw_v2.keras"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    return model, callbacks_list

if __name__ == "__main__":
    print("開始載入資料...")
    X, y = load_data()
    print(f"載入完成. 資料形狀: {X.shape}, 標籤形狀: {y.shape}")
    
    print("分割訓練集和測試集...")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("建立模型...")
    model, callbacks_list = keras_model(28, 28)
    
    print("開始訓練...")
    history = model.fit(X_train, y_train,
                       batch_size=32,
                       epochs=50,
                       validation_data=(X_test, y_test),
                       callbacks=callbacks_list)
    print("訓練完成！")