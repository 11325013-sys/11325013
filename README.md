步驟一：建立與設定您的 Colab 筆記本

1.
開啟 Colab：前往 Google Colab 官方網站。

2.
建立新筆記本：點擊「檔案」(File) -> 「新增筆記本」(New notebook)。
<img width="1205" height="718" alt="螢幕擷取畫面 2025-10-26 195333" src="https://github.com/user-attachments/assets/4db1e0bc-097e-4405-bba1-98eca20fbe3f" />

3.
啟用 GPU：這是關鍵步驟。點擊「執行階段」(Runtime) -> 「變更執行階段類型」(Change runtime type)。在硬體加速器 (Hardware accelerator) 下拉選單中選擇 GPU，然後點擊「儲存」(Save)。
<img width="617" height="745" alt="螢幕擷取畫面 2025-10-26 195544" src="https://github.com/user-attachments/assets/46ecc608-4618-4c6c-afd9-516a3c951a19" />
<img width="856" height="724" alt="螢幕擷取畫面 2025-10-26 195555" src="https://github.com/user-attachments/assets/848088c2-6a4b-443b-97a9-c0ec02a74638" />

步驟二：掛載 Google Drive 以實現持久化儲存

為了避免因執行階段中斷而遺失工作進度，最佳實踐是將您的 Google Drive 掛載到 Colab 環境中。這樣，您可以直接從 Drive 讀取資料集，並將訓練好的模型權重和日誌儲存回 Drive。
<img width="409" height="747" alt="螢幕擷取畫面 2025-10-26 200308" src="https://github.com/user-attachments/assets/c46a5c5c-3cbe-4690-b25e-fb0444ccae91" />

在一個新的程式碼儲存格中輸入並執行以下程式碼：

Python


from google.colab import drive
drive.mount('/content/drive')


執行後，系統會提供一個授權連結。點擊連結，登入您的 Google 帳號，複製授權碼，然後貼回 Colab 的輸入框中並按下 Enter。成功後，您的 Google Drive 將會出現在左側檔案總管的 /content/drive/My Drive/ 目錄下。

步驟三：準備資料與安裝額外函式庫

您可以將資料集預先上傳到 Google Drive，或者使用 !wget 指令從網路直接下載。

Python


# 範例：從網路上直接下載資料集並解壓縮
!wget https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip
!unzip -q cats_and_dogs_filtered.zip


如果您的專案需要 Colab 未預裝的函式庫，可以使用 !pip install 指令進行安裝。

Python


!pip install tqdm


步驟四：編寫與執行模型訓練程式碼

現在，您可以像在本地 Jupyter 環境中一樣編寫您的模型定義、資料載入和訓練迴圈。以下是一個使用 TensorFlow/Keras 訓練一個簡單圖像分類模型的範例框架。

Python


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 設定資料路徑
base_dir = 'cats_and_dogs_filtered'
train_dir = f'{base_dir}/train'
validation_dir = f'{base_dir}/validation'

# 資料增強與產生器
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

# 建立模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 訓練模型
history = model.fit(
    train_generator,
    steps_per_epoch=100,  # 2000 images = batch_size * steps
    epochs=15,
    validation_data=validation_generator,
    validation_steps=50,  # 1000 images = batch_size * steps
    verbose=2)

# 將訓練好的模型儲存到 Google Drive
model.save('/content/drive/My Drive/my_cat_dog_classifier.h5')
print("模型已成功儲存至您的 Google Drive！")


步驟五：監控與最佳化

•
監控 GPU：您可以在任何時候執行 !nvidia-smi 命令來查看 GPU 的使用狀況，包括型號、記憶體使用率和溫度。

•
處理記憶體不足：如果遇到 OOM 錯誤，可以嘗試減小批次大小 (batch size)，或對圖片進行降採樣，或選擇一個更輕量級的模型架構。

•
儲存檢查點：對於長時間的訓練任務，強烈建議在每個 epoch 結束後都儲存一次模型權重（模型檢查點）。這樣即使中途斷線，也可以從上次的進度繼續訓練。

