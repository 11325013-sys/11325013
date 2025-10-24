一、 Google Colab 免費算力限制與最佳實踐

雖然 Colab 提供了強大的免費算力，但為了確保資源公平分配，它設有幾項限制 [1, 2]：

限制項目
免費版限制說明
最佳實踐建議
連線時間
單次連線最長約 12 小時，閒置 90 分鐘會自動斷開。
訓練期間保持瀏覽器開啟，並定期執行代碼以避免閒置斷線。對於長時間訓練，建議使用 Model Checkpointing (模型檢查點) 功能，定期保存模型權重。
資源配額
每週總使用時數有限制 (約 30 小時)，且無法保證隨時能分配到高性能 GPU (如 T4 或 A100)。
在非高峰時段使用，並在訓練完成後立即手動斷開執行階段，釋放資源。
記憶體/磁碟
記憶體和磁碟空間有限 (通常約 12GB RAM，100GB 磁碟)。
優先將大型資料集存放在 Google Drive 或 Google Cloud Storage，並在需要時掛載到 Colab 環境。清理不必要的變數和檔案。


二、 啟動 Colab Notebook 與 GPU/TPU 設定

1. 建立新的 Notebook

1.
開啟您的瀏覽器，前往 Google Colab 網站。

2.
點擊「檔案」 -> 「新增筆記本」。

2. 啟用硬體加速器 (GPU/TPU)

這是使用免費算力的關鍵步驟。

1.
在 Notebook 介面中，點擊上方的「執行階段 (Runtime)」。

2.
選擇「變更執行階段類型 (Change runtime type)」。

3.
在彈出的視窗中，找到「硬體加速器 (Hardware accelerator)」下拉選單：

•
GPU: 適用於大多數深度學習模型 (如 CNN、RNN、Transformer)。免費版通常會分配到 NVIDIA T4 或其他等級的 GPU。

•
TPU: 適用於使用 TensorFlow 框架且需要極致並行運算的特定模型。



4.
選擇 GPU (或 TPU)，然後點擊「儲存」。

圖 1：變更執行階段類型 (GPU/TPU 選項)\n\n









3. 驗證硬體加速器

在 Notebook 的第一個代碼儲存格中，執行以下命令來驗證 GPU/TPU 是否成功啟用：

Python


# 檢查 GPU
!nvidia-smi

# 檢查 TPU (如果選擇了 TPU)
import os
if 'COLAB_TPU_ADDR' in os.environ:
    print('TPU已啟用')
else:
    print('TPU未啟用')


三、 處理資料與環境配置

1. 掛載 Google Drive (推薦)

由於 Colab 執行階段的檔案會在連線結束後消失，將資料和模型權重存放在 Google Drive 是最佳實踐。

1.
在 Notebook 中執行以下代碼：

2.
執行後會彈出一個授權連結，點擊連結，選擇您的 Google 帳號授權。

3.
授權成功後，您的 Google Drive 內容將會出現在 /content/drive/MyDrive/ 路徑下。

圖 2：掛載 Google Drive\n\n









2. 安裝所需函式庫

使用 pip 命令安裝模型訓練所需的函式庫，在 Colab 中，需要在命令前加上 ! 符號。

Python


# 例如：安裝 PyTorch, Hugging Face Transformers
!pip install torch torchvision
!pip install transformers datasets


四、 模型訓練流程範例 (以 PyTorch 圖像分類為例)

接下來，我們將提供一個基礎的 PyTorch 圖像分類模型訓練範例。您可以直接下載我們提供的 Colab Notebook 範例 (colab_training_example.ipynb)，上傳到 Colab 後即可運行。

Colab Notebook 範例檔案： colab_training_example.ipynb

核心步驟包括：

1.
載入資料集: 從 Google Drive 或內建資料集 (如 CIFAR-10) 載入。

2.
定義模型: 定義一個簡單的卷積神經網路 (CNN)。

3.
定義損失函數與優化器: 設定訓練的目標和方法。

4.
訓練迴圈: 迭代資料集，進行前向傳播、計算損失、反向傳播和權重更新。

5.
保存模型: 使用 torch.save() 將訓練好的模型權重保存到掛載的 Google Drive 中。

Python


# 完整 PyTorch 範例代碼將在下一階段提供
# ...
# 訓練完成後保存模型到 Google Drive
# PATH = '/content/drive/MyDrive/colab_models/my_model.pth'
# torch.save(model.state_dict(), PATH)


五、 訓練完成與資源釋放

1.
保存 Notebook: 確保您的 Notebook 已經保存 (Colab 會自動保存)。

2.
保存模型權重: 再次確認模型權重已成功保存到 Google Drive。

3.
斷開執行階段: 點擊「執行階段」 -> 「管理執行階段」 -> 找到您的 Notebook 點擊「終止」或「中斷連線並刪除執行階段」。

這一步非常重要！ 立即釋放資源可以避免不必要的配額消耗，讓您能更長時間地使用免費算力。




參考資料 [1] Google Colab: Colab 限額說明. https://colab.research.google.com/signup?hl=zh-cn [2] 阿里云开发者社区: Google Colab免费GPU大揭晓：超详细使用攻略. https://developer.aliyun.com/article/1207723
