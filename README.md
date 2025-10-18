# 汽車價格預測專案 — 詳細 CRISP-DM 報告

---

## 摘要（Executive Summary）

- 目的：建立可預測二手汽車價格的模型，支援二手車定價、買賣估價與平台推薦。
- 方法：遵循 CRISP-DM，從資料取得 → 清洗 → 特徵工程 → 建模 → 評估 → 部署。候選模型包含線性模型與數個非線性模型（隨機森林、Gradient Boosting）等。
- 關鍵發現（來自實驗）：car_price.ipynb 的 EDA 顯示 price 分布右偏且存在高價離群值，類別高基數（如 CarName）會導致 one-hot 後維度增長；部分數值如 enginesize、horsepower 與 curbweight 對價格有顯著影響。
- 建議：建立資料品質監控、採用模型監測與定期重訓機制、加入地區與維修記錄等額外欄位以提升準確度與公平性。

---

## 1. Business Understanding（商業理解）

### 1.1 具體目標（What success looks like）

- 主目標：在測試集上達到 MAE ≦ 平均價格的 10%（或 R² ≧ 0.6）為可接受門檻（依商業需求可調）。
- 次目標：模型需具備一定解釋性（能提供特徵重要性或局部解釋），方便業務了解價格驅動因素。
- 風險約束：避免系統性偏差（例如對某些品牌或區域產生不公平估價）。

### 1.2 利害關係人與使用情境

- 二手車商：快速估價、批次估值。
- 平台買家：提供參考價、議價依據。
- 內部產品：API 形式嵌入到上架流程，或作為報告/推薦引擎的輸入。

### 1.3 商業需求拆解（Requirements）

- 延遲需求：即時呼叫（API latency ≦ 200 ms）或批次運算（每日/每週）。
- 可解釋性：至少提供全局特徵重要性和個別預測的局部解釋（如 SHAP）。
- 可維運性：模型需要可重訓與版本控管、監控資料漂移與概念漂移。

### 為什麼要這樣做（Reasoning）

把商業目標具體化能轉化成可衡量的技術指標（MAE、R²、API latency），減少模型開發與實際應用間的落差。

---

## 2. Data Understanding（資料理解）

### 2.1 資料來源與欄位說明

- 使用資料集（notebook 下載）：Kaggle — hellbuoy/car-price-prediction  
- 下載檔案範例：CarPrice_Assignment.csv（搭配 Data Dictionary）  
- 欄位（例）：`price`（目標）、`CarName`、`fueltype`、`aspiration`、`doornumber`、`carbody`、`drivewheel`、`enginelocation`、`enginesize`、`horsepower`、`citympg`、`highwaympg` 等（詳見資料字典）

### 2.2 來自 car_price.ipynb 的 EDA 摘要（notebook-derived findings）

- 讀取資料結果：  
  - 檔案：CarPrice_Assignment.csv  
  - 資料形狀：205 rows × 26 columns  
  - 偵測到的 target 欄位：price
- 缺失值：  
  - 原始檢查後：在 notebook 片段中未見大量缺失（isna().sum() 顯示每欄皆為 0）  
  - 處理策略（notebook 採用）：將特殊標記（例如 "?"）替換為 NaN，對數值欄位以中位數填補，類別欄位以眾數填補；處理後資料無缺失
- 目標 price 的描述性統計（notebook 輸出）：  
  - count: 205  
  - mean: 13276.71  
  - std: 7988.85  
  - min: 5118.00  
  - 25%: 7788.00  
  - 50% (median): 10295.00  
  - 75%: 16503.00  
  - max: 45400.00  
  - 解讀：price 分布右偏（存在數個高價離群值），可考慮對 price 做 log 轉換或在評估時報告中位數/MAE 與 percentile-based 指標。
- 部分類別欄位分布（notebook 顯示前幾項）：  
  - fueltype：gas = 185, diesel = 20  
  - aspiration：std = 168, turbo = 37  
  - doornumber：four = 115, two = 90  
  - carbody：sedan = 96, hatchback = 70, wagon = 25, hardtop = 8, convertible = 6  
  - drivewheel：fwd = 120, rwd = 76, 4wd = 9  
  - enginelocation：front = 202, rear = 3  
  - fuelsystem top: mpfi = 94, 2bbl = 66, idi = 20, ...  
  - cylindernumber top: four = 159, six = 24, five = 11, ...
- 特徵工程 / 編碼（notebook 操作）：  
  - 類別欄位使用 One-Hot Encoding (drop_first=True) → 編碼後特徵數: 190
- 資料切分（notebook 操作）：  
  - Train / Test 切分（80% / 20%）  
  - 結果 shapes：X_train (164, 190), X_test (41, 190), y_train (164,), y_test (41,)
- 其他數值欄位摘要（節錄）：  
  - enginesize mean ≈ 126.91（min 61, max 326）  
  - horsepower mean ≈ 104.12（min 48, max 288）  
  - curbweight mean ≈ 2555.57（min 1488, max 4066）

（以上數值皆直接來自 car_price.ipynb 執行輸出）

### 2.3 初步結論（來自 notebook）
- 資料欄位完整、缺失少（或已處理），可專注於偏態、極端值、與類別高基數問題（例如 CarName）。  
- 由於 price 明顯右偏，建議在訓練線性模型時試試 log(price)，或針對 tree-based 模型直接使用原始 price 並以 MAE、RMSE 與分群（price bucket）檢查偏差。  
- 類別型特徵透過 one-hot 編碼後會產生高維度（190 features），在模型選擇上可考慮使用 target encoding 或頻次編碼以減少維度爆炸（特別是在更大資料集上）。

---

## 3. Data Preparation（資料準備）

> 在這個階段，我會把「怎做」和「為什麼」逐一列出，供資料工程師或分析師直接跟著執行。
> 

### 3.1 缺失值處理策略（詳細判斷流程）

1. 判斷缺失型態
    - MCAR（完全隨機）、MAR（條件隨機）、MNAR（非隨機）。
    - 如果是 MNAR（例如車況只在好車時被填寫），要小心直接填補會引入偏差。
2. 數值欄位
    - 少量缺失（<5%）：以中位數填補（對極端值較不敏感）。
    - 中量缺失（5–30%）：視分布採中位數或分群後分別填補（例如依品牌填補）。
    - 大量缺失（>30%）：考慮移除該欄或把缺失視為一種訊息（加上缺失 flag）。
3. 類別欄位
    - 少量缺失：以眾數填補或填為 "Unknown"。
    - 大量缺失：若欄位具業務重要性（如 service_history），考慮保留並用特殊類別標註缺失。
4. 特殊標記（例如 "NA", "-", "?"）
    - 先統一轉為空值再處理。

**為什麼**：不同缺失機制會帶來不同偏差風險；中位數/眾數填補是簡單穩健的 baseline，但對於系統性缺失需更謹慎。

### 3.2 異常值與極端值處理

1. 偵測方法：
    - IQR 方法（低於 Q1 − 1.5·IQR 或高於 Q3 + 1.5·IQR）
    - z-score（標準差方法）或基於 domain rule（車齡 > 100 年顯然錯誤）
2. 處理策略：
    - 修正（若為資料輸入錯誤，可改回合理值）。
    - 刪除（極端且懷疑為錯誤的 row）。
    - 裁剪（capping / winsorization）：把極端值限制在特定百分位（例如 1%–99%）。
    - 用 robust model（如 tree-based）或採 log 變換減少極端值影響。
3. 特別注意：
    - 極端真實案例（如高檔車）不應一律刪除，需由業務判斷是否保留。

**為什麼**：極端值會嚴重影響線性模型與評估指標（MSE），但盲目刪除會丟失真實且重要的商業資訊。

### 3.3 類別變數處理（encoding）

1. One-Hot Encoding（適用於低基數類別）
    - 優點：不引入順序假設、易用於線性模型。
    - 缺點：基數高會導致維度爆炸（notebook 中編碼後 features = 190）。
2. Target Encoding / Mean Encoding（適用於高基數）
    - 為每個類別放上訓練集的平均 price（需做平滑與避免洩漏）。
    - 使用 K-fold target encoding 或 leave-one-out 以避免資料洩漏。
3. 頻次編碼（frequency encoding）：以該類別出現頻率作為特徵。
4. 合併罕見類別：把出現頻率低於 threshold（例如 1%）的類別合併成 `Other`。

**為什麼**：不同 encoding 方法在 bias-variance 與維度效率上權衡，選擇要依模型與欄位基數決定。

### 3.4 特徵衍生（Feature Engineering）

列出常見且實用的衍生特徵與設計原因：

- `age = current_year - year`：車齡直接影響價格，比原始 year 更直觀。
- `mileage_per_year = mileage / age`：衡量使用強度，比單純 mileage 更有意義。
- `brand_group`：把品牌分級（luxury / mainstream / economy）。
- `service_flag`：是否有完整維修紀錄（binary）。
- `is_limited_edition`、`has_navigation`（若有選配欄位）。
- `region_price_index`：不同城市價格基準差異（地區一級特徵）。
- `days_on_market`：上架天數，可反映市場接受度。
- `seasonal`：上架月份或季度（用於考慮季節性價格變動）。

**為什麼**：衍生特徵通常能把原始資料中的隱含關係顯式化，提升模型可學習的訊號。

### 3.5 數值特徵標準化與轉換

- 標準化（StandardScaler）或最小最大縮放（MinMaxScaler）
    - 線性模型與距離敏感算法（KNN、SVM）需要標準化。
    - Tree-based models（RF, GBM）通常不需要，但常做以便 pipeline 一致。
- 變數轉換（如對 price 做 log）
    - 當 price 分布右偏，對 price 做 `log(price)` 可讓誤差更接近正態、穩定模型訓練。
    - 轉換後評估 MAE 時需回轉（exponentiate）並撰寫如何計算回轉後的 MAE。

**為什麼**：確保模型訓練穩定且不同特徵尺度不會不合理影響模型權重。

### 3.6 資料切分（Train/Validation/Test）與時間序列注意事項

- 若資料無時間依賴：隨機切分 80/20（或 70/15/15 train/val/test）。
- 若存在時間依賴（例如要預測未來價格）：採用時間序列切分（例如以時間為界的前 N% 作訓練，後面的做測試）以防止資訊洩漏。
- 保留一個「冷啟動」的外部測試集（即未在模型選擇/調參中使用）做最終評估。

**為什麼**：切分策略直接影響模型泛化評估的可信度，時間依賴需特別小心。

---

## 4. Modeling（建模）

> 這一節分成「模型候選」、「訓練策略」、「調參流程」與「模型比較標準」。
> 

### 4.1 候選模型與使用理由

1. **Baseline：簡單線性回歸 / OLS**
    - 目的：建立基線表現與解釋性。
    - 優點：容易解釋，係數代表邏輯關係。
2. **正則化線性模型：Ridge、Lasso**
    - 目的：減少共線性與過擬合。
    - Lasso 可用於特徵選擇（L1）。
3. **樹狀模型：Random Forest**
    - 優點：自動捕捉非線性，對缺失與標度較不敏感。
4. **梯度提升（GBDT）：XGBoost / LightGBM / CatBoost**
    - 常為結構化資料的高效模型，表現通常優於 RF。
5. **可解釋模型增強：Explainable Boosting Machine（EBM）**
    - 如果解釋性是重點，EBM 能在可解釋與準確度之間取得平衡。
6. **Ensemble**
    - 混合多模型（stacking、blending）常能提升穩定性與表現。

**為什麼**：從簡單到複雜以分步驗證方式評估每種模型的效益與成本，並在商業可接受範圍內選擇最合適方案。

---
## 5. Evaluation（評估）

### 5.1 評估指標詳細說明（及選用理由）

- **MAE（Mean Absolute Error）**：對業務最直觀（平均誤差），對極端值不太敏感。
- **MSE / RMSE**：對極端誤差懲罰較重，若希望避免大錯誤應參考。
- **R²**：解釋變異比例（但對偏態分布解釋性有限）。
- **MAPE（若 price 非零）**：百分比誤差，對低價樣本會非常敏感，需小心使用。
- **Calibration（校準）**：檢查預測與實際是否系統性偏差（例如對某品牌低估）。

### 5.2 交叉驗證與不確定性量化

- 使用 k-fold（建議 k=5 或 10）產生多個分數，報告平均與標準差以表現穩定性。
- 對最終模型做 **bootstrap** 估計預測不確定性（產生信賴區間，例如 95% CI）。

### 5.3 錯誤分析（Error Analysis）

- 按 `price` 高低、品牌、地區、車齡分群檢視誤差分佈（是否對某群體系統性偏差）。
- 可視覺化：殘差圖（residual vs predicted）、殘差分佈直方圖、誤差箱型圖按類別分組。
- 針對高誤差案例做人工檢視（取 top-N 最大誤差的樣本），判斷資料或商業情境是否特殊（例如改裝車、事故車）。

**為什麼**：不只是整體數字好看，業務關心的是模型在不同客群上的偏差與風險。

### 5.4 模型穩定性測試

- 異常值測試：在測試集中加入極端價格（或模擬異常情形）觀察模型輸出變化。
- 時間穩定性：把訓練集往前挪 1 年，測試在新時期的表現，評估概念漂移（concept drift）。
- 對抗測試（若可行）：測試輸入微小變動（例如有誤差的里程）模型輸出是否大幅擾動。

**為什麼**：確保模型在真實世界的輸入不完美情況下仍有合理表現。

---

## 6. Deployment（部署與維運）

### 6.1 部署架構建議（高層）

1. **API 服務**
    - 模型包裝成 REST/GRPC API（支援批次與即時查詢）。
    - 使用容器化（Docker）與 K8s 管理擴展性。
2. **Batch Pipeline**
    - 定期（每日/每週）運行批量評估、重新估價舊庫存。
3. **模型託管**
    - 模型與處理 pipeline 存放版本控制（例如 MLflow 或 S3+Git）。
4. **資料管線**
    - 建立 ETL 流程，把輸入清洗/處理邏輯一致化（同 pipeline code）。

### 6.2 監控與警示

- **資料品質監控**：輸入分布監控（平均、方差、缺失率、類別頻率）。
- **模型效能監控**：線上 MAE 偵測（若有實際成交回饋），或用代理指標（例如預測分佈與歷史分佈差異）。
- **漂移偵測**：分布漂移（Population Drift）與標籤漂移（Label Drift）警示機制。
- **運行監控**：API 延遲、錯誤率、資源使用（CPU/GPU/Memory）。

### 6.3 重訓政策（Retraining）

- 建議政策範例：
    - 定期重訓：每月或每季，視資料量而定。
    - 事件觸發重訓：若監控顯示 MAE 上升超過閾值（例如 10%）或資料分布顯著漂移，立即觸發。
- 使用自動化 CI/CD（如 GitHub Actions + MLflow）完成模型驗證、打包與部署。

### 6.4 安全與隱私考量

- 個人資料（如車主個資）需脫敏或遵守 GDPR/在地法規。
- 記錄模型版本與預測輸出以利稽核與爭議處理。

**為什麼**：良好的監控與重訓機制能確保模型長期可靠並降低商業風險。

---

## 7. Business Impact 與後續建議

### 7.1 預期效益

- 自動估價可節省人工工時、提供即時決策支援並提升交易成功率。
- 建議先以「估價建議（confidence band + point estimate）」形式上線，降低單一預測錯誤風險。

### 7.2 風險與緩解

- 風險：資料偏差導致系統性低估或高估特定族群車輛。
- 緩解：在介面呈現信賴區間、對系統性偏差做分群校正。

### 7.3 長期改進方向

- 收集更多標籤：實際成交價格、修復歷史、事故紀錄。
- 引入影像資料（車輛照片）做 multimodal 模型（結合圖像與 tabular）。
- 開發可解釋的客製化功能（例如業務介面可調整特徵權重以反映店家策略）。

---

## 8. 報告與交付物清單（Deliverables）

- EDA 報告（包含關鍵圖表與結論）
- 資料處理規範文件（包含每個欄位如何填補、轉換、衍生的具體公式）
- 模型比較表（各模型 CV 指標、延遲與要求資源）
- 最佳模型 artifact（序列化模型 + 版本）
- API 規格與部署說明
- 監控面板建議（要監控哪些指標、閾值）
- 使用者說明文件（如何在平台上使用估價結果與解讀信賴區間）

---

## 9. 建議的圖表與表格（報告中一定要有）

- 目標變數（price）直方圖 + log(price) 直方圖（展現偏態）
- 缺失值熱圖（heatmap 或條形圖）
- 特徵重要性條形圖（最佳模型）
- 殘差圖（predicted vs actual + ideal line）
- Cross-validation 指標表（每個模型的 mean ± std）
- Top-N 最大誤差案例表（包含原始欄位說明）
- 部署架構示意圖（API / Batch / Monitoring）

---

## 10. 具體執行步驟（逐步 checklist，給資料工程師/分析師）

1. 收集資料並取得資料字典。
2. 做完整 EDA 並產出 EDA 報告，含視覺化與資料品質結論。
3. 制定資料清洗與缺失處理策略並撰寫 data preprocessing spec。
4. 實作 pipeline（填補 → encoding → scaling → feature derivation），寫單元測試。
5. 切分資料（train/val/test），確定是否採時間切分。
6. 訓練 baseline（OLS）並存 baseline 結果。
7. 訓練候選模型（RF、GBDT、Ridge/Lasso），做 hyperparameter tuning（使用 CV）。
8. 比較模型並做錯誤分析（分群、殘差檢視）。
9. 選擇模型並做不確定性估計（bootstrap / prediction interval）。
10. 建立部署 pipeline（模型序列化、API、容器化），與監控方案。
11. 上線後 1 個月內密集監控（每週檢視 MAE、資料分布），並於每季進行回顧與重訓。

---

## 11. 常見問題（FAQ）

- Q：要不要對 price 做 log 轉換？
    
    A：如果 price 分布高度右偏，做 log 轉換通常能提升線性模型表現並讓誤差分布更接近常態。但要注意回轉（exp）後的 MAE 計算方式與解讀。
    
- Q：one-hot 會造成維度過大，怎麼辦？
    
    A：對高基數欄位用 target encoding 或 frequency encoding，或合併低頻類別。
    
- Q：如何避免 target encoding 洩漏？
    
    A：在 training 時做 K-fold target encoding 或 leave-one-out，且在 pipeline 中把 encoding 放在 cross-validation 的 fold 操作內。    

---

## 結語（Summary）

本報告把汽車價格預測專案依 CRISP-DM 全面拆解，從商業目標、資料檢視、詳細處理策略、模型候選與調參流程、到部署與監控都給出具體步驟與設計理由。接下來建議的行動是：

1. 立刻執行 EDA 與資料品質檢查（產出 EDA 報告）
2. 根據 EDA 決定是否對 `price` 做 log 轉換與是否採用時間切分
3. 快速實作 baseline（OLS）並開始候選模型訓練與 CV 比較
