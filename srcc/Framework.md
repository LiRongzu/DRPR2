# DRPR 框架：方法组合详解 (Framework Method Combinations)

本文档详细介绍了 `DRPR` 框架中支持的各种降dimensionality reduction (DR) 和 prediction 方法的组合。理解每种组合的工作流程、输入输出和关键配置有助于用户选择和比较最适合其研究目标的方法。

## 通用工作流程

无论选择哪种方法组合，项目通常遵循以下阶段化的工作流程：

1.  **数据预处理 (Preprocessing):**
    * 加载原始数据（盐度、风场、径流等）。
    * 处理缺失值 (NaN)。
    * **按维度**对高维数据（特别是用于降维的目标场，如盐度）进行标准化（计算并保存 scaler）。
    * 根据配置划分训练集、验证集和测试集（保存划分索引 `split_indices.npz`）。
    * 保存处理后的数据和 scaler (`data/processed/` 目录下)。

2.  **降维 (Dimensionality Reduction):** (根据 `model.dimensionality_reduction.method` 配置选择)
    * **输入:** 预处理后的高维训练数据 (通常是盐度场)。
    * **过程:** 训练所选的降维模型 (SOM, PCA, 或 Autoencoder)。
    * **输出:**
        * 训练好的降维模型 (如 SOM 对象, PCA 对象, AE 权重)。
        * 用于高维数据输入的 Scaler (主要用于 PCA, AE)。
        * 转换后的**低维表示** (SOM BMU 索引/坐标/距离向量, PCA 主成分, AE 潜在变量)，按 train/val/test 分割保存。

3.  **预测 (Prediction):** (根据 `model.prediction.method` 配置选择)
    * **输入:** 从降维阶段获得的低维表示 (train/val/test)。
    * **过程:** 训练所选的预测模型 (HMM 或 LSTM)，学习低维表示的时间演化规律。
    * **输出:**
        * 训练好的预测模型 (HMM 参数, LSTM 权重)。
        * 预测器内部使用的 Scaler (例如 LSTM 可能对其输入再次进行标准化)。
        * 预测出的**未来低维表示** (预测的 BMU 序列, 预测的 PCA 主成分序列, 预测的 AE 潜在变量序列)，按 train/val/test 分割保存。

4.  **重建 (Reconstruction):**
    * **输入:**
        * 从预测阶段获得的**预测出的低维表示**。
        * 在降维阶段训练好的**降维模型**。
        * 用于原始高维数据标准化的 **Scaler**。
    * **过程:** 使用降维模型的逆过程 (或特定重建逻辑)，将预测的低维表示映射回高维空间。然后使用高维 Scaler 进行反标准化。
    * **输出:** 重建后的高维数据 (按 train/val/test 分割保存)。

5.  **评估 (Evaluation):**
    * **输入:** 重建后的高维数据、原始高维数据 (需要通过 `split_indices.npz` 对齐)。
    * **过程:** 计算各种评估指标 (RMSE, MAE 等)，生成对比图表。
    * **输出:** 指标文件和可视化图表，保存在 Hydra 输出目录中，通常带有方法组合的标签。

## 方法组合详解

以下是框架支持的主要方法组合：

---

### 1. SOM - HMM

* **流程:** `高维数据 -> [SOM] -> BMU序列 -> [HMM] -> 预测的BMU序列 -> [SOM重建] -> 重建的高维数据`
* **降维 (SOM):**
    * 使用自组织映射 (SOM) 将高维盐度数据映射到一个 2D 网格上。
    * 关键输出是每个时间点的最佳匹配单元 (BMU) 的位置序列（通常转换为秩 `rank`）。
    * 需要为隐状态（盐度）和观测变量（风场/径流/组合）分别训练 SOM。
* **预测 (HMM):**
    * 使用隐马尔可夫模型 (HMM) 对 BMU 秩序列进行建模。
    * 当前实现 (`train_hmm.py`, `CategoricalHMM`) 假设输入是**离散的 BMU 秩**。
    * 将观测变量的 BMU 序列作为 HMM 的观测输入，预测盐度（隐状态）的 BMU 序列。
* **重建 (BMU-based):**
    * 将 HMM 预测的 BMU 秩（或索引）映射回状态 SOM 网格上的坐标。
    * 使用状态 SOM 模型中对应坐标的**权重向量**作为重建的高维场。
    * 需要 HMM 参数文件来辅助进行秩到索引的转换 (如果预测的是秩)。
    * 使用 `reconstruct_from_bmu` 函数。
    * 最后应用盐度高维数据的反标准化。
* **关键配置 (`conf/model/default.yaml`):**
    ```yaml
    dimensionality_reduction:
      method: som
    prediction:
      method: hmm
      hmm:
        input_type: bmu_rank # 确认 HMM 期望的是离散秩输入
        observation_features: ["wind", "flow"] # 或 ["wind"] 或 ["flow"]
        # ...其他 HMM 参数...
    ```

---

### 2. SOM - LSTM

* **流程:** `高维数据 -> [SOM] -> 低维表示(BMU坐标或DV) -> [LSTM] -> 预测的低维表示 -> [SOM重建/流形重建] -> 重建的高维数据`
* **降维 (SOM):** 同 SOM-HMM。输出可以是 BMU 坐标 `(T, 2)` 或距离向量 `(T, N_nodes)`。
* **预测 (LSTM):**
    * 使用长短期记忆网络 (LSTM) 学习低维表示的时间序列。
    * `train_lstm.py` 中的 `train_and_predict_lstm` 函数被设计为接受通用的低维数据输入。
    * LSTM 预测输入序列在下一时间步的低维表示。
    * LSTM 内部会对输入的低维序列再次进行标准化 (使用 `StandardScaler`)。
* **重建:**
    * **如果 LSTM 预测 BMU 坐标:** 使用状态 SOM 模型的 `inverse_transform` 方法（即将坐标映射回权重向量）。
    * **如果 LSTM 预测距离向量 (DV):** 需要使用基于 DV 的重建方法，例如训练一个流形模型（类似 `ManifoldReconstruction`）将 DV 映射回高维空间，或者使用 SOM 权重的加权平均（更复杂）。当前 pipeline 可能需要调整以支持此路径。
    * 最后应用盐度高维数据的反标准化。
* **关键配置 (`conf/model/default.yaml`):**
    ```yaml
    dimensionality_reduction:
      method: som
    prediction:
      method: lstm
      lstm:
        # ...LSTM 参数...
    ```
* **注意:** 需要明确 SOM 输出给 LSTM 的是哪种低维表示（BMU 坐标还是距离向量），并确保重建步骤与之对应。

---

### 3. PCA - HMM

* **流程:** `高维数据 -> [标准化] -> [PCA] -> 主成分序列 -> [离散化/GaussianHMM] -> 预测状态/主成分 -> [PCA逆变换] -> 重建的高维数据 -> [反标准化]`
* **降维 (PCA):**
    * 使用主成分分析 (PCA) 对标准化后的高维盐度数据进行线性降维。
    * 输出是连续的主成分系数序列 `(T, n_components)`。
    * 需要保存 PCA 模型 (`pca_model.pkl`) 和用于高维输入的 Scaler (`pca_salinity_input_scaler.pkl`)。
* **预测 (HMM):**
    * **面临挑战:** 标准 HMM (`CategoricalHMM`) 需要离散输入。
    * **选项 A (离散化):** 在 PCA 输出后添加 K-Means 聚类步骤，将连续的主成分向量映射为离散的状态索引，然后将这些索引输入 `CategoricalHMM`。(`input_type: discrete`)
    * **选项 B (连续 HMM):** 修改 `train_hmm.py` 以使用 `GaussianHMM`，直接在连续的主成分序列上训练。`GaussianHMM` 的预测结果通常是最可能的隐藏状态序列，其对应的均值向量（在主成分空间）可作为预测的低维表示。(`input_type: continuous`)
    * **需要实现:** 当前 pipeline 需要明确实现选项 A 或 B 才能运行此组合。
* **重建 (PCA-based):**
    * 获取 HMM 预测的低维表示（如果是选项 B，则为连续的主成分向量；如果是选项 A，则需要将预测的离散状态映射回对应的聚类中心或平均主成分向量）。
    * 使用训练好的 PCA 模型的 `inverse_transform` 方法将预测的低维向量转换回标准化的 高维空间。
    * 使用 PCA 阶段保存的高维输入 Scaler 进行反标准化。
* **关键配置 (`conf/model/default.yaml`):**
    ```yaml
    dimensionality_reduction:
      method: pca
      pca:
        n_components: 20 # PCA 组件数
    prediction:
      method: hmm
      hmm:
        # input_type: continuous # 如果使用 GaussianHMM
        # 或者
        # input_type: discrete # 如果使用 K-Means + CategoricalHMM
        # ... 其他 HMM 参数 ...
    ```

---

### 4. PCA - LSTM

* **流程:** `高维数据 -> [标准化] -> [PCA] -> 主成分序列 -> [LSTM] -> 预测的主成分序列 -> [PCA逆变换] -> 重建的高维数据 -> [反标准化]`
* **降维 (PCA):** 同 PCA-HMM，输出连续的主成分序列。保存 PCA 模型和高维 Scaler。
* **预测 (LSTM):**
    * `train_and_predict_lstm` 函数接收 PCA 主成分序列作为输入。
    * LSTM 学习预测下一时间步的主成分向量。
    * LSTM 内部会对主成分序列再次进行标准化。
* **重建 (PCA-based):**
    * 获取 LSTM 预测的主成分序列（需要反向 LSTM 的内部标准化）。
    * 使用训练好的 PCA 模型的 `inverse_transform` 方法将预测的主成分转换回标准化的 高维空间。
    * 使用 PCA 阶段保存的高维输入 Scaler 进行反标准化。
* **关键配置 (`conf/model/default.yaml`):**
    ```yaml
    dimensionality_reduction:
      method: pca
      pca:
        n_components: 20
    prediction:
      method: lstm
      lstm:
        # ...LSTM 参数...
    ```

---

### 5. Autoencoder - HMM

* **流程:** `高维数据 -> [标准化] -> [AE Encoder] -> 潜在变量序列 -> [离散化/GaussianHMM] -> 预测状态/潜在变量 -> [AE Decoder] -> 重建的高维数据 -> [反标准化]`
* **降维 (Autoencoder):**
    * 使用自动编码器 (AE) 对标准化后的高维盐度数据进行非线性降维。
    * 输出是连续的潜在空间变量序列 `(T, encoding_dim)`。
    * 需要保存 AE 模型 (`ae_model.pt`) 和用于高维输入的 Scaler (`ae_salinity_input_scaler.pkl`)。
* **预测 (HMM):**
    * **面临挑战:** 与 PCA-HMM 类似，需要处理连续的潜在变量。
    * **选项 A (离散化):** K-Means 聚类潜在变量 -> `CategoricalHMM`。(`input_type: discrete`)
    * **选项 B (连续 HMM):** 修改 `train_hmm.py` 使用 `GaussianHMM`。(`input_type: continuous`)
    * **需要实现:** 需要实现选项 A 或 B。
* **重建 (AE-based):**
    * 获取 HMM 预测的低维表示（连续潜在向量或离散状态映射回的向量）。
    * 使用训练好的自动编码器的 **Decoder** 部分将预测的潜在变量映射回标准化的 高维空间。
    * 使用 AE 阶段保存的高维输入 Scaler 进行反标准化。
* **关键配置 (`conf/model/default.yaml`):**
    ```yaml
    dimensionality_reduction:
      method: autoencoder
      autoencoder:
        encoding_dim: 15
        hidden_layers: [64, 32]
        # ...AE 参数...
    prediction:
      method: hmm
      hmm:
        # input_type: continuous # 或 discrete
        # ... 其他 HMM 参数 ...
    ```

---

### 6. Autoencoder - LSTM

* **流程:** `高维数据 -> [标准化] -> [AE Encoder] -> 潜在变量序列 -> [LSTM] -> 预测的潜在变量序列 -> [AE Decoder] -> 重建的高维数据 -> [反标准化]`
* **降维 (Autoencoder):** 同 AE-HMM，输出连续的潜在变量序列。保存 AE 模型和高维 Scaler。
* **预测 (LSTM):**
    * `train_and_predict_lstm` 函数接收 AE 潜在变量序列作为输入。
    * LSTM 学习预测下一时间步的潜在变量向量。
    * LSTM 内部会对潜在变量序列再次进行标准化。
* **重建 (AE-based):**
    * 获取 LSTM 预测的潜在变量序列（需要反向 LSTM 的内部标准化）。
    * 使用训练好的自动编码器的 **Decoder** 部分将预测的潜在变量映射回标准化的 高维空间。
    * 使用 AE 阶段保存的高维输入 Scaler 进行反标准化。
* **关键配置 (`conf/model/default.yaml`):**
    ```yaml
    dimensionality_reduction:
      method: autoencoder
      autoencoder:
        encoding_dim: 15
        # ...AE 参数...
    prediction:
      method: lstm
      lstm:
        # ...LSTM 参数...
    ```

---

## 如何选择和运行组合

1.  **编辑配置:** 修改 `conf/model/default.yaml` 中的 `dimensionality_reduction.method` 和 `prediction.method` 为您想测试的组合。同时，根据需要调整相应方法的具体参数（如 `n_components`, `encoding_dim`, `hidden_size` 等）。如果选择 HMM 与 PCA/AE 组合，请确保 `prediction.hmm.input_type` 设置正确，并已实现相应的处理逻辑。
2.  **运行 Pipeline:** 执行 `python src/run_pipeline.py`。脚本将根据您的配置自动执行选定的降维、预测和重建步骤。
3.  **批量运行 (比较):** 使用 `python src/run_pipeline.py --multirun ...` 命令可以方便地运行多个组合进行比较。

## 比较不同框架

框架的设计目标之一就是便于比较。通过运行不同的组合（建议使用 `--multirun`），您可以获得每个组合在相同数据分割上的评估结果（保存在各自的 Hydra 输出目录中）。比较这些结果（例如，比较不同组合的 RMSE、MAE 或查看重建结果的可视化图表）可以帮助您确定哪种降维-预测框架最适合您的特定数据和预测任务。