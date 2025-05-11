# DRPR: 用于海洋数据的降维与预测框架 (Dimensionality Reduction and Prediction Framework for Oceanographic Data)

本 README 文件档旨在介绍 `DRPR` 项目，一个用于处理和预测海洋时序数据（特别是盐度场）的模块化框架。该框架集成了多种降维技术和预测模型，并允许用户通过配置文件灵活地组合和比较它们。

## 概述

海洋数据的时空预测对于理解和管理海洋环境至关重要。然而，这些数据通常具有高维度和复杂的动态特性。本项目旨在提供一个统一的平台，用于：

1.  **降维:** 使用不同的方法（SOM, PCA, Autoencoder）提取高维海洋数据（如盐度场）的关键低维特征。
2.  **预测:** 基于提取的低维特征，利用时序预测模型（HMM, LSTM）预测未来的状态。
3.  **重建:** 将预测的低维状态重建回原始的高维空间。
4.  **评估与比较:** 提供评估指标和可视化工具，方便比较不同降维-预测组合的效果。

该框架利用 [Hydra](https://hydra.cc/) 进行配置管理，允许用户通过 YAML 文件或命令行参数轻松修改模型、参数和运行流程。

## 主要特点

* **模块化设计:** 将数据处理、降维、预测、重建和评估分解为独立的模块/阶段。
* **多种方法集成:**
    * **降维:**
        * 自组织映射 (SOM - `som_pytorch.py`)
        * 主成分分析 (PCA - `pca.py`)
        * 自动编码器 (Autoencoder - `autoencoder_pytorch.py`)
        * *(代码库中还可能包含 UMAP, POD, DMD 的实现)*
    * **预测:**
        * 隐马尔可夫模型 (HMM - `hmm.py`, 使用 `hmmlearn`)
        * 长短期记忆网络 (LSTM - `lstm_pytorch.py`, `lstm_pca.py`, 使用 `PyTorch`)
* **灵活配置:** 使用 Hydra 管理配置，易于切换方法、调整参数和定义运行流程 (`conf/` 目录)。
* **端到端流程:** 支持从原始数据预处理到最终评估的完整工作流 (例如通过 `src/pipeline_ae_lstm.py`, `src/pipline_pca.py` 或 `run_experiment.sh` 脚本)。
* **可比性:** 通过 Hydra 的 `multirun` 功能，可以方便地运行多种方法组合并比较结果。
* **数据处理:** 包含针对特定海洋数据（盐度、风场、径流）的预处理和序列创建工具。
* **标准化:** 支持按维度标准化，这对于基于距离的降维方法（如 SOM）至关重要。

## 项目结构

```
DRPR/
├── conf/                 # Hydra 配置文件目录
│   ├── config.yaml       # 主配置文件 (包含默认值)
│   ├── data/             # 数据相关配置
│   ├── evaluation/       # 评估相关配置
│   ├── model/            # 模型选择和参数配置
│   ├── paths/            # 路径配置
│   ├── training/         # 训练过程配置
│   └── visualization/    # 可视化配置
├── data/                 # 数据目录 (需要用户创建或配置)
│   ├── raw/              # 原始数据存放处
│   │   └── mini/         # (可选) 小型测试数据集
│   └── processed/        # 预处理后的数据存放处
│       └── mini/         # (可选) 小型测试数据集的处理结果
├── outputs/              # Hydra 运行的默认输出目录 (自动创建)
│   ├── YYYY-MM-DD/       # 按日期分子目录
│   │   └── HH-MM-SS/     # 按时间分子目录 (包含单次运行结果)
│   └── multirun/         # 多次运行 (比较) 的输出目录
├── src/                  # 源代码目录
│   ├── data_processing/  # 数据加载、验证、预处理、序列创建
│   ├── dimensionality_reduction/ # 降维算法实现 (SOM, PCA, AE 等)
│   ├── evaluation/       # 评估指标计算、可视化、比较
│   ├── prediction_models/ # 预测模型实现 (HMM, LSTM)
│   ├── reconstruction/   # 从低维重建高维数据的方法
│   ├── training/         # 各模型训练脚本/函数 (train_som, train_hmm, ...)
│   ├── utils/            # 通用工具 (配置、日志、模型保存/加载、BMU 工具等)
│   ├── pipeline_ae_lstm.py # AE+LSTM 流程示例
│   ├── pipline_pca.py    # PCA+LSTM 流程示例 (注意可能的拼写错误)
│   └── ...               # 其他 pipeline 或辅助脚本
├── requirements.txt      # Python 依赖列表
├── run_experiment.sh     # (可选) 用于运行实验的 Shell 脚本
└── README.md             # 本文件
```
*(注意: 上述结构可能不完全，具体请参考实际项目)*

## 安装与设置

1.  **克隆仓库:**
    ```bash
    git clone <your-repository-url>
    cd DRPR
    ```
2.  **创建环境 (推荐):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```
3.  **安装依赖:**
    项目根目录下已提供 `requirements.txt` 文件。
    ```bash
    pip install -r requirements.txt
    # 如果需要 PyTorch 的 GPU 版本，请参考 PyTorch 官方网站安装说明
    # 如果需要 basemap 或 cartopy，推荐使用 conda 安装
    # conda install -c conda-forge cartopy
    ```

## 数据准备

1.  **原始数据:** 将原始数据文件（如 `salt_grid.npy`, `wind.npy`, `flow.npy`, `mask.npy` 及对应的坐标文件）放置在 `data/raw/mini/` 目录（或您在 `conf/paths/default.yaml` 中配置的其他原始数据目录）。
2.  **创建小型数据集 (可选):** 如果原始数据集很大，可以运行 `src/data_processing/create_mini_dataset.py` 脚本生成一个较小的数据集用于快速测试，默认输出到 `data/raw/mini/`。
3.  **预处理:** 数据预处理通常作为 pipeline 的一部分自动执行，由配置文件 (`conf/data/default.yaml`, `conf/training/default.yaml`) 控制，包括数据划分和标准化。确保相关配置正确。

## 配置

项目的行为主要由 `conf/` 目录下的 YAML 文件控制。

* **`conf/config.yaml`:** 主配置文件，通过 `defaults` 列表组合其他配置文件。定义了 pipeline 中要执行的步骤 (`run_steps`)。
* **`conf/model/default.yaml`:** **核心配置区域**
    * `dimensionality_reduction.method`: 选择降维方法 ('som', 'pca', 'autoencoder')。
    * `prediction.method`: 选择预测方法 ('hmm', 'lstm')。
    * 包含各种方法（SOM, PCA, AE, HMM, LSTM）的特定参数。
* **`conf/training/default.yaml`:** 配置训练过程参数，如学习率、epochs、batch size、随机种子、数据分割比例、**SOM 地图尺寸** (`map_size_state`, `map_size_observation`)。
* **`conf/paths/default.yaml`:** 配置各种数据和模型结果的路径。路径通常相对于项目根目录或 Hydra 的运行目录。
* **其他配置文件:** 分别控制数据加载、评估、可视化等方面的细节。

您可以通过修改这些 YAML 文件或在命令行中覆盖参数来调整实验设置。

## 使用方法

可以通过运行特定的 pipeline 脚本或使用 `run_experiment.sh` 脚本来执行实验。

1.  **运行特定 Pipeline (例如 PCA + LSTM):**
    ```bash
    # 确保配置文件 conf/config.yaml 或命令行参数指定了正确的模型
    # 例如，确保 model.dimensionality_reduction.method=pca 和 model.prediction.method=lstm
    python src/pipline_pca.py # 注意脚本名称的拼写可能为 pipeline_pca.py

    # 运行 AE + LSTM
    python src/pipeline_ae_lstm.py
    ```
    结果将保存在 `outputs/YYYY-MM-DD/HH-MM-SS/` 目录下。

2.  **通过命令行覆盖参数:**
    ```bash
    # 示例：运行 PCA + LSTM 并修改 PCA 组件数
    python src/pipline_pca.py model.dimensionality_reduction.pca.n_components=30

    # 示例：运行 AE + LSTM 并修改 LSTM 隐藏层大小
    python src/pipeline_ae_lstm.py model.prediction.lstm.hidden_size=128
    ```

3.  **使用 `run_experiment.sh` (如果适用):**
    检查 `run_experiment.sh` 脚本的内容，了解其用法。它可能封装了运行特定实验或进行超参数搜索的命令。
    ```bash
    # 示例 (具体命令取决于脚本内容)
    bash run_experiment.sh --dr pca --pred lstm --components 30
    # 或者
    bash run_experiment.sh pca_lstm_config.yaml
    ```

4.  **运行超参数搜索 (如果配置了):**
    项目可能包含使用 Optuna/Hydra 进行超参数搜索的脚本（如 `pipeline_hyper_pca_lstm.py`）。
    ```bash
    # 示例：运行 PCA+LSTM 的超参数搜索
    python src/pipeline_hyper_pca_lstm.py --multirun # Hydra multirun 通常用于搜索
    ```

## 评估与比较

* 每次运行后，相关的输出（模型、低维数据、预测、重建数据、评估指标、图表）会保存在对应的 Hydra 输出目录中（单次运行在 `outputs/YYYY-MM-DD/HH-MM-SS/`，多次运行在 `outputs/multirun/...`）。
* 评估结果（指标 `.npy` 文件和图表 `.png` 文件）通常位于该运行目录下的 `evaluation/<dr_method>_<pred_method>/<split>/` 子目录中。
* `pipeline_summary.pkl` 文件（可能位于 `evaluation/<dr_method>_<pred_method>/` 或运行根目录）包含了该次运行各阶段的详细结果信息字典。
* 通过比较不同 `multirun` 输出目录下的评估结果，可以分析不同降维-预测框架的性能差异。

## 注意事项

* **HMM 与连续输入:** 如果使用 HMM 预测 PCA 或 Autoencoder 的连续输出，请确保 HMM 模型 (`prediction_models/hmm.py`) 支持连续观测 (如 `GaussianHMM`) 或在流程中加入了离散化步骤。检查相关配置 (`conf/model/default.yaml`)。
* **数据对齐:** 评估阶段比较重建数据和原始数据时，确保它们在时间步长和特征维度上正确对齐至关重要。检查 `evaluation` 模块中的逻辑。
* **内存:** 处理高维空间数据（尤其是自动编码器训练和 PCA）可能需要较多内存。
* **脚本名称:** 请注意检查 `src/` 目录下 pipeline 脚本的确切名称（例如 `pipline_pca.py` vs `pipeline_pca.py`）。