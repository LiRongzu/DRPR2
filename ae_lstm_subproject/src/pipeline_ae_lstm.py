# src/pipeline_ae_lstm.py

import os
import logging
# 使用 joblib 保存/加载 .pkl 文件 (运行时 Scaler, PCA 模型)
import joblib
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
# 保持 StandardScaler 用于运行时拟合
from sklearn.preprocessing import StandardScaler
# 如果需要加载预训练 PCA 模型
# from sklearn.decomposition import PCA

# 导入项目模块 (请根据你的实际路径和名称调整)
# from utils import logger # 使用 Hydra 配置的日志
from utils import data_loader # 假设有 load_processed_data
from utils import model_utils
from data_processing import sequence_utils
from dimensionality_reduction.autoencoder_pytorch import AutoencoderDimensionalityReduction
from prediction_models.lstm_ae import LSTMPredictionModel # 保持使用，因为它灵活
from evaluation import metrics as eval_metrics
from utils.data_loader import load_mask # 假设 load_mask 在这里
# 假设可视化函数都在 eval_viz 中
from evaluation import visualization as eval_viz
# 假设指标计算函数在 eval_metrics 中
from evaluation import metrics as eval_metrics
# 添加在导入部分
# from data.sequence_dataset import TimeSeriesDataset  # 多步预测数据集
# from models.multi_step_lstm import MultiStepLSTM  # 多步预测模型
# from training.multi_step_trainer import train_multi_step_lstm  # 多步预测训练函数
# from evaluation.multi_step_evaluation import evaluate_multi_step_predictions  # 多步预测评估
from torch.utils.data import DataLoader
from src.evaluation.visualization import (
    plot_comparison_and_error,
    plot_spatial_rmse,
    plot_spatial_comparison_at_timestep, # 新增：绘制特定时间点对比图
    plot_spatial_difference,            # 新增：绘制特定时间点差异图
    plot_spatial_statistic,             # 新增：绘制通用空间统计图
    calculate_correlation_map,          # 新增：计算相关性图的函数
    plot_comparison_and_error_adjusted_style,
    generate_evaluation_report,         # 新增：生成评估报告的函数
    plot_spatial_distribution,            # 如果需要 heatmap 也导入
    plot_spatial_rmse_optimized
)
import matplotlib.pyplot as plt
import platform
from typing import Dict, Any, List, Optional
# --- 配置日志记录器 ---
log = logging.getLogger(__name__) # Hydra 会配置好
# --- 设备配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"正在使用设备: {device}")

def aelstm_multi_step_forecast_rmse_curve_minimal(
    cfg: DictConfig,
    ae_results: dict, # 需要里面的 'model_path' 和 'salinity_scaler_params'
    pred_results: dict, # 需要里面的模型和 scaler 路径
    all_processed_data: dict, # 需要里面的原始 salinity 数据 ('test' split)
    mask: np.ndarray, # 需要 Mask
    device: torch.device # 需要计算设备
) -> np.ndarray:
    """
    (最小改动版)
    计算 AE-LSTM 的多步迭代预测 RMSE 曲线。
    此函数负责加载所有必需的模型、Scaler 和预存的标准化数据。
    使用标准的 lstm_model.predict() 方法。
    """
    log.info("--- 开始最小改动版 AE-LSTM 迭代预测评估 ---")

    try:
        # --- 1. 加载模型 ---
        # LSTM Model
        lstm_model_path = pred_results.get('model_path')
        if not lstm_model_path or not os.path.exists(lstm_model_path): raise FileNotFoundError("LSTM 模型路径未在 pred_results 中找到或文件不存在")
        lstm_cfg = cfg.model.prediction.lstm_ae
        # 确定 LSTM 输入维度 (需要从 pred_results 或重新计算得到)
        # 假设 pred_results 包含了 input_dims_list (如果 run_lstm_prediction 返回了)
        # if 'input_dims_list' not in pred_results:
        #     raise ValueError("pred_results 中缺少 'input_dims_list'，无法确定 LSTM 输入维度") # 或者在这里重新计算
        lstm_input_dim = 15
        lstm_output_dim = ae_results['encoding_dim']
        lstm_model_eval = LSTMPredictionModel(
            input_size=lstm_input_dim, hidden_size=lstm_cfg.hidden_size,
            num_layers=lstm_cfg.num_layers, output_size=lstm_output_dim,
            dropout=lstm_cfg.get('dropout', 0.0), device=device # 传递 device
        )
        lstm_model_eval.load(lstm_model_path)
        lstm_model_eval.to(device) # 确保在设备上
        lstm_model_eval.eval()
        log.info(f"LSTM 模型已加载: {lstm_model_path}")

        # AE Model
        ae_model_path = ae_results.get('model_path')
        if not ae_model_path or not os.path.exists(ae_model_path): raise FileNotFoundError("AE 模型路径未在 ae_results 中找到或文件不存在")
        ae_model_eval = AutoencoderDimensionalityReduction.load(ae_model_path, device=device)
        ae_model_eval.model.to(device) # 确保在设备上
        ae_model_eval.model.eval() # 确保在评估模式
        log.info(f"AE 模型已加载: {ae_model_path}")

        # --- 2. 加载 Scalers ---
        lstm_input_scaler_path = pred_results.get('lstm_input_scaler_path')
        latent_target_scaler_path = pred_results.get('latent_salinity_target_scaler_path')
        if not lstm_input_scaler_path or not latent_target_scaler_path: raise ValueError("pred_results 中缺少 Scaler 路径")
        lstm_input_scaler = load_scaler_pkl(lstm_input_scaler_path)
        latent_target_scaler = load_scaler_pkl(latent_target_scaler_path)
        salinity_scaler_params = ae_results.get('salinity_scaler_params')
        if not lstm_input_scaler or not latent_target_scaler or not salinity_scaler_params: raise ValueError("一个或多个 Scaler 加载失败")
        log.info("所有 Scaler 加载成功")

        # --- 3. 加载预存的标准化测试数据 ---
        # 从配置文件获取路径 (更健壮)
        scaled_input_path = cfg.paths.get("lstm_scaled_input_path")
        scaled_target_path = cfg.paths.get("lstm_scaled_target_path")
        if not scaled_input_path or not scaled_target_path: raise ValueError("配置文件中缺少 lstm_scaled_input/target_path")
        if not os.path.exists(scaled_input_path) or not os.path.exists(scaled_target_path): raise FileNotFoundError("未找到预存的 scaled input/target npy 文件")

        test_lstm_inputs_scaled_all = np.load(scaled_input_path, allow_pickle=True).item() # 加载字典
        test_latent_salinity_true_scaled_all = np.load(scaled_target_path, allow_pickle=True).item() # 加载字典

        test_split_name = 'test'
        if test_split_name not in test_lstm_inputs_scaled_all or test_split_name not in test_latent_salinity_true_scaled_all:
            raise ValueError(f"预存的 scaled data npy 文件中缺少 '{test_split_name}' 的数据")

        test_lstm_inputs_scaled = test_lstm_inputs_scaled_all[test_split_name]
        # test_latent_salinity_true_scaled = test_latent_salinity_true_scaled_all[test_split_name] # 实际上迭代循环中不需要这个

        # --- 4. 加载原始高维真值数据 (Test) ---
        if 'salinity' not in all_processed_data or test_split_name not in all_processed_data['salinity']:
            raise ValueError("all_processed_data 中缺少 salinity['test'] 数据")
        test_salinity_processed = all_processed_data['salinity'][test_split_name]
        test_salinity_true_raw = inverse_scaling(test_salinity_processed, salinity_scaler_params)

        # --- 5. 准备迭代预测 ---
        sequence_length = cfg.model.prediction.lstm_ae.sequence_length
        max_steps = cfg.evaluation.get("iterative_forecast_steps", 5)
        num_samples = test_lstm_inputs_scaled.shape[0]
        num_rolls = num_samples - sequence_length - max_steps + 1

        if num_rolls <= 0:
            log.error(f"数据不足 ({num_samples}) 以进行 {max_steps} 步迭代预测 (序列长度 {sequence_length})。")
            return np.full((max_steps,), np.nan)

        # 获取目标范围 (与之前函数类似)
        # input_dims_list = pred_results.get('input_dims_list', [])
        input_dims_list = [15,1]
        input_source_names = ["salinity","flow"]
        if not input_dims_list or not input_source_names: raise ValueError("无法从 pred_results 获取 input_dims_list 或 input_source_names")
        input_ranges: Dict[str, tuple[int, int]] = {}
        current_start_idx = 0
        target_feature_name = 'salinity' # 假设目标是盐度
        target_range: Optional[tuple[int, int]] = None
        for name, dim in zip(input_source_names, input_dims_list):
             input_ranges[name] = (current_start_idx, current_start_idx + dim)
             # 假设名字包含 'salinity' 的是目标潜变量
             if 'salinity' in name.lower(): target_range = input_ranges[name]
             current_start_idx += dim
        if target_range is None: raise ValueError("无法确定目标 'salinity' 在输入中的范围")

        rmse_curve = []
        start_step = 2

        # --- 6. 执行迭代预测循环 ---
        for steps_ahead in range(start_step, max_steps + 1):
            errors_for_this_step = []
            for roll_start in range(num_rolls):
                # 准备初始序列 (scaled)
                current_sequence_scaled_np = test_lstm_inputs_scaled[roll_start : roll_start + sequence_length]
                current_sequence_scaled = torch.tensor(current_sequence_scaled_np, dtype=torch.float32).to(device)

                predicted_target_scaled_last_step_np = None # 存最终预测 (scaled latent)

                temp_sequence_tensor = current_sequence_scaled.clone()

                for step in range(steps_ahead):
                     # 使用标准 predict 方法 (传入单一样本的 batch)
                     input_batch_tensor = temp_sequence_tensor.unsqueeze(0) # Shape: (1, seq_len, features)

                     # --- VVVVVV 在这里添加包装代码 VVVVVV ---
                     # 1. 创建一个只包含当前批次的 TensorDataset
                     #    注意：确保 input_batch_tensor 在 CPU 或 GPU 上与你的 predict 方法期望的一致
                     #    如果 predict 方法内部处理 device，放 CPU 可能更安全；否则保持在 device 上。
                     #    我们假设 predict 能处理 device 上的 Tensor。
                     temp_dataset = torch.utils.data.TensorDataset(input_batch_tensor)

                     # 2. 创建一个 DataLoader，batch_size=1
                     temp_loader = torch.utils.data.DataLoader(temp_dataset, batch_size=1)
                     # --- ^^^^^^ 包装代码结束 ^^^^^^ ---

                     # 假设 predict 返回 NumPy 数组 (N, output_dim), N=1 in this case
                     # --- VVVVVV 修改调用，传入 temp_loader VVVVVV ---
                     pred_target_scaled_np = lstm_model_eval.predict(temp_loader) # <--- 传入包装好的 DataLoader
                     # --- ^^^^^^ 修改结束 ^^^^^^ ---

                     # predict 方法应该返回 (N, output_dim) 的 NumPy 数组，对于单步预测 N=1
                     # 可能需要 flatten() 来确保后续处理的是 1D 数组
                     pred_target_scaled_np = pred_target_scaled_np.flatten()

                     if step == steps_ahead - 1:
                          predicted_target_scaled_last_step_np = pred_target_scaled_np


                     # 构造下一步输入 (scaled)
                     current_time_idx_in_test = roll_start + sequence_length + step
                     next_input_features_known_scaled = test_lstm_inputs_scaled[current_time_idx_in_test]
                     next_input_scaled_np = next_input_features_known_scaled.copy()

                     s_start, s_end = target_range
                     if pred_target_scaled_np.shape[0] != (s_end - s_start):
                          log.error(f"预测维度 ({pred_target_scaled_np.shape[0]}) 与目标范围 ({s_end - s_start}) 不匹配 (step {step+1}/{steps_ahead}, roll {roll_start})")
                          predicted_target_scaled_last_step_np = None; break # Fail this roll

                     next_input_scaled_np[s_start:s_end] = pred_target_scaled_np
                     next_input_scaled_tensor = torch.tensor(next_input_scaled_np, dtype=torch.float32).to(device)

                     # 更新序列
                     temp_sequence_tensor = torch.cat([temp_sequence_tensor[1:], next_input_scaled_tensor.unsqueeze(0)], dim=0)

                # --- Loop for 'steps_ahead' finished ---
                if predicted_target_scaled_last_step_np is None:
                    errors_for_this_step.append(np.nan); continue

                # --- 7. 反变换到高维原始空间 ---
                try:
                    pred_latent_unscaled = latent_target_scaler.inverse_transform(predicted_target_scaled_last_step_np.reshape(1, -1))
                    recon_highdim_scaled = ae_model_eval.decode(pred_latent_unscaled)
                    recon_highdim_raw = inverse_scaling(recon_highdim_scaled, salinity_scaler_params).flatten()
                except Exception as e_inv:
                    log.error(f"反变换失败 (roll {roll_start}, step {steps_ahead}): {e_inv}", exc_info=True)
                    errors_for_this_step.append(np.nan); continue

                # --- 8. 获取真值并计算 RMSE ---
                gt_time_idx = roll_start + sequence_length + steps_ahead - 1
                try:
                    # 需要从原始未对齐的数据中获取真值
                    gt_highdim_raw = test_salinity_true_raw[gt_time_idx]
                    # 假设 gt_highdim_raw 已经是展平的有效点数据
                except IndexError:
                    log.error(f"无法获取 GT 数据索引 {gt_time_idx} (roll {roll_start}, step {steps_ahead})")
                    errors_for_this_step.append(np.nan); continue

                # 确保维度匹配 (假设都是有效点展平后)
                if recon_highdim_raw.shape[0] != gt_highdim_raw.shape[0]:
                     log.error(f"预测 ({recon_highdim_raw.shape}) 和 GT ({gt_highdim_raw.shape}) 的有效点维度不匹配")
                     errors_for_this_step.append(np.nan); continue

                mse = np.mean((recon_highdim_raw - gt_highdim_raw) ** 2)
                if np.isnan(mse): errors_for_this_step.append(np.nan)
                else: errors_for_this_step.append(np.sqrt(mse))

            # --- Loop for 'roll_start' finished ---
            valid_errors = [e for e in errors_for_this_step if not np.isnan(e)]
            avg_rmse_for_step = float(np.mean(valid_errors)) if valid_errors else np.nan
            rmse_curve.append(avg_rmse_for_step)
            log.debug(f"Step {steps_ahead}: Avg RMSE = {avg_rmse_for_step:.4f} (from {len(valid_errors)}/{num_rolls} rolls)")

        # --- Loop for 'steps_ahead' finished ---
        log.info("--- 完成最小改动版 AE-LSTM 迭代预测评估 ---")
        return np.array(rmse_curve)

    except (FileNotFoundError, ValueError) as setup_err:
        log.error(f"迭代预测评估设置失败: {setup_err}", exc_info=True)
        return np.full((cfg.evaluation.get("iterative_forecast_steps", 5),), np.nan) # Return NaNs on setup failure
    except Exception as e_runtime:
        log.error(f"迭代预测评估运行时出错: {e_runtime}", exc_info=True)
        return np.full((cfg.evaluation.get("iterative_forecast_steps", 5),), np.nan) # Return NaNs on runtime failure

# === 辅助函数: Scaler 处理 ===
# !! 重要: 这些函数需要你根据 .npy 文件的实际内容进行适配 !!
def load_scaler_params_from_npy(path):
    """示例: 从 .npy 加载 Scaler 参数 (需适配)。"""
    log.debug(f"尝试从 {path} 加载 Scaler 参数 (.npy)...")
    if not os.path.exists(path):
        log.error(f"Scaler NPY 文件未找到: {path}")
        raise FileNotFoundError(f"Scaler NPY 文件未找到: {path}")
    try:
        # 假设 .npy 存储了 {'mean': mean_array, 'std': std_array}
        params = np.load(path, allow_pickle=True).item()
        log.info(f"成功从 {path} 加载 Scaler 参数 (mean/std)")
        # 基本检查
        if 'mean' not in params or 'std' not in params:
             raise ValueError("Loaded scaler params missing 'mean' or 'std' key.")
        return params
    except Exception as e:
        log.error(f"从 {path} 加载/解析 Scaler NPY 时出错: {e}", exc_info=True)
        raise e

def load_processed_data(cfg: DictConfig) -> dict:
    """加载所有预处理的数据，供超参数搜索使用"""
    log.info("加载预处理数据...")
    all_processed_data = {}
    mmap_mode = 'r' if cfg.get('use_mmap', False) else None
    try:
        # --- 直接加载数据 ---
        # 盐度数据
        salinity_data = {}
        for split, path in cfg.paths.processed_paths.salinity.items():
            if not os.path.exists(path): raise FileNotFoundError(f"文件未找到: {path}")
            data_array = np.load(path, mmap_mode=mmap_mode)
            salinity_data[split] = data_array
        all_processed_data['salinity'] = salinity_data
        
        # 风场数据
        if hasattr(cfg.paths.processed_paths, 'wind'):
            wind_data = {}
            for split, path in cfg.paths.processed_paths.wind.items():
                if not os.path.exists(path): raise FileNotFoundError(f"文件未找到: {path}")
                data_array = np.load(path)
                wind_data[split] = data_array
            all_processed_data['wind'] = wind_data
        
        # 径流数据
        if hasattr(cfg.paths.processed_paths, 'flow'):
            flow_data = {}
            for split, path in cfg.paths.processed_paths.flow.items():
                if not os.path.exists(path): raise FileNotFoundError(f"文件未找到: {path}")
                data_array = np.load(path)
                if data_array.ndim == 1: data_array = data_array.reshape(-1, 1)
                flow_data[split] = data_array
            all_processed_data['flow'] = flow_data
            
        return all_processed_data
        
    except Exception as e:
        log.error(f"加载预处理数据时出错: {e}", exc_info=True)
        return {}

def apply_scaling(data, scaler_params):
    """示例: 使用加载的参数应用标准化 (需适配)。"""
    if not scaler_params: return data
    mean = scaler_params['mean']
    std = scaler_params['std']
    # 确保维度匹配 (广播可能处理大部分情况，但显式检查更好)
    if data.shape[1] != mean.shape[0] or data.shape[1] != std.shape[0]:
         log.error(f"Data dimension ({data.shape[1]}) incompatible with scaler dimensions (mean: {mean.shape[0]}, std: {std.shape[0]}).")
         # 根据情况决定是 reshape 还是报错
         # 尝试 reshape scaler (如果 scaler 保存的是 1D)
         if mean.ndim == 1 and std.ndim == 1:
              mean = mean.reshape(1, -1)
              std = std.reshape(1, -1)
         else: # 无法安全处理，报错
              raise ValueError("Scaler dimension mismatch.")
    std = np.where(std == 0, 1e-8, std) # 防除零
    return (data - mean) / std

def inverse_scaling(scaled_data, scaler_params):
    """示例: 使用加载的参数应用逆标准化 (需适配)。"""
    if not scaler_params: return scaled_data
    mean = scaler_params['mean']
    std = scaler_params['std']
    # 维度检查/调整
    if scaled_data.shape[1] != mean.shape[0] or scaled_data.shape[1] != std.shape[0]:
         log.error(f"Data dimension ({scaled_data.shape[1]}) incompatible with scaler dimensions (mean: {mean.shape[0]}, std: {std.shape[0]}).")
         if mean.ndim == 1 and std.ndim == 1:
              mean = mean.reshape(1, -1)
              std = std.reshape(1, -1)
         else: raise ValueError("Scaler dimension mismatch.")
    return (scaled_data * std) + mean

# 用于运行时生成的 Scaler (.pkl)
def save_scaler_pkl(scaler, path):
    """保存运行时生成的 Sklearn Scaler 对象为 .pkl"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(scaler, path)
        log.info(f"运行时 Scaler 对象已保存至 {path}")
    except Exception as e:
        log.error(f"保存运行时 Scaler 到 {path} 时出错: {e}")

def load_scaler_pkl(path):
    """加载运行时生成的 Sklearn Scaler 对象 (.pkl)"""
    try:
        if not os.path.exists(path):
             log.error(f"运行时 Scaler PKL 文件未找到: {path}")
             raise FileNotFoundError(f"运行时 Scaler PKL 文件未找到: {path}")
        scaler = joblib.load(path)
        log.info(f"成功从 {path} 加载运行时 Scaler 对象")
        return scaler
    except Exception as e:
        log.error(f"从 {path} 加载运行时 Scaler PKL 时出错: {e}", exc_info=True)
        raise e

# 辅助函数：拟合 StandardScaler 并转换数据字典
def fit_transform_with_sklearn_scaler(data_dict, split='train'):
    scaler = StandardScaler()
    train_data = data_dict[split]
    if train_data.ndim == 1: train_data = train_data.reshape(-1, 1)
    scaler.fit(train_data)
    transformed_data = {}
    for s, data_array in data_dict.items():
        if data_array.ndim == 1: data_array = data_array.reshape(-1, 1)
        transformed_data[s] = scaler.transform(data_array)
    return scaler, transformed_data

# === 1. 盐度 Autoencoder 步骤 ===
def run_ae_on_salinity(cfg: DictConfig, all_processed_data: dict):
    """在预处理的盐度数据上训练 AE。"""
    log.info("--- 开始步骤 1: 盐度 Autoencoder ---")
    results = {'success': False}
    salinity_data = all_processed_data.get('salinity')
    if not salinity_data:
        log.error("未找到预处理的盐度数据。")
        return results

    try:
        # 1.1 加载预计算的盐度 Scaler 参数 (.npy)
        log.info(f"加载预计算盐度 Scaler 参数: {cfg.paths.scaler_paths.salinity}")
        salinity_scaler_params = load_scaler_params_from_npy(cfg.paths.scaler_paths.salinity) # 需适配
        results['salinity_scaler_params'] = salinity_scaler_params # 存储用于重建

        # 假设 salinity_data 已经是标准化后的数据
        salinity_scaled = salinity_data
        log.info("假定加载的盐度数据已标准化。")

        # 1.2 初始化和训练 AE
        log.info("初始化和训练盐度 Autoencoder...")
        ae_config = cfg.model.dimensionality_reduction.autoencoder
        # --- 获取输入维度 ---
        actual_salinity_dim = salinity_scaled['train'].shape[1]
        # 从配置获取 input_dim (如果定义了)，否则使用实际维度
        input_dim = ae_config.get('input_dim', actual_salinity_dim) # 使用 .get 提供默认值
        if actual_salinity_dim != input_dim:
             log.warning(f"Configured/default input_dim ({input_dim}) != actual data dim ({actual_salinity_dim}). Using actual dim from data.")
             input_dim = actual_salinity_dim # 强制使用实际维度

        # --- 初始化 AutoencoderDimensionalityReduction (!! 主要修改 !!) ---
        ae_model = AutoencoderDimensionalityReduction(
            # **必需参数:**
            input_dim=input_dim, # 传递输入维度
            # **模型结构参数 (从配置读取):**
            encoding_dim=ae_config.encoding_dim,
            hidden_layers=list(ae_config.hidden_layers) if ae_config.hidden_layers else [], # 转为列表或空列表
            activation=ae_config.activation,
            dropout_rate=ae_config.get('dropout_rate', 0.0), # 使用 .get 提供默认值
            # **训练相关参数 (部分在 __init__ 中设置):**
            learning_rate=ae_config.learning_rate,
            weight_decay=ae_config.get('weight_decay', 0.0), # 使用 .get 提供默认值
            device=device, # 传递计算设备
            random_seed=cfg.get('random_seed', None) # 从主配置获取随机种子
        )

        # --- DataLoaders (保持不变) ---
        train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(salinity_scaled['train'])) # .to(device) 可以省略，在训练循环内部处理
        val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(salinity_scaled['val']))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.model.dimensionality_reduction.autoencoder.batch_size, shuffle=True) # 从 training 配置获取 batch_size
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.model.dimensionality_reduction.autoencoder.batch_size, shuffle=False) # 从 training 配置获取 batch_size

        # --- 调用 fit 方法 (!! 主要修改 !!) ---
        ae_model.fit(train_loader, val_loader, epochs=cfg.model.dimensionality_reduction.autoencoder.epochs)

        # 1.3 保存 AE 模型到 *运行目录* (保持不变)
        ae_model_path = cfg.paths.ae_model_path
        os.makedirs(os.path.dirname(ae_model_path), exist_ok=True)
        ae_model.save(ae_model_path) # 调用修改后的 save 方法
        log.info(f"盐度 AE 模型已保存至 (运行目录): {ae_model_path}")
        results['model_path'] = ae_model_path
        results['encoding_dim'] = ae_config.encoding_dim # 保存编码维度

        # 1.4 编码盐度数据 -> 生成潜空间向量并保存到 *运行目录*
        log.info("编码盐度数据生成潜空间向量...")
        latent_salinity_paths = {}
        latent_salinity_data = {}
        for split in salinity_scaled.keys():
            # 调用修改后的 encode 方法
            Z_sal_np = ae_model.encode(salinity_scaled[split]) # 直接传递 NumPy 数组
            latent_salinity_data[split] = Z_sal_np
            latent_path = cfg.paths.latent_salinity_paths[split]
            os.makedirs(os.path.dirname(latent_path), exist_ok=True)
            np.save(latent_path, Z_sal_np)
            latent_salinity_paths[split] = latent_path
            log.info(f"{split} 的潜空间盐度 ({Z_sal_np.shape}) 已保存至 (运行目录): {latent_path}")
        results['latent_salinity_paths'] = latent_salinity_paths
        results['latent_salinity_data'] = latent_salinity_data

        results['success'] = True
        log.info("--- 完成步骤 1: 盐度 Autoencoder ---")

    except Exception as e:
        log.error(f"盐度 Autoencoder 步骤出错: {e}", exc_info=True)
        results['success'] = False
    return results

# === 2. 风场数据处理步骤 (简化: 只加载预计算的 PCA 结果) ===
# (这个函数现在只加载预计算的 PCA .npy 文件)
def run_wind_processing(cfg: DictConfig): # 不再需要 all_processed_data
    """加载预先计算好的低维风场数据 (PCA 组件)。"""
    log.info("--- 开始步骤 2: 加载预计算的低维风场数据 (PCA) ---")
    results = {'success': False}
    processed_wind_data = {}
    # 使用 cfg.paths.processed_paths.wind_pca 访问路径
    pca_wind_paths = cfg.paths.processed_paths.get('wind_pca')
    if not pca_wind_paths:
        log.error("配置中未找到 'paths.processed_paths.wind_pca'。")
        return results

    try:
        # 加载预计算的 PCA 低维风场数据
        # !! 注意路径现在来自 cfg.paths.processed_paths.wind_pca !!
        log.info(f"加载预计算的 PCA 风场数据...")
        actual_dims = []
        for split, path in pca_wind_paths.items():
            log.debug(f"  加载 {split} 分割从: {path}")
            if not os.path.exists(path):
                 log.error(f"预计算的 PCA 风场文件未找到: {path}")
                 raise FileNotFoundError(f"预计算的 PCA 风场文件未找到: {path}")
            data_array = np.load(path)
            actual_dims.append(data_array.shape[1])
            processed_wind_data[split] = data_array
            log.info(f"  已加载 {split} 的 PCA 风场数据，形状: {data_array.shape}")

        # 检查加载的维度是否一致，并与配置比较
        if len(set(actual_dims)) > 1:
            raise ValueError(f"加载的 PCA 风场数据维度不一致: {actual_dims}")
        actual_dim = actual_dims[0]
        configured_dim = cfg.data.wind_pca_dim
        if actual_dim != configured_dim:
            log.warning(f"加载的 PCA 风场维度 ({actual_dim}) 与配置 ({configured_dim}) 不符。将使用实际加载的维度。")
            results['processed_wind_dim'] = actual_dim
        else:
            results['processed_wind_dim'] = configured_dim

        # (加载预训练 PCA 模型部分保持不变，如果需要的话)

        results['processed_wind_data'] = processed_wind_data
        results['success'] = True
        log.info("--- 完成步骤 2: 加载预计算的低维风场数据 ---")

    except Exception as e:
        log.error(f"加载预计算 PCA 风场数据步骤出错: {e}", exc_info=True)
        results['success'] = False
    return results

# === 2.5 径流数据处理步骤 (简化: 只加载预计算的径流数据) ===
def run_flow_processing(cfg: DictConfig):
    """加载预先计算好的径流数据。支持 t x 1 或 t x 3 维度的数据。"""
    log.info("--- 开始步骤: 加载预计算的径流数据 ---")
    results = {'success': False}
    processed_flow_data = {}
    
    # 使用配置中的径流数据路径
    flow_paths = cfg.paths.processed_paths.get('flow')
    if not flow_paths:
        log.warning("配置中未找到径流数据路径，流场数据将不可用")
        return results
        
    try:
        # 加载径流数据
        log.info("加载预计算的径流数据...")
        actual_dims = []
        for split, path in flow_paths.items():
            if not os.path.exists(path):
                log.error(f"径流数据文件未找到: {path}")
                raise FileNotFoundError(f"径流数据文件未找到: {path}")
                
            data_array = np.load(path)
            
            # 处理不同的维度格式 (t x 1 或 t x 3)
            if data_array.ndim == 1:
                # 如果是1D数组，则重塑为 t x 1
                data_array = data_array.reshape(-1, 1)
                log.info(f"  将1D径流数据重塑为 {data_array.shape}")
            elif data_array.ndim > 2:
                # 如果维度大于2，尝试重塑为保留第一维度
                original_shape = data_array.shape
                data_array = data_array.reshape(data_array.shape[0], -1)
                log.info(f"  将径流数据从形状 {original_shape} 重塑为 {data_array.shape}")
                
            actual_dims.append(data_array.shape[1])
            processed_flow_data[split] = data_array
            log.info(f"已加载{split}的径流数据，形状: {data_array.shape}")
            
        # 检查维度一致性
        if len(set(actual_dims)) > 1:
            log.warning(f"警告：径流数据的不同分割具有不同的维度: {actual_dims}")
            # 可以继续处理，但需要提醒用户
            
        actual_dim = actual_dims[0]  # 使用第一个分割的维度作为参考
        configured_dim = cfg.data.get('flow_dim', actual_dim)
        if actual_dim != configured_dim:
            log.warning(f"加载的径流数据维度({actual_dim})与配置({configured_dim})不符，使用实际维度")
            
        results['processed_flow_data'] = processed_flow_data
        results['processed_flow_dim'] = actual_dim
        results['success'] = True
        log.info(f"--- 完成径流数据加载：维度 = {actual_dim} ---")
        
    except Exception as e:
        log.error(f"加载径流数据时出错: {e}", exc_info=True)
        results['success'] = False
        
    return results

# === 3. LSTM 预测步骤 ===
def run_lstm_prediction(cfg, ae_results, wind_results=None, flow_results=None):
    """
    运行LSTM预测 - 支持多步预测
    """
    log.info("--- 开始步骤 3: LSTM 预测 ---")
    results = {'success': False}
    
    # 检查必要的前置步骤是否成功，盐度数据是必须的
    if not ae_results.get('success'):
        log.error("因盐度AE步骤失败，跳过LSTM预测。")
        return results

    try:
        # 3.1 准备所有可能的输入数据源
        log.info("根据配置选择性加载输入数据源...")
        
        # 确定要使用的数据源
        input_sources = []  # 存储实际数据
        input_dims = []     # 存储每个数据源的维度
        data_source_names = []  # 存储数据源名称（用于日志）
        
        # 从配置中读取使用哪些数据源
        input_config = cfg.model.prediction.get("input_sources", {})
        use_salinity = input_config.get("use_salinity", True)  # 默认使用盐度
        use_wind = input_config.get("use_wind", True)  # 默认使用风场
        use_flow = input_config.get("use_flow", False)  # 默认不使用流场
        
        # 加载并添加盐度数据（必须）
        Z_sal = ae_results.get('latent_salinity_data')
        if not Z_sal:
            Z_sal = {split: np.load(path) for split, path in ae_results['latent_salinity_paths'].items()}
        
        if use_salinity:
            input_sources.append(Z_sal)
            input_dims.append(ae_results['encoding_dim'])
            data_source_names.append("潜空间盐度")
        
        # 加载并添加风场数据（如果配置启用）
        if use_wind:
            if not wind_results.get('success'):
                log.warning("风场数据处理未成功，将不使用风场数据。")
            else:
                Proc_Wind = wind_results['processed_wind_data']
                input_sources.append(Proc_Wind)
                input_dims.append(wind_results['processed_wind_dim'])
                data_source_names.append("PCA风场")
        
        # 加载并添加流场数据（如果配置启用）
        if use_flow:
            if not flow_results or not flow_results.get('success'):
                log.warning("流场数据处理未成功或未提供，将不使用流场数据。")
            else:
                Proc_Flow = flow_results['processed_flow_data']
                input_sources.append(Proc_Flow)
                input_dims.append(flow_results['processed_flow_dim'])
                data_source_names.append("流场")
        
        # 确认至少有一个数据源可用
        if not input_sources:
            raise ValueError("没有可用的输入数据源")
        
        log.info(f"LSTM将使用以下数据源: {', '.join(data_source_names)}")
        
        
        # 3.2 确定所有数据源共有的分割
        common_splits = set(Z_sal.keys())  # 从盐度数据初始化
        for src in input_sources:
            common_splits = common_splits.intersection(set(src.keys()))
        
        if not common_splits:
            raise ValueError("输入数据源之间没有共同的分割")
        
        log.info(f"将处理以下共有分割: {common_splits}")
        
        # 3.3 动态组合数据
        lstm_input_data = {}
        for split in common_splits:
            data_parts = [src[split] for src in input_sources]
            lstm_input_data[split] = np.concatenate(data_parts, axis=1)
            log.info(f"{split}分割组合数据形状: {lstm_input_data[split].shape}")
            
        # 后续逻辑与原函数相同
        # 3.4 标准化组合的输入数据
        log.info("标准化组合的LSTM输入数据...")
        lstm_input_scaler_path = cfg.paths.lstm_input_scaler_path
        lstm_input_scaler, lstm_input_scaled = fit_transform_with_sklearn_scaler(lstm_input_data, 'train')
        save_scaler_pkl(lstm_input_scaler, lstm_input_scaler_path)
        results['lstm_input_scaler_path'] = lstm_input_scaler_path
        
        # 3.5 标准化LSTM目标（只有潜空间盐度）
        log.info("标准化LSTM目标数据（潜空间盐度）...")
        latent_salinity_target_scaler_path = cfg.paths.latent_salinity_target_scaler_path
        latent_target_scaler, Z_sal_target_scaled = fit_transform_with_sklearn_scaler(Z_sal, 'train')
        save_scaler_pkl(latent_target_scaler, latent_salinity_target_scaler_path)
        results['latent_salinity_target_scaler_path'] = latent_salinity_target_scaler_path

        # ————————————————————————————————————————————————————————————————————————————————————————————————————————
            
        
        # 3.6 创建 LSTM 序列
        log.info("为 LSTM 创建序列...")
        sequences = {}
        seq_len = cfg.model.prediction.lstm_ae.sequence_length
        for split in lstm_input_scaled.keys():
            if split not in Z_sal_target_scaled: continue
            # 确保数据足够长以创建序列
            if lstm_input_scaled[split].shape[0] <= seq_len or Z_sal_target_scaled[split].shape[0] <= seq_len:
                log.warning(f"Split '{split}' data length ({lstm_input_scaled[split].shape[0]}) is not greater than sequence length ({seq_len}). Skipping sequence creation.")
                continue
            X_seq, _ = sequence_utils.create_sequences_ae(lstm_input_scaled[split], seq_len)
            _, y_seq = sequence_utils.create_sequences_ae(Z_sal_target_scaled[split], seq_len)
            if X_seq.shape[0] != y_seq.shape[0]: raise ValueError(f"X_seq 和 y_seq 的样本数不匹配: {X_seq.shape[0]} vs {y_seq.shape[0]}")
            if X_seq.shape[0] == 0: continue
            sequences[split] = {'X': X_seq, 'y': y_seq}
            log.info(f"{split} 的序列: X={X_seq.shape}, y={y_seq.shape}")
        results['target_sequences'] = {split: sequences[split]['y'] for split in sequences}
        if 'train' not in sequences or sequences['train']['X'].shape[0] == 0: raise ValueError("未创建训练序列。")

        # 3.6 初始化和训练 LSTM
        lstm_input_dim = sum(input_dims)
        lstm_output_dim = ae_results['encoding_dim']
        log.info(f"LSTM输入维度: {lstm_input_dim}, 输出维度: {lstm_output_dim}")
        lstm_model = LSTMPredictionModel(
            input_size=lstm_input_dim,
            hidden_size=cfg.model.prediction.lstm_ae.hidden_size,
            num_layers=cfg.model.prediction.lstm_ae.num_layers,
            output_size=lstm_output_dim,
            dropout=cfg.model.prediction.lstm_ae.get('dropout', 0.0),
            device=device
        )
        # DataLoaders 和 fit
        train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(sequences['train']['X']).to(device), torch.FloatTensor(sequences['train']['y']).to(device))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.model.prediction.lstm_ae.batch_size, shuffle=True)
        val_loader = None
        if 'val' in sequences and sequences['val']['X'].shape[0] > 0:
            val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(sequences['val']['X']).to(device), torch.FloatTensor(sequences['val']['y']).to(device))
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.model.prediction.lstm_ae.batch_size, shuffle=False)
        lstm_model.fit(
            train_loader,
            val_loader,
            epochs=cfg.model.prediction.lstm_ae.epochs, # 从 training 配置获取
            learning_rate=cfg.model.prediction.lstm_ae.learning_rate, # 从 model 配置获取
            patience=cfg.model.prediction.lstm_ae.get('patience', 10) # 从 training 配置获取，带默认值
        )
        # 3.7 保存 LSTM 模型到 *运行目录*
        lstm_model_path = cfg.paths.lstm_model_path # 指向 output_dir/.../model.pth
        os.makedirs(os.path.dirname(lstm_model_path), exist_ok=True)
        lstm_model.save(lstm_model_path)
        log.info(f"LSTM 模型已保存至 (运行目录): {lstm_model_path}")
        results['model_path'] = lstm_model_path

        # 3.8 使用 LSTM 进行预测并保存到 *运行目录*
        log.info("使用 LSTM 预测未来的潜空间盐度向量...")
        predicted_latent_salinity_paths = {}
        predicted_latent_data_scaled = {}
        for split in sequences.keys():
            if sequences[split]['X'].shape[0] == 0: continue
            split_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.FloatTensor(sequences[split]['X']).to(device)),
                batch_size=cfg.model.prediction.lstm_ae.batch_size,
                shuffle=False
            )
            with torch.no_grad(): # 预测时不需要梯度
                # 直接将 predict 的返回值 (NumPy 数组) 赋给 NumPy 变量
                Z_pred_sal_scaled_np = lstm_model.predict(split_loader)
            # 不再需要 .cpu().numpy() 转换
            predicted_latent_data_scaled[split] = Z_pred_sal_scaled_np
            # 保存预测的 *标准化的* 潜空间盐度到 *运行目录*
            pred_path = cfg.paths.predicted_latent_salinity_paths[split] # 指向 output_dir/.../predicted...npy
            os.makedirs(os.path.dirname(pred_path), exist_ok=True)
            np.save(pred_path, Z_pred_sal_scaled_np)
            predicted_latent_salinity_paths[split] = pred_path
            log.info(f"{split} 的预测潜空间盐度 ({Z_pred_sal_scaled_np.shape}) 已保存至 (运行目录): {pred_path}")
        results['predicted_latent_salinity_paths'] = predicted_latent_salinity_paths
        results['predicted_latent_data_scaled'] = predicted_latent_data_scaled

        # --- (最小改动) 保存标准化后的数据以供后续迭代评估使用 ---
        try:
            scaled_input_path = cfg.paths.get("lstm_scaled_input_path", os.path.join(os.getcwd(), "intermediate", "lstm_input_scaled.npy")) # 从配置获取路径或默认
            scaled_target_path = cfg.paths.get("lstm_scaled_target_path", os.path.join(os.getcwd(), "intermediate", "lstm_target_scaled.npy")) # 从配置获取路径或默认
            os.makedirs(os.path.dirname(scaled_input_path), exist_ok=True)
            os.makedirs(os.path.dirname(scaled_target_path), exist_ok=True)

            # 保存包含所有分割的字典，或者只保存 test 分割的数据
            # 保存字典更灵活
            np.save(scaled_input_path, lstm_input_scaled)
            np.save(scaled_target_path, Z_sal_target_scaled)
            log.info(f"已保存标准化 LSTM 输入数据至: {scaled_input_path}")
            log.info(f"已保存标准化 LSTM 目标数据至: {scaled_target_path}")
            # 将路径存入 results，方便后续查找 (可选)
            results['lstm_scaled_input_path'] = scaled_input_path
            results['lstm_scaled_target_path'] = scaled_target_path
        except Exception as e_save:
            log.warning(f"无法保存标准化的 LSTM 输入/目标数据: {e_save}", exc_info=True)
            # 不应让此失败导致整个步骤失败，但后续迭代评估可能无法进行

        results['success'] = True
        log.info("--- 完成步骤 3: LSTM 预测 ---")
        
    except Exception as e:
        log.error(f"LSTM 预测步骤出错: {e}", exc_info=True)
        results['success'] = False
        
    return results


# === 4. 盐度重建步骤 ===
def run_salinity_reconstruction(cfg: DictConfig, ae_results: dict, pred_results: dict):
    """重建盐度场。"""
    log.info("--- 开始步骤 4: 盐度重建 ---")
    results = {'success': False}
    if not ae_results.get('success') or not pred_results.get('success'):
        log.error("因先前步骤失败，跳过重建。")
        return results

    try:
        # 4.1 加载模型和 Scaler
        log.info("加载模型和 scaler 用于重建...")
        ae_model = AutoencoderDimensionalityReduction.load(ae_results['model_path'], device=device)
        ae_model.model.eval()
        latent_target_scaler = load_scaler_pkl(pred_results['latent_salinity_target_scaler_path']) # 加载运行时 .pkl
        salinity_scaler_params = ae_results['salinity_scaler_params'] # 使用第一步加载的 .npy 参数
        if not latent_target_scaler or not salinity_scaler_params:
             raise ValueError("重建所需的 Scaler 未能成功加载。")

        # 4.2 加载预测的潜空间盐度
        predicted_latent_data_scaled = pred_results.get('predicted_latent_data_scaled', {})
        if not predicted_latent_data_scaled:
            predicted_latent_data_scaled = {split: np.load(path) for split, path in pred_results['predicted_latent_salinity_paths'].items()}
        # 添加: 处理主要预测结果的重建
        log.info("重建主要预测结果...")
        reconstructed_data = {}
        reconstructed_paths = {}
        
        for split, Z_pred_sal_scaled in predicted_latent_data_scaled.items():
            # 逆标准化潜空间数据
            Z_pred_sal_orig = latent_target_scaler.inverse_transform(Z_pred_sal_scaled)
            
            # 使用AE模型解码
            X_rec_sal_scaled_np = ae_model.decode(Z_pred_sal_orig)
            
            # 逆变换回原始尺度
            X_reconstructed_salinity = inverse_scaling(X_rec_sal_scaled_np, salinity_scaler_params)
            
            # 保存重建的数据
            reconstructed_data[split] = X_reconstructed_salinity
            
            # 保存到文件
            rec_path = cfg.paths.reconstructed_salinity_paths[split]
            os.makedirs(os.path.dirname(rec_path), exist_ok=True)
            np.save(rec_path, X_reconstructed_salinity)
            reconstructed_paths[split] = rec_path
            log.info(f"{split}的重建盐度({X_reconstructed_salinity.shape})已保存至: {rec_path}")
        
        # 将重建数据添加到结果中
        results['reconstructed_data'] = reconstructed_data
        results['reconstructed_paths'] = reconstructed_paths

     
        results['success'] = True
        log.info("--- 完成步骤 4: 盐度重建 ---")

    except Exception as e:
        log.error(f"重建步骤出错: {e}", exc_info=True)
        results['success'] = False
    return results

# === 5. 评估步骤 ===
def run_evaluation(cfg: DictConfig, rec_results: dict, all_processed_data: dict, ae_results: dict, pred_results: dict):
    """评估重建结果，对所有可用的分割 (train, val, test) 进行详细评估和可视化。"""
    log.info("--- 开始步骤 5: 评估 (详细版 - 所有分割) ---")
    results = {'success': False, 'evaluation_details': {}} # 初始化 success 和详细结果字典
    # eval_split = 'test' # <--- 删除硬编码

    # --- 检查输入 ---
    # 检查重建结果是否存在且包含数据
    reconstructed_data_all_splits = rec_results.get('reconstructed_data')
    if not rec_results.get('success') or not reconstructed_data_all_splits:
        log.error("重建失败或缺少重建数据，跳过评估。")
        return results
    # 检查原始数据是否存在
    original_salinity_all_splits = all_processed_data.get('salinity')
    if not original_salinity_all_splits:
        log.error("未找到原始处理后的盐度数据，跳过评估。")
        return results
    # 检查 Scaler 参数是否存在
    if 'salinity_scaler_params' not in ae_results:
        log.error("未找到原始盐度 Scaler 参数用于评估。跳过评估。")
        return results

    # --- 创建输出目录 (保持不变) ---
    hydra_output_dir = os.getcwd()
    results_dir = os.path.join(hydra_output_dir, cfg.paths.results_subdir)
    plots_dir = os.path.join(hydra_output_dir, cfg.paths.plots_subdir)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    log.info(f"评估结果将保存在: {results_dir} 和 {plots_dir}")

    # --- 加载 Mask (保持不变) ---
    try:
        # ... (加载 Mask 的逻辑不变) ...
        mask = load_mask(cfg) # 或 np.load
        if mask is None: raise ValueError("Failed to load mask.")
        boolean_mask = (mask == 0)
        target_spatial_shape = mask.shape
        num_valid_points_mask = np.sum(boolean_mask)
        total_points = mask.size
        log.info(f"加载 Mask 形状: {target_spatial_shape}, 有效点数: {num_valid_points_mask}")
    except Exception as e_mask:
        log.error(f"加载 Mask 时出错: {e_mask}", exc_info=True)
        return results

    # --- 获取要处理的分割列表 ---
    splits_to_evaluate = list(reconstructed_data_all_splits.keys()) # 通常是 ['train', 'val', 'test']
    log.info(f"将对以下分割进行评估: {splits_to_evaluate}")

    evaluation_successful_for_any_split = False # 跟踪是否有任何分割成功

    # --- !!! 添加循环，遍历所有分割 !!! ---
    for split in splits_to_evaluate:
        log.info(f"\n--- 开始评估分割: {split} ---")
        figure_paths_split = [] # 重置当前分割的图表列表
        split_metrics = {}      # 存储当前分割的指标
        split_success = False   # 标记当前分割是否成功

        # 检查当前分割的数据是否存在
        if split not in original_salinity_all_splits:
            log.warning(f"跳过评估 {split}，因为缺少对应的原始数据。")
            results['evaluation_details'][split] = {"error": "Missing original data"}
            continue
        if split not in reconstructed_data_all_splits:
             log.warning(f"跳过评估 {split}，因为缺少对应的重建数据。")
             results['evaluation_details'][split] = {"error": "Missing reconstructed data"}
             continue

        try:
            # --- 5.1 加载当前分割的数据 (原始尺度) ---
            log.info(f"加载用于评估的数据 ({split} split)...")
            X_rec_sal = reconstructed_data_all_splits[split]
            X_orig_sal_processed = original_salinity_all_splits[split]
            salinity_scaler_params = ae_results['salinity_scaler_params']
            X_orig_sal_raw_scale = inverse_scaling(X_orig_sal_processed, salinity_scaler_params) # 需适配

            # --- 5.2 对齐序列 (原始尺度) ---
            seq_len = cfg.model.prediction.lstm_ae.sequence_length
            orig_eval = X_orig_sal_raw_scale[seq_len:]
            recon_eval = X_rec_sal # 假设重建数据已是对齐长度

            if orig_eval.shape[0] != recon_eval.shape[0]:
                 log.warning(f"分割 '{split}': 对齐后时间步不匹配: 原始 {orig_eval.shape[0]}, 重建 {recon_eval.shape[0]}。截断以匹配。")
                 min_len = min(orig_eval.shape[0], recon_eval.shape[0])
                 orig_eval = orig_eval[:min_len]
                 recon_eval = recon_eval[:min_len]
            if orig_eval.shape[0] == 0:
                 log.warning(f"分割 '{split}': 对齐后数据样本数为 0。跳过此分割。")
                 results['evaluation_details'][split] = {"error": "Zero samples after alignment"}
                 continue
            n_samples = orig_eval.shape[0]
            n_features_data = orig_eval.shape[1]
            log.info(f"用于评估的数据 ({split}, 原始尺度, 已对齐): 原始形状={orig_eval.shape}, 重建形状={recon_eval.shape}")

            # --- 5.3 数据准备与验证 ---
            log.info(f"验证数据维度并准备空间场 ({split})...")
            if n_features_data != num_valid_points_mask:
                log.error(f"分割 '{split}': 数据特征数 ({n_features_data}) 与 Mask 有效点数 ({num_valid_points_mask}) 不匹配！")
                results['evaluation_details'][split] = {"error": "Feature/Mask dimension mismatch"}
                continue # 跳到下一个分割
            log.info(f"分割 '{split}': 数据特征数与 Mask 有效点数匹配。")

            orig_valid_flat = orig_eval
            recon_valid_flat = recon_eval

            # 重塑回空间网格
            log.info(f"将有效点数据放回空间网格 ({split})...")
            recon_spatial = np.full((n_samples,) + target_spatial_shape, np.nan)
            orig_spatial_masked = np.full((n_samples,) + target_spatial_shape, np.nan)
            diff_spatial = np.full((n_samples,) + target_spatial_shape, np.nan)
            if boolean_mask.shape != target_spatial_shape: raise ValueError(...) # 安全检查

            for t in range(n_samples):
                recon_spatial[t, boolean_mask] = recon_valid_flat[t]
                orig_spatial_masked[t, boolean_mask] = orig_valid_flat[t]
            diff_spatial = np.where(boolean_mask, recon_spatial - orig_spatial_masked, np.nan)
            log.info(f"空间场数据已准备好 ({split})。")

            # --- 5.4 计算指标 ---
            log.info(f"计算评估指标 ({split})...")
            current_metrics = {} # 存储当前分割的指标
            rmse_field = np.full(target_spatial_shape, np.nan)

            if num_valid_points_mask > 0:
                # ... (指标计算逻辑与之前相同, 使用 orig_valid_flat, recon_valid_flat, diff_spatial) ...
                 current_metrics["mean_rmse"] = np.sqrt(eval_metrics.calculate_mse(orig_valid_flat, recon_valid_flat))
                 current_metrics["mean_mae"] = eval_metrics.calculate_mae(orig_valid_flat, recon_valid_flat)
                 rmse_field = np.sqrt(np.nanmean(np.square(diff_spatial), axis=0))
                 current_metrics["rmse_map"] = rmse_field
                 current_metrics["max_rmse"] = np.nanmax(rmse_field) if np.any(np.isfinite(rmse_field)) else np.nan
                 current_metrics["min_rmse"] = np.nanmin(rmse_field) if np.any(np.isfinite(rmse_field)) else np.nan
                 if cfg.evaluation.get("calculate_correlation", False):
                    try:
                        corr_map = calculate_correlation_map(orig_spatial_masked, recon_spatial, boolean_mask)
                        current_metrics['correlation_map'] = corr_map
                        mean_corr = np.nanmean(corr_map) if np.any(np.isfinite(corr_map)) else np.nan
                        current_metrics['mean_correlation'] = mean_corr
                        log.info(f"  {split} - 平均空间相关性: {mean_corr:.6f}")
                    except Exception as e_corr:
                        log.error(f"计算相关性时出错: {e_corr}", exc_info=True)
                        current_metrics['mean_correlation'] = np.nan
                        # 确保错误处理中也设置correlation_map
                        current_metrics['correlation_map'] = np.full(target_spatial_shape, np.nan)
            else:
                 log.warning(f"分割 '{split}': Mask 中没有有效点，指标将为 NaN。")
                 current_metrics = {k: np.nan for k in ["mean_rmse", "mean_mae", "max_rmse", "min_rmse"]}
                 current_metrics["rmse_map"] = rmse_field
                 if cfg.evaluation.get("calculate_correlation", False): current_metrics['mean_correlation'] = np.nan

            # --- 5.5 保存指标 ---
            log.info(f"保存评估指标 ({split})...")
            metrics_to_save_converted = {}
            for key, value in current_metrics.items():
                 if key not in ["rmse_map", "correlation_map"]:
                      if isinstance(value, (np.float16, np.float32, np.float64, np.floating)): 
                          metrics_to_save_converted[key] = float(value)
                      elif isinstance(value, (np.int8, np.int16, np.int32, np.int64, np.integer)): 
                          metrics_to_save_converted[key] = int(value)
                      elif isinstance(value, (float, int)) or value is None: 
                          metrics_to_save_converted[key] = value
                      elif np.isnan(value): 
                          metrics_to_save_converted[key] = None
                      else: 
                          log.warning(f"无法识别的度量类型 ({key}): {type(value)}")

            metrics_file = os.path.join(results_dir, f"metrics_{split}.yaml") # 文件名包含 split
            try:
                OmegaConf.save(config=OmegaConf.create(metrics_to_save_converted), f=metrics_file)
                log.info(f"评估指标 ({split}) 已保存至: {metrics_file}")
            except Exception as e_save:
                log.error(f"保存指标文件失败 ({split}): {e_save}", exc_info=True)

            # 保存空间指标图 npy 文件
            if "rmse_map" in current_metrics and np.any(np.isfinite(current_metrics["rmse_map"])):
                 rmse_map_file = os.path.join(results_dir, f"rmse_map_{split}.npy") # 文件名包含 split
                 np.save(rmse_map_file, current_metrics["rmse_map"])
                 log.info(f"空间 RMSE 图 ({split}) 已保存至: {rmse_map_file}")
            if "correlation_map" in current_metrics and np.any(np.isfinite(current_metrics["correlation_map"])):
                 corr_map_file = os.path.join(results_dir, f"correlation_map_{split}.npy") # 文件名包含 split
                 np.save(corr_map_file, current_metrics["correlation_map"])
                 log.info(f"空间相关性图 ({split}) 已保存至: {corr_map_file}")

            split_metrics = metrics_to_save_converted # 存储转换后的指标

            # --- 5.6 可视化 ---
            log.info(f"生成可视化图表 ({split})...")
            if cfg.visualization.enabled and num_valid_points_mask > 0:
                dr_method = cfg.model.dimensionality_reduction.method
                pred_method = cfg.model.prediction.method
                title_suffix = f"({split.capitalize()})"

                try:
                    # 时间序列对比图 (空间平均)
                    rec_mean_ts = np.nanmean(recon_valid_flat, axis=1)
                    orig_mean_ts = np.nanmean(orig_valid_flat, axis=1)
                    ts_save_path = os.path.join(plots_dir, f"time_series_comparison_{split}.png") # 文件名包含 split
                    eval_viz.plot_comparison_and_error(
                        original=orig_mean_ts, reconstructed=rec_mean_ts, config=cfg,
                        fig_title=f"时间序列对比与误差分析 (Time Series Comparison and Error Analysis)", save_path=ts_save_path
                    )
                    figure_paths_split.append(ts_save_path)

                    # 空间 RMSE 图
                    if "rmse_map" in current_metrics and np.any(np.isfinite(current_metrics["rmse_map"])):
                         spatial_rmse_save_path = os.path.join(plots_dir, f"spatial_rmse_{split}.png") # 文件名包含 split
                         eval_viz.plot_spatial_rmse(
                              current_metrics["rmse_map"], cfg=cfg, mask=boolean_mask,
                              title=f"空间RMSE分布", save_path=spatial_rmse_save_path
                         )
                         figure_paths_split.append(spatial_rmse_save_path)

                    # (可选) 空间相关性图
                    if cfg.evaluation.get("calculate_correlation", True) and "correlation_map" in current_metrics :
                         if cfg.evaluation.visualization.get("plot_spatial_correlation", True):
                              corr_save_path = os.path.join(plots_dir, f"spatial_correlation_{split}.png") # 文件名包含 split
                              eval_viz.plot_spatial_statistic(
                                   current_metrics['correlation_map'], cfg=cfg, mask=~boolean_mask, save_path=corr_save_path,
                                   title=f"相关性分步",
                                   cmap='coolwarm', vmin=-1, vmax=1, cbar_label="相关性系数"
                              )
                              figure_paths_split.append(corr_save_path)

                    # (可选) 瞬时对比图
                    if cfg.evaluation.get("plot_instantaneous", True):
                        time_indices_to_plot = cfg.evaluation.visualization.get("time_indices_to_plot", [0, n_samples // 2, n_samples - 1])
                        log.info(f"  绘制特定时间点的空间图 ({split}, indices={time_indices_to_plot})...")
                        for t_idx in time_indices_to_plot:
                            if 0 <= t_idx < n_samples:
                                 try:
                                      comp_save_path = os.path.join(plots_dir, f"spatial_comparison_{split}_t{t_idx}.png") # 文件名包含 split 和 t_idx
                                      eval_viz.plot_spatial_comparison_at_timestep(
                                          np.squeeze(orig_spatial_masked[t_idx]), 
                                          recon_spatial[t_idx], 
                                          diff_spatial[t_idx], 
                                          cfg=cfg, 
                                          mask=~boolean_mask, 
                                          time_index=t_idx, 
                                          save_path=comp_save_path,
                                          title_prefix=f"盐度场抽样对比"
                                      )
                                      figure_paths_split.append(comp_save_path)
                                      if cfg.evaluation.visualization.get("plot_instantaneous_difference_only", True):
                                           diff_save_path = os.path.join(plots_dir, f"spatial_difference_{split}_t{t_idx}.png") # 文件名包含 split 和 t_idx
                                           eval_viz.plot_spatial_difference(
                                               diff_spatial[t_idx], 
                                               cfg=cfg, 
                                               mask=~boolean_mask, 
                                               time_index=t_idx, 
                                               save_path=diff_save_path,
                                               title=f"盐度场空间误差"
                                           )
                                           figure_paths_split.append(diff_save_path)
                                 except Exception as e_inst: log.error(f"  绘制瞬时图失败 (t={t_idx}, split={split}): {e_inst}", exc_info=True)

                except ImportError as viz_imp_err:
                     log.error(f"可视化导入错误 ({split}): {viz_imp_err}")
                     # 不标记 split 失败
                except Exception as viz_e:
                     log.error(f"可视化过程中捕获到异常 ({split})! 类型: {type(viz_e).__name__}, 消息: {viz_e}", exc_info=True)
                     # 不标记 split 失败

            elif not cfg.visualization.enabled:
                 log.info(f"可视化在配置中被禁用 ({split})。")
            elif num_valid_points_mask == 0:
                 log.info(f"没有有效数据点，跳过可视化 ({split})。")


            # --- 完成当前分割评估 ---
            # 记录当前分割的结果
            results['evaluation_details'][split] = {'metrics': split_metrics, 'figures': figure_paths_split}
            split_success = True # 只要没在上面因为错误 return，就认为这个 split 成功了
            evaluation_successful_for_any_split = True # 标记至少有一个分割成功
            log.info(f"--- 完成评估 {split} ---")

        except Exception as e_std_eval:
            log.error(f"标准评估分割 {split} 出错: {e_std_eval}", exc_info=True)
            if split not in results['evaluation_details']: results['evaluation_details'][split] = {}
            results['evaluation_details'][split]['standard_evaluation_status'] = f'Error: {e_std_eval}'


    # --- (最小改动) Iterative Prediction Evaluation (only for test split) ---
    log.info("--- (评估子步骤 5b: 最小改动多步迭代预测评估 - 仅 Test 集) ---")
    if cfg.evaluation.get("run_iterative_forecast", True):
        try:
            # 调用新的、加载依赖的函数
            iter_rmse_steps = aelstm_multi_step_forecast_rmse_curve_minimal(
                cfg=cfg,
                ae_results=ae_results,
                pred_results=pred_results,
                all_processed_data=all_processed_data,
                mask=mask, # Pass the already loaded mask
                device=device # Pass the global device
            )

            # --- 保存和绘图 ---
            if not np.all(np.isnan(iter_rmse_steps)): # 检查是否全为 NaN
                iter_results_dir = os.path.join(results_dir, "iterative_forecast")
                iter_plots_dir = os.path.join(plots_dir, "iterative_forecast")
                os.makedirs(iter_results_dir, exist_ok=True)
                os.makedirs(iter_plots_dir, exist_ok=True)

                rmse_curve_path = os.path.join(iter_results_dir, "ae_lstm_iterative_rmse_curve.npy")
                np.save(rmse_curve_path, iter_rmse_steps)
                log.info(f"AE-LSTM 迭代预测 RMSE 曲线数据已保存: {rmse_curve_path}")

                # Plotting (与之前的版本相同)
                plot_save_path = os.path.join(iter_plots_dir, "ae_lstm_iterative_rmse_curve_comparison.png")
                plt.figure(figsize=(10, 6))
                # ... (设置中文字体的 try-except block) ...
                try:
                    available_font_name = 'Microsoft YaHei' # Windows 示例 (需要替换!)
                    plt.rcParams['font.sans-serif'] = [available_font_name]
                    plt.rcParams['axes.unicode_minus'] = False
                except Exception as font_e:
                    log.warning(f"设置中文字体失败: {font_e}")

                valid_rmse_steps = iter_rmse_steps[~np.isnan(iter_rmse_steps)]
                if len(valid_rmse_steps) > 0:
                    x_steps = np.arange(1, len(valid_rmse_steps) + 1)
                    plt.plot(x_steps, valid_rmse_steps, marker='s', color='purple', label='AE-LSTM Iterative RMSE')
                    plt.xlim(left=0.8, right=len(valid_rmse_steps) + 0.2)
                # ... (绘制对比基线 plt.axhline, plt.plot for SOM-HMM, SOM-LSTM, PCA-LSTM step 1) ...
                som_hmm_rmse = cfg.evaluation.get("comparison_rmse_som_hmm", 3.0)
                plt.axhline(y=som_hmm_rmse, color='red', linestyle='--', linewidth=1.5, label=f'SOM-HMM RMSE = {som_hmm_rmse:.2f}')
                som_lstm_step1_rmse = cfg.evaluation.get("comparison_rmse_som_lstm_step1", 2.45)
                plt.plot(1, som_lstm_step1_rmse, marker='^', markersize=10, linestyle='none', color='green', label=f'SOM-LSTM RMSE (Step 1) = {som_lstm_step1_rmse:.2f}')
                pca_lstm_step1_rmse = cfg.evaluation.get("comparison_rmse_pca_lstm_step1", 1.68)
                plt.plot(1, pca_lstm_step1_rmse, marker='o', markersize=8, linestyle='none', color='blue', label=f'PCA-LSTM RMSE (Step 1) = {pca_lstm_step1_rmse:.2f}')


                plt.xlabel("预测步长 (天)")
                plt.ylabel("RMSE (PSU)")
                plt.title("迭代预测 RMSE 对比 (Test Set)")
                plt.grid(True, alpha=0.5)
                plt.legend()
                plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
                plt.close()
                log.info(f"已生成测试集迭代预测 RMSE 对比曲线: {plot_save_path}")

                # 存储结果
                test_split_name = 'test'
                if test_split_name not in results['evaluation_details']: results['evaluation_details'][test_split_name] = {}
                results['evaluation_details'][test_split_name]['iterative_rmse_curve'] = iter_rmse_steps.tolist() # Save as list
                results['evaluation_details'][test_split_name]['iterative_figures'] = [plot_save_path]
                iterative_eval_success = True # 标记迭代评估成功
            else:
                log.warning("迭代预测评估计算得到的 RMSE 均为 NaN，跳过保存和绘图。")

        except Exception as e_iter:
            log.error(f"迭代预测评估失败: {e_iter}", exc_info=True)
            # 记录错误到 results['evaluation_details']
            test_split_name = 'test'
            if test_split_name not in results['evaluation_details']: results['evaluation_details'][test_split_name] = {}
            results['evaluation_details'][test_split_name]['iterative_evaluation_error'] = str(e_iter)

    else:
        log.info("根据配置跳过迭代预测评估。")

    # --- Final Success Status ---
    results['success'] = evaluation_successful_for_any_split or iterative_eval_success
    log.info(f"--- 评估步骤完成 (最终状态: {'成功' if results['success'] else '失败'}) ---")
    return results

# === 主 Pipeline 函数 ===
@hydra.main(config_path="../conf", config_name="config", version_base=None) # 假设 conf 在上一级目录
def run_pipeline(cfg: DictConfig) -> None:
    """主函数"""
    log.info("=== 开始 AE(盐度) + LSTM(盐度+预计算风场PCA+径流) Pipeline ===")
    log.info(f"Pipeline 名称: {cfg.get('pipeline_name', 'N/A')}")
    log.info(f"输出目录 (Hydra): {os.getcwd()}")

    # --- 加载初始 *预处理* 和 *预计算* 数据 ---
    log.info("加载预处理的盐度和预计算的低维风场数据...")
    all_processed_data = {} # 初始化为空字典
    try:
        # --- 直接加载数据 ---
        log.info("直接根据配置路径加载 .npy 文件...")
        # 检查配置中是否有 processed_paths
        if not cfg.paths.get('processed_paths'):
             raise ValueError("配置中缺少 'paths.processed_paths' 部分。")

        # 加载盐度数据
        if cfg.paths.processed_paths.get('salinity'):
            salinity_data = {}
            for split, path in cfg.paths.processed_paths.salinity.items():
                 log.debug(f"  加载盐度 {split} 从: {path}")
                 if not os.path.exists(path): raise FileNotFoundError(f"文件未找到: {path}")
                 data_array = np.load(path)
                 # 可选的维度处理
                 if data_array.ndim == 1: data_array = data_array.reshape(-1, 1)
                 elif data_array.ndim > 2: data_array = data_array.reshape(data_array.shape[0], -1)
                 salinity_data[split] = data_array
                 log.debug(f"    加载形状: {data_array.shape}")
            all_processed_data['salinity'] = salinity_data
            log.info("盐度数据加载完成。")
        else:
            log.error("配置中缺少 'paths.processed_paths.salinity'。")
            return # 或者抛出错误

        # 加载预计算的风场 PCA 数据
        if cfg.model.prediction.input_sources.use_wind:
            wind_pca_data = {}
            for split, path in cfg.paths.processed_paths.wind_pca.items():
                 log.debug(f"  加载风场 PCA {split} 从: {path}")
                 if not os.path.exists(path): raise FileNotFoundError(f"文件未找到: {path}")
                 data_array = np.load(path)
                 # 可选的维度处理
                 if data_array.ndim == 1: data_array = data_array.reshape(-1, 1)
                 # PCA结果通常已经是 (n_samples, n_components)
                 wind_pca_data[split] = data_array
                 log.debug(f"    加载形状: {data_array.shape}")
            all_processed_data['wind_pca'] = wind_pca_data # 注意键名是 wind_pca
            log.info("预计算的风场 PCA 数据加载完成。")
        else:
            # 如果不需要风场，可以继续；如果需要，则报错
            log.warning("配置中缺少 'paths.processed_paths.wind_pca'。LSTM 将只使用盐度信息。")
            # 如果风场是必需的，应该报错:
            # log.error("配置中缺少 'paths.processed_paths.wind_pca'，无法继续。")
            # return

                # 加载预计算的径流数据 (可选)
        if cfg.model.prediction.input_sources.use_flow:
            flow_data = {}
            for split, path in cfg.paths.processed_paths.flow.items():
                 log.debug(f"  加载径流数据 {split} 从: {path}")
                 if not os.path.exists(path): raise FileNotFoundError(f"文件未找到: {path}")
                 data_array = np.load(path)
                 # 维度处理
                 if data_array.ndim == 1: data_array = data_array.reshape(-1, 1)
                 flow_data[split] = data_array
                 log.debug(f"    加载形状: {data_array.shape}")
            all_processed_data['flow'] = flow_data
            log.info("预计算的径流数据加载完成。")
        else:
            log.info("配置中未指定径流数据路径，将不使用径流数据。")

    except FileNotFoundError as fnf_err:
        log.error(f"加载初始数据失败: {fnf_err}", exc_info=True)
        return
    except Exception as load_err:
        log.error(f"加载初始数据时发生其他错误: {load_err}", exc_info=True)
        return

    # 检查是否成功加载了必要的数据
    if 'salinity' not in all_processed_data:
        log.error("未能加载必要的盐度数据。正在退出。")
        return
    # 如果风场是必需的，也检查 'wind_pca'
    # if 'wind_pca' not in all_processed_data:
    #     log.error("未能加载必要的风场 PCA 数据。正在退出。")
    #     return

    # --- 执行 Pipeline 步骤 ---
    ae_results = run_ae_on_salinity(cfg, all_processed_data)
    wind_results = run_wind_processing(cfg) # 只需配置，不需原始数据
    flow_results = None
    if cfg.model.prediction.input_sources.get("use_flow", False):
        flow_results = run_flow_processing(cfg)
        log.info(f"径流数据处理{'成功' if flow_results.get('success') else '失败'}")
        
    pred_results = run_lstm_prediction(cfg, ae_results, wind_results, flow_results)
    rec_results = run_salinity_reconstruction(cfg, ae_results, pred_results)
    eval_results = run_evaluation(cfg, rec_results, all_processed_data, ae_results, pred_results) # 传入 all_processed_data 和 ae_results

    # --- 完成 ---
    success = (ae_results.get('success', False) and
               wind_results.get('success', False) and
               pred_results.get('success', False) and
               rec_results.get('success', False) and
               eval_results.get('success', False))
    if success: log.info("=== Pipeline 成功完成 ===")
    else: log.error("=== Pipeline 完成但出现错误 ===")


if __name__ == "__main__":
    run_pipeline()