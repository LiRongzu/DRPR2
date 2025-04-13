# src/training/train_hmm.py

import os
import sys
import numpy as np
import logging
import time
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Optional, Tuple, Dict, Any, Callable
import joblib 
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
import re

# --- 项目设置 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.hydra_config import DrprConfig
from src.utils.logger import setup_logger
from src.utils.model_utils import get_device_from_config
from src.dimensionality_reduction.som_pytorch import SOMTorch # 导入 SOMTorch

# ----------------------------------------------- #
# 从 utility 模块导入所需函数
from src.utils.data_loader import load_bmu_positions
# 更新导入语句，使用更细粒度的函数
from src.utils.bmu_utils import calculate_l2_norm, create_ranking_map
from src.utils.bmu_utils import convert_grid_to_linear, convert_linear_to_rank
# ----------------------------------------------- #

logger = logging.getLogger(__name__)

def estimate_hmm_params(state_seq: np.ndarray, obs_seq: np.ndarray, n_states: int, n_obs: int, smoothing_eps: float = 1e-7) -> Dict[str, np.ndarray]:
    """估计HMM参数"""
    # ... (检查维度和长度的代码) ...
    n_samples = len(state_seq)

    # --- 估计初始概率 (startprob) ---
    startprob = np.zeros(n_states)
    counts = np.bincount(state_seq, minlength=n_states)
    startprob = counts / n_samples if n_samples > 0 else np.ones(n_states) / n_states
    startprob += smoothing_eps
    startprob /= np.sum(startprob)

    # --- 估计转移矩阵 (transmat) ---
    transmat = np.zeros((n_states, n_states))
    for i in range(n_samples - 1):
        transmat[state_seq[i], state_seq[i+1]] += 1
    transmat += smoothing_eps
    sum_transmat = np.sum(transmat, axis=1, keepdims=True) # Shape (n_states, 1)

    # 处理从未作为起始状态的状态
    # **** 修改开始 ****
    zero_rows_mask_1d = (sum_transmat < (smoothing_eps * n_states / 2)).flatten() # 获取 1D 布尔掩码
    num_zero_rows = np.sum(zero_rows_mask_1d)

    if num_zero_rows > 0:
        logger.info(f"处理转移矩阵中 {num_zero_rows} 个零和行...")
        # 1. 对零和行的所有元素赋 epsilon
        transmat[zero_rows_mask_1d, :] = smoothing_eps
        # 2. 重新计算这些行的和 (结果是一维数组)
        recalculated_sums_1d = np.sum(transmat[zero_rows_mask_1d, :], axis=1) # 不用 keepdims=True
        # 3. 将一维的和赋值给 sum_transmat 对应行的第一列
        sum_transmat[zero_rows_mask_1d, 0] = recalculated_sums_1d
    # **** 修改结束 ****

    # 避免除以零（如果某行即使在处理后仍然是零，例如只有一个状态）
    sum_transmat[sum_transmat == 0] = 1.0
    transmat = transmat / sum_transmat # 现在可以安全地进行归一化

    # --- 估计发射矩阵 (emissionprob) ---
    emissionprob = np.zeros((n_states, n_obs))
    for t in range(n_samples):
        state = state_seq[t]
        obs = obs_seq[t]
        if 0 <= obs < n_obs:
            emissionprob[state, obs] += 1
        else:
             logger.warning(f"时间步 {t}: 观测等级 {obs} 超出预期范围 [0, {n_obs-1}]，跳过。")
    emissionprob += smoothing_eps
    sum_emissionprob = np.sum(emissionprob, axis=1, keepdims=True) # Shape (n_states, 1)

    # 处理从未发射任何观测的状态
    # **** 修改开始 ****
    zero_emission_rows_mask_1d = (sum_emissionprob < (smoothing_eps * n_obs / 2)).flatten() # 获取 1D 布尔掩码
    num_zero_emission_rows = np.sum(zero_emission_rows_mask_1d)

    if num_zero_emission_rows > 0:
        logger.info(f"处理发射矩阵中 {num_zero_emission_rows} 个零和行...")
        # 1. 对零和行的所有元素赋 epsilon
        emissionprob[zero_emission_rows_mask_1d, :] = smoothing_eps
        # 2. 重新计算这些行的和 (结果是一维数组)
        recalculated_emission_sums_1d = np.sum(emissionprob[zero_emission_rows_mask_1d, :], axis=1) # 不用 keepdims=True
        # 3. 将一维的和赋值给 sum_emissionprob 对应行的第一列
        sum_emissionprob[zero_emission_rows_mask_1d, 0] = recalculated_emission_sums_1d
    # **** 修改结束 ****

    # 避免除以零
    sum_emissionprob[sum_emissionprob == 0] = 1.0
    emissionprob = emissionprob / sum_emissionprob # 现在可以安全地进行归一化

    logger.info("通过直接计数（最大似然估计 + 平滑）估计HMM参数。")
    return {
        'startprob_': startprob,
        'transmat_': transmat,
        'emissionprob_': emissionprob
    }

def extract_feature_name_from_path(file_path: str) -> str:
    """从文件路径中提取特征名称"""
    # 尝试匹配 bmu_positions_{特征名}_{分割}.npy 模式
    # 允许特征名包含下划线，例如 wind_flow
    match = re.search(r'bmu_positions_([^_]+(?:_[^_]+)*)_(train|val|test)\.npy', os.path.basename(file_path))
    if match:
        # 提取特征名部分
        return match.group(1)
    # 如果上述模式不匹配（例如不同的命名约定），提供回退或警告
    logger.warning(f"无法使用标准模式从路径提取特征名称: {file_path}")
    return "unknown"

# --- Modified main function ---
def main(
    cfg: DictConfig,
    state_som_model_path: str, # 状态 SOM 模型路径
    obs_som_model_path: str,   # 观测 SOM 模型路径
    param_save_path: str,      # HMM 参数保存路径
    # 新增：预测结果保存路径模板
    predicted_state_save_pattern: str = "predicted_salinity_states_{split}.npy"

) -> Dict[str, Any]:
    """
    训练 HMM (仅用 train 数据)，并预测 train, val, test 的状态序列。
    """
    start_run_time = time.time()
    config = DrprConfig.from_hydra_config(cfg)
    device = get_device_from_config(cfg)
    results = {} # 用于存储返回信息

    logger.info("开始 HMM 训练 (仅用 train) 和预测 (train, val, test) 流程...")
    logger.info(f"参数保存路径: {param_save_path}")

    os.makedirs(os.path.dirname(param_save_path), exist_ok=True)
    # 确保预测结果保存目录存在 (假设它们都在同一个目录下)
    prediction_base_dir = os.path.dirname(os.path.join(config.paths.predictions_base_dir, predicted_state_save_pattern)) # 获取基础目录
    os.makedirs(prediction_base_dir, exist_ok=True)

    # --- 0. 加载 SOM 模型和创建 Ranking Maps ---
    state_som_model, obs_som_model = None, None
    state_rank_map, observation_rank_map = None, None
    state_feature_name = "salinity" # 通常是固定的
    obs_feature_name = "_".join(sorted(list(cfg.model.prediction.hmm.observation_features))) # 从配置获取观测特征名

    try:
        logger.info(f"加载状态 SOM 模型从: {state_som_model_path}")
        state_som_model = SOMTorch.load(state_som_model_path, device=device)
        logger.info(f"加载观测 SOM 模型从: {obs_som_model_path}")
        obs_som_model = SOMTorch.load(obs_som_model_path, device=device)

        logger.info("根据原型 L2 范数创建等级映射...")
        state_rank_map = create_ranking_map(state_som_model, calculate_l2_norm)
        observation_rank_map = create_ranking_map(obs_som_model, calculate_l2_norm)

    except Exception as e:
        logger.error(f"加载 SOM 模型或创建等级映射时失败: {e}", exc_info=True)
        return results # 返回空字典表示失败

    # --- 1. 加载 Train BMU 数据并转换为 Rank ---
    logger.info("加载 Train BMU 数据并应用等级映射...")
    state_bmu_train_raw = None
    obs_bmu_train_raw = None
    try:
        # 使用 data_loader 加载 BMU 位置数据
        state_bmu_train_raw = load_bmu_positions(cfg, state_feature_name, "train")
        obs_bmu_train_raw = load_bmu_positions(cfg, obs_feature_name, "train")
    except FileNotFoundError as e:
         logger.error(f"加载训练 BMU 数据失败: {e}。无法训练 HMM。")
         return results
    except Exception as e:
         logger.error(f"加载训练 BMU 数据时发生未知错误: {e}")
         return results

    if state_bmu_train_raw is None or obs_bmu_train_raw is None:
        logger.error("加载训练 BMU 数据失败，无法训练 HMM。")
        return results

    map_height_state, map_width_state = tuple(state_som_model.map_size)
    map_height_obs, map_width_obs = tuple(obs_som_model.map_size)

    # BMU 转换: 网格索引(row,col) -> 线性索引 -> 等级索引
    try:
        # 1. 将 BMU 位置（网格索引）转换为线性索引
        state_linear_indices = convert_grid_to_linear(state_bmu_train_raw, map_width_state)
        obs_linear_indices = convert_grid_to_linear(obs_bmu_train_raw, map_width_obs)
        
        # 2. 将线性索引转换为等级索引
        S_train = convert_linear_to_rank(state_linear_indices, state_rank_map)
        O_train = convert_linear_to_rank(obs_linear_indices, observation_rank_map)
        
        logger.info(f"成功将 BMU 位置转换为等级索引。状态: {S_train.shape}, 观测: {O_train.shape}")
    except Exception as e:
        logger.error(f"转换 BMU 到等级索引失败: {e}")
        return results

    if S_train is None or O_train is None:
        logger.error("转换训练 BMU 到 Rank 失败。")
        return results

    n_states = len(state_rank_map)
    n_obs = len(observation_rank_map)
    logger.info(f"状态 BMU (Train, Ranked) 形状: {S_train.shape}, 状态数: {n_states}")
    logger.info(f"观测 BMU (Train, Ranked) 形状: {O_train.shape}, 观测数: {n_obs}")

    # --- 2. 训练 HMM 模型 (仅用 Train 数据) ---
    logger.info("开始训练 HMM 模型 (仅用 Train 数据)...")
    smoothing_epsilon = cfg.model.prediction.hmm.get("smoothing_epsilon", 1e-7)
    # 确保 S_train 和 O_train 长度一致
    min_len_train = min(len(S_train), len(O_train))
    if min_len_train <= 0:
         logger.error("训练数据有效长度为 0。")
         return results
    S_train = S_train[:min_len_train]
    O_train = O_train[:min_len_train]

    estimated_params = estimate_hmm_params(
        S_train, O_train, n_states, n_obs, smoothing_eps=smoothing_epsilon
    )

    model = hmm.CategoricalHMM(n_components=n_states, init_params="", params="")
    try:
        model.startprob_ = estimated_params['startprob_']
        model.transmat_ = estimated_params['transmat_']
        model.emissionprob_ = estimated_params['emissionprob_']
        model.n_features = n_obs # hmmlearn 需要知道观测的数量
        model._check() # 验证参数
        logger.info("HMM 模型参数设置完成。")
    except Exception as e:
         logger.error(f"设置 HMM 参数时出错: {e}", exc_info=True)
         return results

    # --- 3. 保存 HMM 参数 ---
    model_params = {
        'hmm_params': estimated_params,
        'input_desc': obs_feature_name,
        'n_states': n_states,
        'n_obs': n_obs,
        'bmu_representation': 'ranked_by_L2_norm',
        'state_rank_map': state_rank_map,
        'state_linear_map': {rank: linear_idx for linear_idx, rank in state_rank_map.items()},
        'map_size': list(state_som_model.map_size),
        # 注意：不再需要保存 train/val/test indices，因为训练和预测分开了
    }
    try:
        joblib.dump(model_params, param_save_path)
        logger.info(f"HMM 参数已保存到: {param_save_path}")
        results['model_path'] = param_save_path
    except Exception as e:
        logger.error(f"保存 HMM 参数失败: {e}", exc_info=True)
        # 保存失败也继续尝试预测

    # --- 4. 预测状态 (Train, Val, Test) ---
    logger.info("开始预测状态序列 (Train, Val, Test)...")
    results['predicted_states_paths'] = {} # 存储预测结果路径

    for split in ["train", "val", "test"]:
        logger.info(f"  处理分割: {split}...")
        obs_bmu_raw = None
        try:
            # ----------------------------------------------- #
            # 使用新的加载函数
            obs_bmu_raw = load_bmu_positions(cfg, obs_feature_name, split)
            # ----------------------------------------------- #
        except FileNotFoundError:
             logger.warning(f"  跳过 {split} 分割，因为观测 BMU 文件未找到。")
             continue
        except Exception as e:
             logger.error(f"  加载 {split} 观测 BMU 时发生未知错误: {e}")
             continue # 跳过此分割

        if obs_bmu_raw is None:
            logger.warning(f"  跳过 {split} 分割，因为观测 BMU 文件未找到。")
            continue

        try:
            # 1. 将 BMU 位置（网格索引）转换为线性索引
            obs_linear_indices = convert_grid_to_linear(obs_bmu_raw, map_width_obs)
            
            # 2. 将线性索引转换为等级索引
            O_split = convert_linear_to_rank(obs_linear_indices, observation_rank_map)
            
            if O_split is None:
                logger.warning(f"  跳过 {split} 分割，因为 BMU 转换为 Rank 失败。")
                continue
            if len(O_split) == 0:
                logger.warning(f"  跳过 {split} 分割，因为转换后的 Rank 序列为空。")
                continue
        except Exception as e:
            logger.error(f"  转换 {split} 分割的 BMU 到等级索引失败: {e}")
            continue

        try:
            predicted_states_split = model.predict(O_split.reshape(-1, 1))
            logger.info(f"  {split} 状态预测完成，形状: {predicted_states_split.shape}")

            # 保存预测结果
            pred_save_path = os.path.join(prediction_base_dir, predicted_state_save_pattern.format(split=split))
            np.save(pred_save_path, predicted_states_split)
            logger.info(f"  预测的 {split} 状态序列已保存到: {pred_save_path}")
            results['predicted_states_paths'][split] = pred_save_path

        except ValueError as ve: # 捕捉 hmmlearn 可能因输入问题抛出的 ValueError
             if "Expected 2D array, got 1D array instead" in str(ve):
                 logger.error(f"  HMM 预测失败 ({split}): 输入 O_split.reshape(-1, 1) 可能有问题。O_split shape: {O_split.shape}")
             elif "Unexpected observations tensor shape" in str(ve):
                 logger.error(f"  HMM 预测失败 ({split}): 观测值可能超出模型预期范围 [0, {n_obs-1}]。Max observed rank: {np.max(O_split) if len(O_split)>0 else 'N/A'}")
             else:
                 logger.error(f"  HMM 预测失败 ({split}): {ve}")
        except Exception as e:
            logger.error(f"  预测 {split} 状态序列失败: {e}", exc_info=True)

    total_run_time = time.time() - start_run_time
    logger.info(f"HMM 训练 (仅用 train) 和预测 (train, val, test) 完成，总用时：{total_run_time:.2f}秒")

    return results

# --- 更新 hydra_main 以不传递 bmu 目录 ---
@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def hydra_main(cfg: DictConfig) -> Dict[str, Any]:
    config = DrprConfig.from_hydra_config(cfg)
    
    # 构建 SOM 模型路径
    state_som_model_path = os.path.join(config.paths.som_models_dir, "som_model_salinity.pt")

    # 构建观测 SOM 模型路径
    try:
         observation_features = list(cfg.model.prediction.hmm.observation_features)
         obs_feature_name = "_".join(sorted(observation_features))
    except Exception as e:
         logger.error(f"无法从配置确定观测特征名称: {e}")
         obs_feature_name = "unknown"
    obs_som_model_path = os.path.join(config.paths.som_models_dir, f"som_model_{obs_feature_name}.pt")

    # 构建 HMM 参数保存路径
    hmm_param_filename = f"hmm_params_obs_{obs_feature_name}.pkl"
    param_save_path = os.path.join(config.paths.hmm_params_dir, hmm_param_filename) # 使用带特征名的路径

    # 构建预测状态保存路径模式
    predicted_state_save_pattern = "predicted_salinity_states_{split}.npy"

    # 检查必要的模型文件是否存在
    missing = []
    required_models = [state_som_model_path, obs_som_model_path]
    for f_path in required_models:
        if not os.path.exists(f_path):
            missing.append(f_path)
    if missing:
        logger.error("缺少运行 HMM 所需的 SOM 模型文件:")
        for f in missing: logger.error(f"  - {f}")
        return {}

    # 调用更新后的 main 函数，不再传递 bmu 目录
    return main(
        cfg,
        state_som_model_path=state_som_model_path,
        obs_som_model_path=obs_som_model_path,
        param_save_path=param_save_path,
        predicted_state_save_pattern=predicted_state_save_pattern
    )

if __name__ == "__main__":
    hydra_main()