import numpy as np
import pandas as pd # Keep import if other functions might use it elsewhere
import os
import gc
import warnings # To warn about small std dev
import logging # 确保导入 logging
# Assuming these are defined elsewhere:
from .data_validation import load_salinity_data, load_wind_data, load_river_flow_data
logger = logging.getLogger(__name__) 

def preprocess_salinity_data(salinity_data, normalize=True, is_training=True, scaler=None, epsilon=1e-8):
    """
    预处理3D盐度场数据 (时间 x 高 x 宽)，处理NaN值并进行标准化。
    *** 修改后：执行按维度（每个有效格点独立）的标准化 ***

    参数:
        salinity_data (numpy.ndarray): 原始3D盐度场数据 (时间步数, N, M)。
        normalize (bool, 默认为True): 是否进行标准化处理。
        is_training (bool, 默认为True): 是否为训练数据。
        scaler (dict, 可选): 用于测试/验证集的标准化器，包含 'mean' 和 'std' 向量。
        epsilon (float, 默认为1e-8): 防止除以零的小值。

    返回:
        tuple:
            - processed_data (numpy.ndarray): 预处理后的2D盐度数据 (时间步数, 有效格点数)。
            - scaler (dict 或 None): 数据缩放器 {'mean': mean_vector, 'std': std_vector}。
            - valid_mask_2d (numpy.ndarray): 2D布尔掩码 (N, M)，True表示有效格点。
    """
    # --- 输入验证 (保持不变) ---
    if not isinstance(salinity_data, np.ndarray):
        raise TypeError(f"输入 salinity_data 必须是 NumPy 数组, 得到 {type(salinity_data)}")
    if salinity_data.ndim != 3:
        raise ValueError(f"输入 salinity_data 必须是 3D 数组 (时间 x 高 x 宽), 得到维度 {salinity_data.ndim}")

    T, N, M = salinity_data.shape
    original_grid_points = N * M
    logger.info(f"输入盐度数据形状: ({T}, {N}, {M})") # 使用 logger

    # --- 识别有效格点 (保持不变) ---
    valid_mask_2d = ~np.isnan(salinity_data).any(axis=0)
    num_valid_points = np.sum(valid_mask_2d)

    if num_valid_points == 0:
        raise ValueError("数据错误：所有栅格点在所有时间步中都包含NaN值。无法处理。")
    logger.info(f"有效格点数: {num_valid_points} / {original_grid_points}") # 使用 logger

    # --- 提取有效数据并转换为2D (保持不变) ---
    if num_valid_points == original_grid_points:
        valid_data = salinity_data.reshape(T, -1)
    else:
        valid_data = salinity_data[:, valid_mask_2d] # 形状 (T, num_valid_points)
    logger.info(f"提取的有效数据形状: {valid_data.shape}") # 使用 logger

    # --- 数据标准化 (修改部分) ---
    processed_data = valid_data # 默认值

    if normalize:
        current_op = "训练集" if is_training else "验证/测试集"
        logger.info(f"对有效盐度数据执行按维度标准化 ({current_op})...") # 使用 logger

        if is_training:
            # --- 计算训练集每个有效格点的均值和标准差 (沿时间轴 axis=0) ---
            mean_vals = np.mean(valid_data, axis=0) # 形状 (num_valid_points,)
            std_vals = np.std(valid_data, axis=0)   # 形状 (num_valid_points,)

            # 检查标准差向量中是否有过小的值
            if np.any(std_vals < epsilon):
                num_small_std = np.sum(std_vals < epsilon)
                warnings.warn(
                    f"训练集计算的标准差向量中包含 {num_small_std} 个小于 epsilon ({epsilon:.1e}) 的值。"
                    f"将在分母加上 epsilon 以避免除零错误。", UserWarning
                )
                # 可以选择将这些维度的 std 设为 epsilon 或 1，以避免产生过大值
                # std_vals[std_vals < epsilon] = epsilon # 例如，设为 epsilon

            # 保存按维度的 scaler
            scaler = {'mean': mean_vals, 'std': std_vals}
            logger.info(f"  计算得到按维度的均值和标准差向量，形状均为: {mean_vals.shape}") # 使用 logger

            # --- 应用按维度标准化 (利用 NumPy 广播机制) ---
            processed_data = (valid_data - mean_vals) / (std_vals + epsilon)

        else: # is_training = False (验证或测试)
            if scaler is None:
                raise ValueError("使用测试/验证模式且normalize=True时必须提供scaler")
            if 'mean' not in scaler or 'std' not in scaler:
                 raise ValueError("提供的scaler缺少 'mean' 或 'std' 键")

            mean_vals = scaler['mean'] # 应该是向量
            std_vals = scaler['std']   # 应该是向量

            # --- 检查 scaler 和数据的维度是否匹配 ---
            if not isinstance(mean_vals, np.ndarray) or mean_vals.ndim != 1 or \
               not isinstance(std_vals, np.ndarray) or std_vals.ndim != 1:
                 raise TypeError("Scaler 中的 'mean' 和 'std' 必须是 NumPy 向量 (1D array)。")

            if len(mean_vals) != valid_data.shape[1]:
                 raise ValueError(
                     f"Scaler 维度 ({len(mean_vals)}) 与当前数据的有效特征数 ({valid_data.shape[1]}) 不匹配。"
                 )

            logger.info(f"  使用按维度的 Scaler (均值/标准差向量形状: {mean_vals.shape})") # 使用 logger

            # 检查来自 scaler 的标准差
            if np.any(std_vals < epsilon):
                 num_small_std = np.sum(std_vals < epsilon)
                 warnings.warn(
                     f"来自 Scaler 的标准差向量中包含 {num_small_std} 个小于 epsilon ({epsilon:.1e}) 的值。"
                     f"将在分母加上 epsilon。", UserWarning
                 )

            # --- 应用按维度标准化 (利用 NumPy 广播机制) ---
            processed_data = (valid_data - mean_vals) / (std_vals + epsilon)

    else: # normalize = False
        scaler = None
        logger.info("信息：未执行标准化处理。") # 使用 logger

    logger.info(f"盐度场数据预处理完成。输出数据形状: {processed_data.shape}" + (" (训练集)" if is_training else " (验证/测试集)")) # 使用 logger

    return processed_data, scaler, valid_mask_2d

def preprocess_wind_data(wind_data, is_training=True, scaler=None, epsilon=1e-8):
    """
    预处理风场数据 (适应性修改以包含epsilon)
    ... (原始文档字符串) ...
    """
    # 检查并填充NaN (这里的逻辑可以保持，或根据风场数据的特性调整)
    if isinstance(wind_data, pd.DataFrame):
        # 假设DataFrame是 T x (2*H*W) 或类似结构
        wind_data_np = wind_data.fillna(method='ffill').fillna(method='bfill').to_numpy()
        # 可能需要根据实际DataFrame结构reshape回 (T, 2, H, W)
        # 这里假设输入已经是Numpy或能转换
    elif isinstance(wind_data, np.ndarray):
         wind_data_np = np.copy(wind_data) # 操作副本
         if np.isnan(wind_data_np).any():
            print("警告：风场数据包含NaN，将尝试使用线性插值填充（按通道）。")
            # 尝试更通用的插值（例如按时间序列插值每个点）
            # 假设形状是 (T, C, H, W) 或 (T, ...)
            original_shape = wind_data_np.shape
            # 展平空间和通道维度进行插值
            data_flat_spatial = wind_data_np.reshape(original_shape[0], -1)
            for i in range(data_flat_spatial.shape[1]):
                 series = data_flat_spatial[:, i]
                 mask = np.isnan(series)
                 if mask.any():
                     non_nan_indices = np.flatnonzero(~mask)
                     nan_indices = np.flatnonzero(mask)
                     if len(non_nan_indices) > 0: # 必须有非NaN值才能插值
                         series[mask] = np.interp(nan_indices, non_nan_indices, series[~mask])
                     else: # 如果整个序列都是NaN
                         series[mask] = 0 # 或者其他默认值
                         warnings.warn(f"风场数据中，索引 {i} 的整个时间序列都是NaN，已填充为0。", UserWarning)
            wind_data_np = data_flat_spatial.reshape(original_shape)
    else:
         raise TypeError(f"风场数据类型不支持: {type(wind_data)}")


    # 标准化
    if is_training:
        # 计算全局均值和标准差，保持通道维度
        # 假设形状 (T, C, H, W) 或 (T, C)
        axes_to_reduce = tuple(range(wind_data_np.ndim))
        mean_val = np.mean(wind_data_np, axis=axes_to_reduce, keepdims=True) # 可能需要调整axis
        std_val = np.std(wind_data_np, axis=axes_to_reduce, keepdims=True)   # 可能需要调整axis
        # 或者按通道计算？ axis=(0, *range(2, wind_data_np.ndim)) ? 取决于需求
        # 保持原代码逻辑: axis=(0, 1, 2) for (T, 2, H, W)
        if wind_data_np.ndim >= 3: # 适用于 (T, C, ...)
             mean_val = np.mean(wind_data_np, axis=(0, *range(2, wind_data_np.ndim)), keepdims=True)
             std_val = np.std(wind_data_np, axis=(0, *range(2, wind_data_np.ndim)), keepdims=True)
        elif wind_data_np.ndim == 2: # (T, C)
             mean_val = np.mean(wind_data_np, axis=0, keepdims=True)
             std_val = np.std(wind_data_np, axis=0, keepdims=True)
        else: # 其他情况，全局
             mean_val = np.mean(wind_data_np)
             std_val = np.std(wind_data_np)


        if np.any(std_val < epsilon):
            warnings.warn(f"风场训练集计算的标准差包含小于epsilon的值。将在分母加上epsilon。", UserWarning)
        scaler = {'mean': mean_val, 'std': std_val}
    else:
        if scaler is None:
            raise ValueError("使用测试/验证模式时必须提供风场scaler")
        mean_val = scaler['mean']
        std_val = scaler['std']
        if np.any(std_val < epsilon):
             warnings.warn(f"来自风场Scaler的标准差包含小于epsilon的值。将在分母加上epsilon。", UserWarning)

    # 应用标准化 (分母加epsilon)
    processed_data = (wind_data_np - mean_val) / (std_val + epsilon)
    print("风场数据预处理完成" + (" (训练集)" if is_training else " (验证/测试集)"))
    return processed_data, scaler


def preprocess_river_flow_data(flow_data, is_training=True, scaler=None, epsilon=1e-8):
    """
    预处理径流量数据 (适应性修改以包含epsilon)
     ... (原始文档字符串) ...
    """
    # 检查并填充NaN
    if isinstance(flow_data, pd.DataFrame):
        flow_data_np = flow_data.fillna(method='ffill').fillna(method='bfill').to_numpy()
    elif isinstance(flow_data, np.ndarray):
        flow_data_np = np.copy(flow_data) # 操作副本
        mask = np.isnan(flow_data_np)
        if mask.any():
            print("警告：径流数据包含NaN，将尝试使用线性插值填充（按列/河流）。")
            for i in range(flow_data_np.shape[1]): # 假设是 (T, 河流数)
                series = flow_data_np[:, i]
                mask_col = np.isnan(series)
                if mask_col.any():
                    non_nan_indices = np.flatnonzero(~mask_col)
                    nan_indices = np.flatnonzero(mask_col)
                    if len(non_nan_indices) > 0:
                        series[mask_col] = np.interp(nan_indices, non_nan_indices, series[~mask_col])
                    else:
                        series[mask_col] = 0 # 或其他默认值
                        warnings.warn(f"径流数据中，河流 {i} 的整个时间序列都是NaN，已填充为0。", UserWarning)

    else:
        raise TypeError(f"径流数据类型不支持: {type(flow_data)}")

    # 标准化
    if is_training:
        # 按列（河流）计算均值和标准差
        mean_val = np.mean(flow_data_np, axis=0, keepdims=True)
        std_val = np.std(flow_data_np, axis=0, keepdims=True)
        if np.any(std_val < epsilon):
             warnings.warn(f"径流训练集计算的标准差包含小于epsilon的值。将在分母加上epsilon。", UserWarning)
        scaler = {'mean': mean_val, 'std': std_val}
    else:
        if scaler is None:
            raise ValueError("使用测试/验证模式时必须提供径流scaler")
        mean_val = scaler['mean']
        std_val = scaler['std']
        if np.any(std_val < epsilon):
             warnings.warn(f"来自径流Scaler的标准差包含小于epsilon的值。将在分母加上epsilon。", UserWarning)

    # 应用标准化 (分母加epsilon)
    processed_data = (flow_data_np - mean_val) / (std_val + epsilon)
    print("径流量数据预处理完成" + (" (训练集)" if is_training else " (验证/测试集)"))
    return processed_data, scaler


def process_dataset_split(salinity_data, wind_data, flow_data, grid_info, wind_info,
                          train_size, val_size, processed_data_dir):
    """
    处理并保存训练集、验证集和测试集 (调整以适应新的预处理函数)
     ... (原始文档字符串) ...
    """
    print("-" * 30)
    print("开始处理训练集...")
    # 处理训练集 - 盐度
    train_salinity = salinity_data[:train_size]
    # is_training=True 会计算scaler和mask
    train_salinity_processed, salinity_scaler, salinity_valid_mask = \
        preprocess_salinity_data(train_salinity, is_training=True)
    # 保存 scaler 和 mask
    np.save(os.path.join(processed_data_dir, "salinity_scaler.npy"), salinity_scaler)
    np.save(os.path.join(processed_data_dir, "salinity_valid_mask.npy"), salinity_valid_mask)
    np.save(os.path.join(processed_data_dir, "train_salinity_processed.npy"), train_salinity_processed)
    print(f"训练集盐度处理完毕，保存scaler, mask, data。数据形状: {train_salinity_processed.shape}")
    del train_salinity, train_salinity_processed # 及时释放内存
    gc.collect()

    # 处理训练集 - 风场和径流 (保持原逻辑，但调用更新后的函数)
    train_wind = wind_data[:train_size]
    train_wind_processed, wind_scaler = preprocess_wind_data(train_wind, is_training=True)
    np.save(os.path.join(processed_data_dir, "wind_scaler.npy"), wind_scaler)
    np.save(os.path.join(processed_data_dir, "train_wind_processed.npy"), train_wind_processed)
    del train_wind, train_wind_processed
    gc.collect()

    train_flow = flow_data[:train_size]
    train_flow_processed, flow_scaler = preprocess_river_flow_data(train_flow, is_training=True)
    np.save(os.path.join(processed_data_dir, "flow_scaler.npy"), flow_scaler)
    np.save(os.path.join(processed_data_dir, "train_flow_processed.npy"), train_flow_processed)
    del train_flow, train_flow_processed
    gc.collect()
    print("训练集处理完毕。")
    print("-" * 30)

    # --- 处理验证集 ---
    print("开始处理验证集...")
    val_start = train_size
    val_end = train_size + val_size

    # 处理验证集 - 盐度
    val_salinity = salinity_data[val_start:val_end]
    # is_training=False 使用训练时得到的 scaler 和 mask
    # preprocess_salinity_data 内部会使用 mask 提取有效点，然后用 scaler 标准化
    val_salinity_processed, _, _ = \
        preprocess_salinity_data(val_salinity, is_training=False, scaler=salinity_scaler)
        # 注意：这里不需要重新保存 scaler 或 mask，返回值中的 scaler 和 mask 会是 None 或输入的 scaler
    np.save(os.path.join(processed_data_dir, "val_salinity_processed.npy"), val_salinity_processed)
    print(f"验证集盐度处理完毕。数据形状: {val_salinity_processed.shape}")
    del val_salinity, val_salinity_processed
    gc.collect()

    # 处理验证集 - 风场和径流
    val_wind = wind_data[val_start:val_end]
    val_wind_processed, _ = preprocess_wind_data(val_wind, is_training=False, scaler=wind_scaler)
    np.save(os.path.join(processed_data_dir, "val_wind_processed.npy"), val_wind_processed)
    del val_wind, val_wind_processed
    gc.collect()

    val_flow = flow_data[val_start:val_end]
    val_flow_processed, _ = preprocess_river_flow_data(val_flow, is_training=False, scaler=flow_scaler)
    np.save(os.path.join(processed_data_dir, "val_flow_processed.npy"), val_flow_processed)
    del val_flow, val_flow_processed
    gc.collect()
    print("验证集处理完毕。")
    print("-" * 30)

    # --- 处理测试集 ---
    print("开始处理测试集...")
    # 处理测试集 - 盐度
    test_salinity = salinity_data[val_end:]
    if test_salinity.shape[0] > 0: # 确保测试集有数据
        test_salinity_processed, _, _ = \
            preprocess_salinity_data(test_salinity, is_training=False, scaler=salinity_scaler)
        np.save(os.path.join(processed_data_dir, "test_salinity_processed.npy"), test_salinity_processed)
        print(f"测试集盐度处理完毕。数据形状: {test_salinity_processed.shape}")
        del test_salinity, test_salinity_processed
    else:
        print("测试集盐度数据为空，跳过处理。")
    gc.collect()


    # 处理测试集 - 风场和径流
    test_wind = wind_data[val_end:]
    if test_wind.shape[0] > 0:
        test_wind_processed, _ = preprocess_wind_data(test_wind, is_training=False, scaler=wind_scaler)
        np.save(os.path.join(processed_data_dir, "test_wind_processed.npy"), test_wind_processed)
        del test_wind, test_wind_processed
    else:
        print("测试集风场数据为空，跳过处理。")
    gc.collect()

    test_flow = flow_data[val_end:]
    if test_flow.shape[0] > 0:
        test_flow_processed, _ = preprocess_river_flow_data(test_flow, is_training=False, scaler=flow_scaler)
        np.save(os.path.join(processed_data_dir, "test_flow_processed.npy"), test_flow_processed)
        del test_flow, test_flow_processed
    else:
        print("测试集径流数据为空，跳过处理。")
    gc.collect()
    print("测试集处理完毕。")
    print("-" * 30)

    # 保存网格信息 (保持不变)
    print("保存网格信息...")
    if grid_info is not None:
        # 确保 grid_info 中的数组也是 numpy array
        grid_info_np = {k: np.array(v) if not isinstance(v, np.ndarray) else v for k, v in grid_info.items()}
        np.savez(os.path.join(processed_data_dir, "grid_info.npz"), **grid_info_np)
        print("  grid_info 已保存。")
    if wind_info is not None:
        wind_info_np = {k: np.array(v) if not isinstance(v, np.ndarray) else v for k, v in wind_info.items()}
        np.savez(os.path.join(processed_data_dir, "wind_info.npz"), **wind_info_np)
        print("  wind_info 已保存。")
    print("网格信息保存完毕。")


def process_and_save_data(raw_data_dir, processed_data_dir, cfg=None):
    """
    主函数：加载、处理并保存所有数据集，并保存分割索引
    
    参数:
        raw_data_dir: 原始数据目录
        processed_data_dir: 处理后数据保存目录
        cfg: Hydra配置对象，用于读取数据分割比例
    """
    os.makedirs(processed_data_dir, exist_ok=True)
    print(f"原始数据目录: {raw_data_dir}")
    print(f"处理后数据目录: {processed_data_dir}")

    # 定义文件路径 (保持不变)
    salt_path = os.path.join(raw_data_dir, "salt_grid.npy")
    vertices_path = os.path.join(raw_data_dir, "vertices.npy")
    triangles_path = os.path.join(raw_data_dir, "triangles.npy")
    salt_lonlat_path = os.path.join(raw_data_dir, "salt_lonlat_grid.npy")
    wind_path = os.path.join(raw_data_dir, "wind.npy")
    wind_lonlat_path = os.path.join(raw_data_dir, "wind_lonlat.npy")
    flow_path = os.path.join(raw_data_dir, "flow.npy")

    # 检查主数据文件是否存在
    required_files = [salt_path, wind_path, flow_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
         print(f"错误: 缺少必要的原始数据文件: {', '.join(missing_files)}")
         print("请确保所有必需的 .npy 文件都在原始数据目录中。")
         return # 无法继续

    try:
        # --- 加载数据 ---
        # 使用假设的加载函数，你需要确保它们能正确加载数据
        print("加载数据...")
        # load_salinity_data 现在应该只返回 3D 盐度数据和 grid_info
        salinity_data, grid_info = load_salinity_data(salt_path, vertices_path, triangles_path, salt_lonlat_path)
        wind_data, wind_info = load_wind_data(wind_path, wind_lonlat_path)
        flow_data = load_river_flow_data(flow_path)
        print("数据加载完毕。")

        # --- 计算数据分割点 ---
        # 假设盐度数据的时间维度是主导
        total_samples = salinity_data.shape[0]
        if total_samples == 0:
            print("错误：加载的盐度数据时间步数为0。")
            return

        # 从配置中读取分割比例，如果没有提供配置，则使用默认值
        if cfg is not None:
            # 读取配置中的分割比例
            validation_split = cfg.training.get("validation_split", 0.2)
            test_split = cfg.training.get("test_split", 0.1)
            train_ratio = 1.0 - validation_split - test_split
            val_ratio = validation_split
            
            print(f"使用配置中的数据分割比例: 训练集 {train_ratio:.2f}，验证集 {val_ratio:.2f}，测试集 {test_split:.2f}")
        else:
            # 使用默认比例分割
            train_ratio = 0.7
            val_ratio = 0.15
            print(f"使用默认数据分割比例: 训练集 {train_ratio}，验证集 {val_ratio}，测试集 {1.0-train_ratio-val_ratio}")
        
        train_size = int(train_ratio * total_samples)
        val_size = int(val_ratio * total_samples)
        test_size = total_samples - train_size - val_size

        # 防止计算误差导致 test_size < 0
        if test_size < 0:
            val_size = total_samples - train_size # 调整验证集大小
            test_size = 0

        logger.info(f"数据总量: {total_samples}，分割为：训练集 {train_size}，验证集 {val_size}，测试集 {test_size}")

        # --- 定义并保存原始分割索引 ---
        train_indices = np.arange(0, train_size)
        val_indices = np.arange(train_size, train_size + val_size)
        test_indices = np.arange(train_size + val_size, total_samples)

        split_indices_path = os.path.join(processed_data_dir, "split_indices.npz")
        try:
            # 使用 savez 保存多个数组
            np.savez(split_indices_path,
                     train_indices=train_indices,
                     val_indices=val_indices,
                     test_indices=test_indices)
            logger.info(f"原始数据分割索引已保存到: {split_indices_path}")
        except Exception as e:
             logger.error(f"保存分割索引失败: {e}")
             # 不退出，继续后续处理

        # --- 处理并保存各个数据集 ---
        process_dataset_split(salinity_data, wind_data, flow_data, grid_info, wind_info,
                              train_size, val_size, processed_data_dir)

        logger.info(f"数据预处理完成，结果保存在 {processed_data_dir} 目录下")

    except FileNotFoundError as e:
        logger.error(f"加载数据时文件未找到错误: {e}")
        logger.error("请检查原始数据目录中的文件是否齐全且路径正确。")
    except MemoryError:
        logger.error("内存错误：处理数据时内存不足。请尝试在内存更大的机器上运行或减少处理的数据量。")
    except Exception as e:
        logger.error(f"处理数据时发生未预料的错误: {e}", exc_info=True)

# 假设的加载函数 (你需要根据实际情况实现)
def load_salinity_data(salt_path, vertices_path, triangles_path, salt_lonlat_path):
    print(f"  加载盐度数据从: {salt_path}")
    salinity_data = np.load(salt_path) # 假设已经是 (T, N, M) 或需要reshape
    # 示例：如果原始是 (T, K)，需要根据lonlat或grid信息reshape
    # K = N * M
    # salinity_data = salinity_data.reshape(salinity_data.shape[0], N, M) # 需要知道N, M

    grid_info = {}
    if os.path.exists(vertices_path):
        grid_info['vertices'] = np.load(vertices_path)
        print(f"    加载顶点: {grid_info['vertices'].shape}")
    if os.path.exists(triangles_path):
        grid_info['triangles'] = np.load(triangles_path)
        print(f"    加载三角元: {grid_info['triangles'].shape}")
    if os.path.exists(salt_lonlat_path):
        grid_info['lonlat'] = np.load(salt_lonlat_path)
        print(f"    加载盐度经纬度: {grid_info['lonlat'].shape}")
        # 如果需要根据lonlat推断 N, M
        # N, M = calculate_grid_dims(grid_info['lonlat'])
        # if salinity_data.ndim == 2:
        #    salinity_data = salinity_data.reshape(salinity_data.shape[0], N, M)
    # 确保返回的是 3D 数组
    if salinity_data.ndim == 2 and 'lonlat' in grid_info:
         # 尝试根据 lonlat 点数推断 reshape
         num_points = grid_info['lonlat'].shape[0]
         if salinity_data.shape[1] == num_points:
              # 尝试找到合适的 N, M. 这比较困难，除非网格是规则的
              # 假设 grid_info['lonlat'] 对应的是 (N*M, 2)
              # 这里需要更具体的逻辑来确定 N 和 M
              # **** 这是一个需要你根据数据具体情况填充的逻辑 ****
              print("警告: 盐度数据是2D，尝试根据lonlat重塑为3D，但这需要明确的N,M信息。")
              # 示例：如果假设 N=sqrt(num_points)
              # N_approx = int(np.sqrt(num_points))
              # if N_approx * N_approx == num_points:
              #    salinity_data = salinity_data.reshape(salinity_data.shape[0], N_approx, N_approx)
              # else:
              #    print("无法自动推断N, M，保持2D。后续处理可能失败。")
              pass # 保持2D或根据实际情况修改

    elif salinity_data.ndim != 3:
         print(f"警告: 加载的盐度数据维度不是3D ({salinity_data.ndim})，后续处理可能失败。")


    return salinity_data, grid_info

def load_wind_data(wind_path, wind_lonlat_path):
    print(f"  加载风场数据从: {wind_path}")
    wind_data = np.load(wind_path)
    wind_info = {}
    if os.path.exists(wind_lonlat_path):
        wind_info['lonlat'] = np.load(wind_lonlat_path)
        print(f"    加载风场经纬度: {wind_info['lonlat'].shape}")
    return wind_data, wind_info

def load_river_flow_data(flow_path):
    print(f"  加载径流数据从: {flow_path}")
    flow_data = np.load(flow_path)
    # 确保是2D (T, n_rivers)
    if flow_data.ndim == 1:
        print("警告: 径流数据是1D，假设只有一条河流，reshape为 (T, 1)")
        flow_data = flow_data.reshape(-1, 1)
    elif flow_data.ndim > 2:
        print(f"警告: 径流数据维度大于2 ({flow_data.ndim})，可能处理错误。")
    return flow_data

# --- 使用示例 ---
if __name__ == "__main__":
    raw_dir = "path/to/your/raw/data"
    processed_dir = "path/to/save/processed/data"
    process_and_save_data(raw_dir, processed_dir)