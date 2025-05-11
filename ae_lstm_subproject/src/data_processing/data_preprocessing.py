import numpy as np
import pandas as pd # Keep import if other functions might use it elsewhere
import os
import gc
import warnings # To warn about small std dev
import logging # 确保导入 logging
# Assuming these are defined elsewhere:
from .data_validation import load_salinity_data, load_wind_data, load_river_flow_data
logger = logging.getLogger(__name__)

# Modify signature to accept external_mask
def preprocess_salinity_data(salinity_data, external_mask=None, normalize=True, is_training=True, scaler=None, epsilon=1e-8):
    """
    Preprocesses 3D salinity field data (time x height x width), handles NaN values, and performs normalization.
    *** Modified: Prioritizes using the provided external_mask (0=valid) to determine valid points ***

    Args:
        salinity_data (numpy.ndarray): Raw 3D salinity field data (num_timesteps, N, M).
        external_mask (numpy.ndarray, optional): Externally provided 2D mask (N, M), where 0 indicates valid points and 1 indicates invalid points.
                                                 If provided, this mask will be used preferentially.
        normalize (bool, default=True): Whether to perform normalization.
        is_training (bool, default=True): Whether this is training data.
        scaler (dict, optional): Scaler for test/validation sets, containing 'mean' and 'std' vectors.
        epsilon (float, default=1e-8): Small value to prevent division by zero.

    Returns:
        tuple:
            - processed_data (numpy.ndarray): Preprocessed 2D salinity data (num_timesteps, num_valid_points).
            - scaler (dict or None): Data scaler {'mean': mean_vector, 'std': std_vector}.
            - valid_mask_2d (numpy.ndarray): The final 2D boolean mask used (N, M), True indicates valid points.
    """
    # --- Input validation (unchanged) ---
    if not isinstance(salinity_data, np.ndarray):
        raise TypeError(f"输入 salinity_data 必须是 NumPy 数组, 得到 {type(salinity_data)}")
    if salinity_data.ndim != 3:
        raise ValueError(f"输入 salinity_data 必须是 3D 数组 (时间 x 高 x 宽), 得到维度 {salinity_data.ndim}")

    T, N, M = salinity_data.shape
    original_grid_points = N * M
    logger.info(f"输入盐度数据形状: ({T}, {N}, {M})")

    # --- Identify valid grid points (Modified: Prioritize external_mask) ---
    if external_mask is not None:
        logger.info("Using provided external_mask (0=valid) to determine valid grid points.")
        if not isinstance(external_mask, np.ndarray):
            raise TypeError(f"Provided external_mask must be a NumPy array, got {type(external_mask)}")
        if external_mask.shape != (N, M):
            raise ValueError(f"Provided external_mask shape {external_mask.shape} does not match salinity data spatial shape ({N}, {M}).")
        # Ensure mask contains only 0 and 1
        if not np.all(np.isin(external_mask, [0, 1])):
             warnings.warn("Provided external_mask contains values other than 0 and 1. Treating non-0 values as invalid.", UserWarning)
             external_mask = np.where(external_mask == 0, 0, 1) # Force to 0/1

        valid_mask_2d = (external_mask == 0) # True indicates valid points (value is 0)
        num_valid_points = np.sum(valid_mask_2d)
        logger.info(f"Number of valid grid points determined by external_mask: {num_valid_points} / {original_grid_points}")

    else:
        logger.info("external_mask not provided, determining valid grid points based on NaN values.")
        # --- Identify valid grid points (original logic) ---
        valid_mask_2d = ~np.isnan(salinity_data).any(axis=0) # True indicates non-NaN points
        num_valid_points = np.sum(valid_mask_2d)
        logger.info(f"Number of valid grid points determined by NaN values: {num_valid_points} / {original_grid_points}")


    if num_valid_points == 0:
        # Adjust error message based on the logic used
        source = "external_mask" if external_mask is not None else "NaN value detection"
        raise ValueError(f"Data error: No valid grid points found based on {source}. Cannot proceed.")

    # --- Extract valid data and convert to 2D (using the finally determined valid_mask_2d) ---
    # Note: valid_mask_2d here is boolean, True means valid
    if num_valid_points == original_grid_points:
        # If all points are valid, just reshape
        valid_data = salinity_data.reshape(T, -1)
    else:
        # Use boolean mask indexing
        valid_data = salinity_data[:, valid_mask_2d] # Shape (T, num_valid_points)
    logger.info(f"Shape of extracted valid data: {valid_data.shape}") # Use logger

    # --- Data normalization (unchanged, operates on valid_data) ---
    processed_data = valid_data # Default value

    if normalize:
        # ... (normalization logic remains unchanged, operates on valid_data) ...
        current_op = "Training set" if is_training else "Validation/Test set"
        logger.info(f"Performing per-feature normalization on valid salinity data ({current_op})...") # Use logger

        if is_training:
            # --- Calculate mean and std dev for each valid grid point in the training set (along time axis=0) ---
            mean_vals = np.mean(valid_data, axis=0) # Shape (num_valid_points,)
            std_vals = np.std(valid_data, axis=0)   # Shape (num_valid_points,)

            # Check for very small std dev values
            if np.any(std_vals < epsilon):
                num_small_std = np.sum(std_vals < epsilon)
                warnings.warn(
                    f"Standard deviation vector calculated for the training set contains {num_small_std} values smaller than epsilon ({epsilon:.1e}). "
                    f"Adding epsilon to the denominator to avoid division by zero.", UserWarning
                )
                # Optionally set std dev for these dimensions to epsilon or 1 to avoid large values
                # std_vals[std_vals < epsilon] = epsilon # e.g., set to epsilon

            # Save the per-feature scaler
            scaler = {'mean': mean_vals, 'std': std_vals}
            logger.info(f"  Calculated per-feature mean and std dev vectors, both with shape: {mean_vals.shape}") # Use logger

            # --- Apply per-feature normalization (using NumPy broadcasting) ---
            processed_data = (valid_data - mean_vals) / (std_vals + epsilon)

        else: # is_training = False (validation or test)
            if scaler is None:
                raise ValueError("Scaler must be provided when using test/validation mode and normalize=True")
            if 'mean' not in scaler or 'std' not in scaler:
                 raise ValueError("Provided scaler is missing 'mean' or 'std' key")

            mean_vals = scaler['mean'] # Should be a vector
            std_vals = scaler['std']   # Should be a vector

            # --- Check if scaler dimensions match the current data ---
            if not isinstance(mean_vals, np.ndarray) or mean_vals.ndim != 1 or \
               not isinstance(std_vals, np.ndarray) or std_vals.ndim != 1:
                 raise TypeError("Scaler 'mean' and 'std' must be NumPy vectors (1D array).")

            # Check if scaler dimension matches the number of valid features in the current data
            if len(mean_vals) != valid_data.shape[1]:
                 # valid_data.shape[1] here is already the dimension after applying the mask
                 raise ValueError(
                     f"Scaler dimension ({len(mean_vals)}) does not match the number of valid features in the current data ({valid_data.shape[1]}). "
                     f"This usually means the mask used during training differs from the mask used now."
                 )

            logger.info(f"  Using per-feature Scaler (mean/std dev vector shape: {mean_vals.shape})") # Use logger

            # Check std dev from scaler
            if np.any(std_vals < epsilon):
                 num_small_std = np.sum(std_vals < epsilon)
                 warnings.warn(
                     f"Standard deviation vector from Scaler contains {num_small_std} values smaller than epsilon ({epsilon:.1e}). "
                     f"Adding epsilon to the denominator.", UserWarning
                 )

            # --- Apply per-feature normalization (using NumPy broadcasting) ---
            processed_data = (valid_data - mean_vals) / (std_vals + epsilon)

    else: # normalize = False
        scaler = None
        logger.info("Info: Normalization not performed.") # Use logger

    logger.info(f"Salinity field data preprocessing finished. Output data shape: {processed_data.shape}" + (" (Training set)" if is_training else " (Validation/Test set)")) # Use logger

    # Return the final boolean mask used (True=valid)
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


# Modify signature to accept mask_data
def process_dataset_split(salinity_data, wind_data, flow_data, grid_info, wind_info,
                          train_size, val_size, processed_data_dir, mask_data=None): # Add mask_data parameter
    """
    处理并保存训练集、验证集和测试集 (调整以适应新的预处理函数和外部掩码)
     ... (原始文档字符串) ...
     Args:
         ...
         mask_data (numpy.ndarray, optional): Pre-loaded mask data (0=valid).
    """
    print("-" * 30)
    print("开始处理训练集...")
    # 处理训练集 - 盐度
    train_salinity = salinity_data[:train_size]
    # is_training=True 会计算scaler和mask
    # Pass mask_data as external_mask
    train_salinity_processed, salinity_scaler, salinity_valid_mask = preprocess_salinity_data(train_salinity, external_mask=mask_data, is_training=True)
    # 保存 scaler 和 mask (salinity_valid_mask is the mask actually used, could be from external or NaN)
    np.save(os.path.join(processed_data_dir, "salinity_scaler.npy"), salinity_scaler)
    # It might be more informative to save the mask that was *actually* used by preprocess_salinity_data
    np.save(os.path.join(processed_data_dir, "salinity_valid_mask_used.npy"), salinity_valid_mask)
    np.save(os.path.join(processed_data_dir, "train_salinity_processed.npy"), train_salinity_processed)
    print(f"训练集盐度处理完毕，保存scaler, used_mask, data。数据形状: {train_salinity_processed.shape}")
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
    # is_training=False 使用训练时得到的 scaler
    # Pass mask_data as external_mask
    val_salinity_processed, _, _ = preprocess_salinity_data(val_salinity, external_mask=mask_data, is_training=False, scaler=salinity_scaler)
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
        # Pass mask_data as external_mask
        test_salinity_processed, _, _ = preprocess_salinity_data(test_salinity, external_mask=mask_data, is_training=False, scaler=salinity_scaler)
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
    *** 修改：加载并传递 mask.npy ***

    参数:
        raw_data_dir: 原始数据目录
        processed_data_dir: 处理后数据保存目录
        cfg: Hydra配置对象，用于读取数据分割比例
    """ 
    os.makedirs(processed_data_dir, exist_ok=True)
    print(f"原始数据目录: {raw_data_dir}")
    print(f"处理后数据目录: {processed_data_dir}")

    # 定义文件路径
    salt_path = os.path.join(raw_data_dir, "salt_grid.npy")
    vertices_path = os.path.join(raw_data_dir, "vertices.npy")
    triangles_path = os.path.join(raw_data_dir, "triangles.npy")
    salt_lonlat_path = os.path.join(raw_data_dir, "salt_lonlat_grid.npy")
    wind_path = os.path.join(raw_data_dir, "wind.npy")
    wind_lonlat_path = os.path.join(raw_data_dir, "wind_lonlat.npy")
    flow_path = os.path.join(raw_data_dir, "flow.npy")
    mask_path = os.path.join(raw_data_dir, "mask.npy") # Define mask path

    # 检查主数据文件是否存在
    required_files = [salt_path, wind_path, flow_path, mask_path] # Add mask_path to check
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
         print(f"错误: 缺少必要的原始数据文件: {', '.join(missing_files)}")
         print("请确保所有必需的 .npy 文件都在原始数据目录中。")
         return # 无法继续

    try:
        # --- 加载数据 ---
        print("加载数据...")
        salinity_data, grid_info = load_salinity_data(salt_path, vertices_path, triangles_path, salt_lonlat_path)
        wind_data, wind_info = load_wind_data(wind_path, wind_lonlat_path)
        flow_data = load_river_flow_data(flow_path)
        mask_data = np.load(mask_path) # Load mask data
        print(f"  加载掩码数据从: {mask_path}, 形状: {mask_data.shape}")
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
        # Pass loaded mask_data to process_dataset_split
        process_dataset_split(salinity_data, wind_data, flow_data, grid_info, wind_info,
                              train_size, val_size, processed_data_dir, mask_data=mask_data)

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