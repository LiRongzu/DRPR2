import numpy as np
import os
import gc
import pickle
import psutil

def create_input_output_pairs_batched(data, input_seq_length, output_seq_length, stride=1, batch_size=100, output_path=None, dataset_type=None):
    """
    创建用于时序预测的输入-输出数据对，支持批处理以节省内存
    
    参数:
        data: 对齐后的数据字典
        input_seq_length: 输入序列长度
        output_seq_length: 输出序列长度（预测步数）
        stride: 滑动窗口的步长
        batch_size: 批处理大小，处理和保存的批次大小
        output_path: 输出路径，如果不为None则保存每个批次的结果
        dataset_type: 数据集类型，用于生成保存文件名
        
    返回:
        如果output_path为None:
            X: 输入特征字典
            y: 输出标签
        否则:
            None (数据已保存到文件)
    """
    salinity_data = data['salinity']
    wind_data = data['wind']
    flow_data = data['flow']
    
    # 获取时间步数和数据形状
    n_samples = len(salinity_data)
    
    # 计算可以创建的样本数
    n_pairs = (n_samples - input_seq_length - output_seq_length) // stride + 1
    
    print(f"将创建{n_pairs}个输入-输出数据对")
    print(f"使用批处理大小: {batch_size}")
    
    # 如果开启了直接保存模式，创建批次索引
    if output_path is not None and dataset_type is not None:
        X_file = os.path.join(output_path, f"X_{dataset_type}.npy")
        y_file = os.path.join(output_path, f"y_{dataset_type}.npy")
        
        # 检查是否需要创建新文件
        is_first_batch = True
        
        # 打印内存使用情况
        process = psutil.Process(os.getpid())
        print(f"当前内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    # 如果不开启直接保存模式，初始化输出列表
    else:
        X_salinity_all = []
        X_wind_all = []
        X_flow_all = []
        y_all = []
    
    # 按批次处理
    for batch_start in range(0, n_pairs, batch_size):
        batch_end = min(batch_start + batch_size, n_pairs)
        current_batch_size = batch_end - batch_start
        print(f"处理批次 {batch_start//batch_size + 1}/{(n_pairs+batch_size-1)//batch_size}，样本 {batch_start}-{batch_end-1}")
        
        # 初始化当前批次的数组
        X_salinity_batch = np.zeros((current_batch_size, input_seq_length) + salinity_data.shape[1:], dtype=salinity_data.dtype)
        X_wind_batch = np.zeros((current_batch_size, input_seq_length) + wind_data.shape[1:], dtype=wind_data.dtype)
        X_flow_batch = np.zeros((current_batch_size, input_seq_length) + flow_data.shape[1:], dtype=flow_data.dtype)
        y_batch = np.zeros((current_batch_size, output_seq_length) + salinity_data.shape[1:], dtype=salinity_data.dtype)
        
        # 填充当前批次的数据
        for i in range(current_batch_size):
            idx = (batch_start + i) * stride
            X_salinity_batch[i] = salinity_data[idx:idx+input_seq_length]
            X_wind_batch[i] = wind_data[idx:idx+input_seq_length]
            X_flow_batch[i] = flow_data[idx:idx+input_seq_length]
            y_batch[i] = salinity_data[idx+input_seq_length:idx+input_seq_length+output_seq_length]
        
        # 创建特征字典
        X_batch = {
            'salinity': X_salinity_batch,
            'wind': X_wind_batch,
            'flow': X_flow_batch
        }
        
        # 如果开启了直接保存模式
        if output_path is not None and dataset_type is not None:
            # 第一个批次直接写入文件
            if is_first_batch:
                # 保存输入-输出对 - 使用np.save标准参数
                np.save(X_file, X_batch, allow_pickle=True)
                np.save(y_file, y_batch, allow_pickle=True)
                is_first_batch = False
            else:
                # 后续批次追加到已有文件
                # 先读取之前保存的数据
                X_prev = np.load(X_file, allow_pickle=True).item()
                y_prev = np.load(y_file)
                
                # 合并数据
                for key in X_prev:
                    X_prev[key] = np.concatenate([X_prev[key], X_batch[key]], axis=0)
                y_concat = np.concatenate([y_prev, y_batch], axis=0)
                
                # 保存合并后的数据 - 使用np.save标准参数
                np.save(X_file, X_prev, allow_pickle=True)
                np.save(y_file, y_concat, allow_pickle=True)
                
                # 清理内存
                del X_prev, y_prev, y_concat
                gc.collect()
            
            # 清理当前批次的内存
            del X_batch, X_salinity_batch, X_wind_batch, X_flow_batch, y_batch
            gc.collect()
            
            # 打印内存使用情况
            print(f"当前内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        
        # 如果未开启直接保存模式，保存到列表中
        else:
            X_salinity_all.append(X_salinity_batch)
            X_wind_all.append(X_wind_batch)
            X_flow_all.append(X_flow_batch)
            y_all.append(y_batch)
            
            # 清理当前批次的内存
            del X_batch, X_salinity_batch, X_wind_batch, X_flow_batch, y_batch
            gc.collect()
    
    # 如果开启了直接保存模式，返回None（数据已保存）
    if output_path is not None and dataset_type is not None:
        return None, None
    
    # 如果未开启直接保存模式，返回合并后的结果
    else:
        # 合并所有批次的结果
        X_salinity_all = np.concatenate(X_salinity_all, axis=0)
        X_wind_all = np.concatenate(X_wind_all, axis=0)
        X_flow_all = np.concatenate(X_flow_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)
        
        # 创建最终特征字典
        X = {
            'salinity': X_salinity_all,
            'wind': X_wind_all,
            'flow': X_flow_all
        }
        
        return X, y_all

def create_and_save_pairs(data_dir, output_dir, dataset_type, input_seq_length, output_seq_length, stride=1):
    """
    从保存的处理后数据创建并保存输入-输出对
    
    参数:
        data_dir: 原始数据目录（存放处理后的salinity、wind和flow数据）
        output_dir: 输出目录（存放生成的X和y数据）
        dataset_type: 数据集类型（train, val, test）
        input_seq_length: 输入序列长度
        output_seq_length: 输出序列长度
        stride: 滑动窗口步长，默认值改为1以增加样本间的重叠
    """
    try:
        print(f"为{dataset_type}集创建输入-输出对...")
        
        # 检查是否已存在输出文件
        x_file_path = os.path.join(output_dir, f"X_{dataset_type}.npy")
        y_file_path = os.path.join(output_dir, f"y_{dataset_type}.npy")
        
        if (os.path.exists(x_file_path) and os.path.exists(y_file_path)):
            print(f"{dataset_type}集的输入-输出对已存在，跳过处理")
            return
        
        # 加载处理后的数据
        salinity = np.load(os.path.join(data_dir, f"{dataset_type}_salinity_processed.npy"))
        print(f"加载{dataset_type}集盐度场数据成功")
        wind = np.load(os.path.join(data_dir, f"{dataset_type}_wind_processed.npy"))
        print(f"加载{dataset_type}集风场数据成功")
        flow = np.load(os.path.join(data_dir, f"{dataset_type}_flow_processed.npy"))
        print(f"加载{dataset_type}径流数据成功")
        
        # 获取时间步数
        n_samples = len(salinity)
        
        # 计算可以创建的样本数
        n_pairs = (n_samples - input_seq_length - output_seq_length) // stride + 1
        if n_pairs <= 0:
            print(f"警告: {dataset_type}集数据不足，无法创建输入-输出对")
            return
        
        print(f"将为{dataset_type}集创建{n_pairs}个输入-输出数据对")
        
        # 估算内存需求
        sample_size_mb = (salinity[0].nbytes + wind[0].nbytes + flow[0].nbytes) / (1024 * 1024)
        estimated_memory_mb = n_pairs * sample_size_mb * (input_seq_length + output_seq_length) * 2  # 包括输入输出和临时变量
        
        print(f"估计内存需求: {estimated_memory_mb:.2f} MB")
        
        # 获取可用内存
        available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
        print(f"当前可用内存: {available_memory_mb:.2f} MB")
        
        # 创建数据字典
        data = {
            'salinity': salinity,
            'wind': wind,
            'flow': flow
        }
        
        # 根据内存情况选择批处理大小
        if estimated_memory_mb > available_memory_mb * 0.7:  # 使用70%可用内存作为阈值
            print("预计内存使用较大，使用批处理并直接保存")
            # 计算合适的批处理大小
            batch_size = max(1, int(n_pairs * 0.3 * available_memory_mb / estimated_memory_mb))
            print(f"自动设置批处理大小为: {batch_size}")
            
            # 使用批处理并直接保存
            create_input_output_pairs_batched(data, input_seq_length, output_seq_length, stride, 
                                            batch_size=batch_size, output_path=output_dir, dataset_type=dataset_type)
        else:
            print("内存充足，一次性处理所有数据")
            # 创建输入-输出对
            X, y = create_input_output_pairs_batched(data, input_seq_length, output_seq_length, stride)
            
            # 保存输入-输出对 - 使用标准参数
            np.save(os.path.join(output_dir, f"X_{dataset_type}.npy"), X, allow_pickle=True)
            np.save(os.path.join(output_dir, f"y_{dataset_type}.npy"), y, allow_pickle=True)
            
            # 清理内存
            del X, y
        
        # 清理内存
        del data, salinity, wind, flow
        gc.collect()
        
        print(f"{dataset_type}集输入-输出对创建完成并保存")
        
    except Exception as e:
        print(f"处理{dataset_type}集时出错: {e}")
        import traceback
        traceback.print_exc()

def process_all_pairs(data_dir, output_dir, input_seq_length=6, output_seq_length=1, stride=1):
    """
    处理所有数据集的输入-输出对
    
    参数:
        data_dir: 原始数据目录（存放处理后的salinity、wind和flow数据）
        output_dir: 输出目录（存放生成的X和y数据）
        input_seq_length: 输入序列长度
        output_seq_length: 输出序列长度
        stride: 滑动窗口步长
    """
    try:
        # 检查psutil库是否安装
        try:
            import psutil
        except ImportError:
            print("警告: psutil库未安装，无法监控内存使用。建议安装: pip install psutil")
        
        # 为每个数据集创建输入-输出对
        for dataset_type in ['train', 'val', 'test']:
            create_and_save_pairs(data_dir, output_dir, dataset_type, input_seq_length, output_seq_length, stride)
        print("所有数据集的输入-输出对创建完成")
        
    except Exception as e:
        print(f"创建输入-输出对时出错: {e}")
        import traceback
        traceback.print_exc()

# 保留原始函数以兼容现有代码
def create_input_output_pairs(data, input_seq_length, output_seq_length, stride=1, batch_size=None):
    """
    创建用于时序预测的输入-输出数据对 (兼容性函数)
    """
    # 调用批处理版本的函数
    return create_input_output_pairs_batched(data, input_seq_length, output_seq_length, stride, 
                                           batch_size=batch_size if batch_size else 100)