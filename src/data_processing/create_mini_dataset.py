"""
数据压缩脚本

该脚本将完整的2557天数据集压缩为257天的小型数据集，
用于快速测试和开发。压缩后的数据保存在data/raw/mini目录中。
"""

import os
import numpy as np
import shutil
from pathlib import Path

def create_mini_dataset(source_dir, target_dir, target_days=257):
    """
    从原始数据集创建小型数据集
    
    参数:
        source_dir: 源数据目录
        target_dir: 目标数据目录
        target_days: 压缩后的天数
    """
    print(f"开始创建{target_days}天的小型数据集...")
    
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 定义需要压缩的时间序列数据文件
    time_series_files = ['salt.npy', 'wind.npy', 'flow.npy','salt_grid.npy']
    
    # 定义不需要压缩的网格和坐标文件
    static_files = [
        'salt_lonlat.npy', 'wind_lonlat.npy',
        'vertices.npy', 'triangles.npy',
        'salt_lonlat_grid.npy',
        'mask.npy'
    ]
    
    # 压缩时间序列数据
    for file_name in time_series_files:
        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(target_dir, file_name)
        
        if not os.path.exists(source_path):
            print(f"警告: 源文件 {source_path} 不存在，跳过")
            continue
        
        print(f"压缩数据文件: {file_name}")
        
        # 加载数据
        data = np.load(source_path)
        original_days = data.shape[0]
        
        # 计算采样间隔
        if original_days <= target_days:
            print(f"警告: 源数据天数({original_days})小于或等于目标天数({target_days})，将直接复制")
            mini_data = data
        else:
            # 采样间隔 - 使用线性间隔确保覆盖整个时间范围
            indices = np.linspace(0, original_days-1, target_days, dtype=int)
            mini_data = data[indices]
            print(f"从 {original_days} 天压缩到 {target_days} 天，采样率: 1/{original_days/target_days:.2f}")
        
        # 保存压缩后的数据
        np.save(target_path, mini_data)
        print(f"压缩后的数据形状: {mini_data.shape}, 保存到 {target_path}")
        
        # 释放内存
        del data, mini_data
    
    # 复制静态文件
    for file_name in static_files:
        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(target_dir, file_name)
        
        if not os.path.exists(source_path):
            print(f"警告: 源文件 {source_path} 不存在，跳过")
            continue
        
        print(f"复制静态文件: {file_name}")
        shutil.copy2(source_path, target_path)
        print(f"复制完成: {target_path}")
    
    print(f"小型数据集创建完成，保存在: {target_dir}")

def main():
    # 定义源数据和目标数据目录
    source_dir = os.path.join('data', 'raw')
    target_dir = os.path.join('data', 'raw', 'mini')
    
    # 获取当前工作目录
    current_dir = os.getcwd()
    
    # 如果当前不是项目根目录，尝试找到项目根目录
    if not os.path.exists(os.path.join(current_dir, source_dir)):
        # 检查是否在DRPR目录下
        if os.path.basename(current_dir) == 'DRPR':
            source_dir = os.path.join(current_dir, source_dir)
            target_dir = os.path.join(current_dir, target_dir)
        else:
            # 向上查找至DRPR目录
            parent_dir = Path(current_dir).parent
            if os.path.basename(parent_dir) == 'DRPR':
                source_dir = os.path.join(parent_dir, source_dir)
                target_dir = os.path.join(parent_dir, target_dir)
            else:
                print("错误: 无法确定项目根目录，请在项目根目录或src目录下运行此脚本")
                return
    else:
        source_dir = os.path.join(current_dir, source_dir)
        target_dir = os.path.join(current_dir, target_dir)
    
    print(f"源数据目录: {source_dir}")
    print(f"目标数据目录: {target_dir}")
    
    # 创建小型数据集（257天）
    create_mini_dataset(source_dir, target_dir, target_days=257)
    
    # 提示下一步操作
    print("\n小型数据集创建成功！您可以通过以下方式使用它:")
    print("1. 直接使用 data/raw/mini 目录中的数据")
    print("2. 修改配置文件中的数据路径指向mini目录")
    print("3. 使用命令行参数指定数据目录")

if __name__ == "__main__":
    main()