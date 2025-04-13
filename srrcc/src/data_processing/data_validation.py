"""
数据验证模块 (data_validation.py)

该模块负责加载和验证河口盐度场预测所需的各类原始数据，包括盐度场数据、风场数据和河流径流量数据。
提供了一系列加载函数，用于读取不同格式的数据文件并进行基本验证，确保数据完整性和一致性。
所有加载函数都具有详细的错误处理和日志记录功能，便于调试和问题排查。

主要功能:
- 加载三角网格的河口盐度场时序数据
- 加载风场栅格数据及其经纬度坐标信息
- 加载河流径流量时序数据
- 验证数据完整性和格式正确性
- 详细的数据加载状态日志输出
"""

import numpy as np
import os

def load_salinity_data(salt_path="/home/lirz6/PROGRAM/NEWDC2/data/raw/salt.npy",
                        vertices_path="/home/lirz6/PROGRAM/NEWDC2/data/raw/vertices.npz",
                        triangles_path="/home/lirz6/PROGRAM/NEWDC2/data/raw/triangles.npz", 
                        salt_lonlat_path="/home/lirz6/PROGRAM/NEWDC2/data/raw/salt_lonlat.npy"):
    """
    加载三角网格的河口盐度场时序数据
    
    该函数负责加载河口盐度场的原始数据和相关的三角网格信息，包括顶点坐标和连接关系。
    这些数据通常来自数值模拟或观测，表示为时间序列的三角网格数据。
    
    参数:
        salt_path (str): 盐度场数据文件路径，形状为 (时间步数, 顶点数)
                         包含每个时间步在每个网格点上的盐度值
        vertices_path (str): 三角网格顶点坐标文件路径，形状为 (顶点数, 2)
                            包含每个顶点的平面坐标 (x, y)
        triangles_path (str): 三角形顶点组合方式文件路径，形状为 (三角形数, 3)
                             每行包含构成一个三角形的三个顶点索引
        salt_lonlat_path (str): 盐度场数据点的经纬度坐标文件路径，形状为 (2, 高, 宽)
                               索引0为经度，索引1为纬度
        
    返回:
        tuple: 包含两个元素:
            - salinity_data (numpy.ndarray): 盐度场数据，形状为 (时间步数, 网格点数)
            - grid_info (dict): 网格信息字典，包含以下键:
                - 'vertices': 顶点坐标数组
                - 'triangles': 三角形连接信息
                - 'salt_lonlat': 盐度场网格点的经纬度坐标（如果存在）

    """
    try:
        # 加载盐度场数据
        salinity_data = np.load(salt_path)
        print(f"成功加载盐度场数据，形状: {salinity_data.shape}")
        
        # 加载网格信息
        grid_info = {}
        
        # 加载顶点坐标
        vertices_data = np.load(vertices_path)
        grid_info['vertices'] = vertices_data['arr_0'] if 'arr_0' in vertices_data else vertices_data
        print(f"成功加载顶点坐标，形状: {grid_info['vertices'].shape}")
        
        # 加载三角形连接信息
        triangles_data = np.load(triangles_path)
        grid_info['triangles'] = triangles_data['arr_0'] if 'arr_0' in triangles_data else triangles_data
        print(f"成功加载三角形连接信息，形状: {grid_info['triangles'].shape}")
        
        # 加载盐度场经纬度坐标
        if os.path.exists(salt_lonlat_path):
            grid_info['salt_lonlat'] = np.load(salt_lonlat_path)
            print(f"成功加载盐度场经纬度坐标，形状: {grid_info['salt_lonlat'].shape}")
        
        return salinity_data, grid_info
    
    except Exception as e:
        print(f"加载盐度场数据失败: {e}")
        return None, None

def load_wind_data(wind_path="/home/lirz6/PROGRAM/NEWDC2/data/raw/wind.npy", 
                  wind_lonlat_path="/home/lirz6/PROGRAM/NEWDC2/data/raw/wind_lonlat.npy"):
    """
    加载风场栅格数据
    
    该函数负责加载风场数据及其相关的经纬度坐标信息。风场数据通常包含
    风速的两个分量(u, v)，表示为时间序列的多维栅格数据。
    
    参数:
        wind_path (str): 风场数据文件路径，形状为 (时间步数, 2, 高, 宽)
                         其中2表示风速的两个分量(u, v)
        wind_lonlat_path (str): 风场栅格点的经纬度坐标文件路径，形状为 (2, 高, 宽)
                                索引0为经度，索引1为纬度
        
    返回:
        tuple: 包含两个元素:
            - wind_data (numpy.ndarray): 风场数据，形状为 (时间步数, 2, 高, 宽)
            - wind_info (dict): 风场信息字典，包含经纬度坐标等内容
                
    异常:
        - 如果文件加载失败，打印错误信息并返回 None, None
        
    注意:
        - 该函数会检查文件是否存在并打印加载的数据形状，便于调试
    """
    try:
        # 加载风场数据
        wind_data = np.load(wind_path)
        print(f"成功加载风场数据，形状: {wind_data.shape}")
        
        # 加载风场经纬度坐标
        wind_info = {}
        if os.path.exists(wind_lonlat_path):
            wind_info['lonlat'] = np.load(wind_lonlat_path)
            print(f"成功加载风场经纬度坐标，形状: {wind_info['lonlat'].shape}")
        
        return wind_data, wind_info
    
    except Exception as e:
        print(f"加载风场数据失败: {e}")
        return None, None

def load_river_flow_data(flow_path="/home/lirz6/PROGRAM/NEWDC2/data/raw/flow.npy"):
    """
    加载河流径流量时序数据
    
    该函数负责加载河流径流量的时间序列数据，用作河口盐度场预测的输入条件之一。
    径流量数据表示为多条河流的时间序列流量数据。
    
    参数:
        flow_path (str): 径流量数据文件路径，形状为 (时间步数, 河流数)
                         包含每个时间步各条河流的流量数据
        
    返回:
        numpy.ndarray: 径流量数据，形状为 (时间步数, 河流数)
        
    异常:
        - 如果文件加载失败，打印错误信息并返回 None
        
    注意:
        - 该函数会打印加载的数据形状，便于调试
        - 径流量数据是影响河口盐度分布的重要因素之一
    """
    try:
        # 加载径流量数据
        flow_data = np.load(flow_path)
        print(f"成功加载径流量数据，形状: {flow_data.shape}")
        
        return flow_data
    
    except Exception as e:
        print(f"加载径流量数据失败: {e}")
        return None