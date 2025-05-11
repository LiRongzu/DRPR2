import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import logging
from sklearn.metrics import r2_score
from typing import Dict, List, Union, Optional

from .metrics import calculate_mse, calculate_mae

def calculate_reconstruction_error(original_data, reconstructed_data):
    """计算重构误差"""
    mse = calculate_mse(original_data, reconstructed_data)
    mae = calculate_mae(original_data, reconstructed_data)
    return mse, mae

def calculate_prediction_error(true_data, predicted_data):
    """计算预测误差"""
    mse = calculate_mse(true_data, predicted_data)
    mae = calculate_mae(true_data, predicted_data)
    return mse, mae

class DimensionalityReductionComparison:
    """
    降维方法比较类
    
    用于比较不同降维方法的性能，包括重构误差、压缩率等指标
    """
    
    def __init__(self):
        """初始化比较器"""
        self.models = {}
        self.reduced_data = {}
        self.reconstructed_data = {}
        self.results = {}
        self.original_data = None
    
    def add_method(self, method_name, model, original_data, reduced_data, reconstructed_data):
        """
        添加一个降维方法的结果
        
        参数:
            method_name: 方法名称
            model: 训练好的降维模型
            original_data: 原始数据
            reduced_data: 降维后的数据
            reconstructed_data: 重构后的数据
        """
        self.original_data = original_data
        self.models[method_name] = model
        self.reduced_data[method_name] = reduced_data
        self.reconstructed_data[method_name] = reconstructed_data
        
        # 计算评估指标
        mse, mae = calculate_reconstruction_error(original_data, reconstructed_data)
        r2 = r2_score(original_data.reshape(-1), reconstructed_data.reshape(-1))
        
        # 计算压缩率
        compression_ratio = original_data.size / reduced_data.size
        
        # 获取训练时间
        training_time = getattr(model, 'training_time', None)
        
        # 存储结果
        self.results[method_name] = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'compression_ratio': compression_ratio,
            'reduced_dim': reduced_data.shape[1],
            'training_time': training_time
        }
        
        # 对于特定的降维方法，添加额外的指标
        if hasattr(model, 'explained_variance_ratio'):
            self.results[method_name]['explained_variance'] = np.sum(model.explained_variance_ratio)
        
        logging.info(f"添加方法 {method_name} 的评估结果")
        logging.info(f"重构误差 (MSE): {mse:.6f}, (MAE): {mae:.6f}")
        logging.info(f"决定系数 (R²): {r2:.6f}")
        if training_time is not None:
            logging.info(f"训练时间: {training_time:.4f}秒")

    def plot_reconstruction_error_comparison(self, figsize=(10, 6)):
        """
        绘制不同方法的重构误差比较图
        
        参数:
            figsize: 图形大小
        """
        if not self.results:
            logging.warning("没有可用的结果")
            return
        
        plt.figure(figsize=figsize)
        
        methods = list(self.results.keys())
        mse_values = [self.results[method]['mse'] for method in methods]
        mae_values = [self.results[method]['mae'] for method in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        plt.bar(x - width/2, mse_values, width, label='MSE')
        plt.bar(x + width/2, mae_values, width, label='MAE')
        
        plt.xlabel('降维方法')
        plt.ylabel('误差值')
        plt.title('不同降维方法的重构误差比较')
        plt.xticks(x, methods)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
    
    def plot_error_vs_compression(self, figsize=(10, 6)):
        """
        绘制误差与压缩率的权衡图
        
        参数:
            figsize: 图形大小
        """
        if not self.results:
            logging.warning("没有可用的结果")
            return
        
        plt.figure(figsize=figsize)
        
        methods = list(self.results.keys())
        mse_values = [self.results[method]['mse'] for method in methods]
        compression_ratios = [self.results[method]['compression_ratio'] for method in methods]
        
        # 绘制散点图
        plt.scatter(compression_ratios, mse_values, s=100, alpha=0.7)
        
        # 添加方法标签
        for i, method in enumerate(methods):
            plt.annotate(method, (compression_ratios[i], mse_values[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('压缩率')
        plt.ylabel('均方误差 (MSE)')
        plt.title('误差与压缩率的权衡')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
    
    def get_results_table(self):
        """
        获取结果比较表
        
        返回:
            results_df: 包含所有方法评估指标的DataFrame
        """
        if not self.results:
            logging.warning("没有可用的结果")
            return None
        
        # 将结果转换为DataFrame
        results_df = pd.DataFrame(self.results).T
        
        # 重新排序列
        columns_order = ['mse', 'mae', 'r2', 'compression_ratio', 'reduced_dim', 'training_time']
        extra_columns = [col for col in results_df.columns if col not in columns_order]
        columns_order.extend(extra_columns)
        
        results_df = results_df[columns_order]
        
        # 重命名列
        column_names = {
            'mse': '均方误差',
            'mae': '平均绝对误差',
            'r2': '决定系数',
            'compression_ratio': '压缩率',
            'reduced_dim': '降维维度',
            'training_time': '训练时间(秒)'
        }
        results_df = results_df.rename(columns=column_names)
        
        return results_df

class PredictionModelComparison:
    """
    预测模型比较类
    
    用于比较不同预测方法的性能，包括预测误差、计算时间等指标
    """
    
    def __init__(self):
        """初始化比较器"""
        self.models = {}
        self.predictions = {}
        self.results = {}
        self.features = None
        self.true_values = None
    
    def add_method(self, method_name, model, input_features, true_data, predicted_data, prediction_time=None):
        """
        添加一个预测方法的结果
        
        参数:
            method_name: 预测方法名称
            model: 训练好的预测模型
            input_features: 输入特征
            true_data: 真实值
            predicted_data: 预测值
            prediction_time: 预测用时
        """
        self.features = input_features
        self.true_values = true_data
        self.models[method_name] = model
        self.predictions[method_name] = predicted_data
        
        # 计算评估指标
        mse, mae = calculate_prediction_error(true_data, predicted_data)
        r2 = r2_score(true_data.reshape(-1), predicted_data.reshape(-1))
        
        # 存储结果
        self.results[method_name] = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'prediction_time': prediction_time
        }
        
        logging.info(f"添加预测方法 {method_name} 的评估结果")
        logging.info(f"预测误差 (MSE): {mse:.6f}, (MAE): {mae:.6f}")
        logging.info(f"决定系数 (R²): {r2:.6f}")
        if prediction_time is not None:
            logging.info(f"预测时间: {prediction_time:.4f}秒")
    
    def plot_prediction_error_comparison(self, figsize=(10, 6)):
        """
        绘制不同方法的预测误差比较图
        
        参数:
            figsize: 图形大小
        """
        if not self.results:
            logging.warning("没有可用的结果")
            return
        
        plt.figure(figsize=figsize)
        
        methods = list(self.results.keys())
        mse_values = [self.results[method]['mse'] for method in methods]
        mae_values = [self.results[method]['mae'] for method in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        plt.bar(x - width/2, mse_values, width, label='MSE')
        plt.bar(x + width/2, mae_values, width, label='MAE')
        
        plt.xlabel('预测方法')
        plt.ylabel('误差值')
        plt.title('不同预测方法的误差比较')
        plt.xticks(x, methods)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
    def plot_time_comparison(self, figsize=(10, 6)):
        """
        绘制不同方法的计算时间比较图
        
        参数:
            figsize: 图形大小
        """
        if not self.results:
            logging.warning("没有可用的结果")
            return
            
        # 过滤掉没有时间信息的方法
        methods = []
        times = []
        for method in self.results:
            if self.results[method]['prediction_time'] is not None:
                methods.append(method)
                times.append(self.results[method]['prediction_time'])
                
        if not methods:
            logging.warning("没有可用的时间数据")
            return
        
        plt.figure(figsize=figsize)
        plt.bar(methods, times, color='lightgreen')
        plt.xlabel('预测方法')
        plt.ylabel('预测时间 (秒)')
        plt.title('不同预测方法的计算时间比较')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
    def plot_all_comparisons(self, figsize=(15, 10)):
        """
        绘制所有比较图表
        
        参数:
            figsize: 图形大小
        """
        if not self.results:
            logging.warning("没有可用的结果")
            return
        
        plt.figure(figsize=figsize)
        
        # 预测误差比较
        plt.subplot(2, 2, 1)
        methods = list(self.results.keys())
        mse_values = [self.results[method]['mse'] for method in methods]
        mae_values = [self.results[method]['mae'] for method in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        plt.bar(x - width/2, mse_values, width, label='MSE')
        plt.bar(x + width/2, mae_values, width, label='MAE')
        
        plt.xlabel('预测方法')
        plt.ylabel('误差值')
        plt.title('预测误差比较')
        plt.xticks(x, methods)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 决定系数比较
        plt.subplot(2, 2, 2)
        r2_values = [self.results[method]['r2'] for method in methods]
        
        plt.bar(methods, r2_values, color='skyblue')
        plt.xlabel('预测方法')
        plt.ylabel('决定系数 (R²)')
        plt.title('决定系数比较')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 预测时间比较
        plt.subplot(2, 2, 3)
        # 过滤掉没有预测时间的方法
        time_methods = []
        times = []
        for method in methods:
            if self.results[method]['prediction_time'] is not None:
                time_methods.append(method)
                times.append(self.results[method]['prediction_time'])
        
        if time_methods:
            plt.bar(time_methods, times, color='lightgreen')
            plt.xlabel('预测方法')
            plt.ylabel('预测时间 (秒)')
            plt.title('计算时间比较')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
        else:
            plt.text(0.5, 0.5, '没有可用的时间数据', ha='center', va='center')
            plt.axis('off')
        
        plt.tight_layout()
        
    def get_results_table(self):
        """
        获取结果比较表
        
        返回:
            results_df: 包含所有方法评估指标的DataFrame
        """
        if not self.results:
            logging.warning("没有可用的结果")
            return None
        
        # 将结果转换为DataFrame
        results_df = pd.DataFrame(self.results).T
        
        # 重命名列
        column_names = {
            'mse': '均方误差',
            'mae': '平均绝对误差',
            'r2': '决定系数',
            'prediction_time': '预测时间(秒)'
        }
        results_df = results_df.rename(columns=column_names)
        
        return results_df