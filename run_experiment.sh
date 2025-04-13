#!/bin/bash

export PROJECT_ROOT=$(pwd) # 假设你总是在项目根目录运行此脚本
echo "PROJECT_ROOT set to: $PROJECT_ROOT" # 添加日志确认
# ... rest of your script ...

# 激活Python环境（如果需要）
# source /path/to/your/env/bin/activate

# 设置PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 1. 运行完整流程
echo "开始运行实验流程..."
# python src/run_pipeline.py
python src/run_som_lstm_bmu.py

# 检查运行状态
if [ $? -eq 0 ]; then
    echo "实验流程完毕"
    echo "结果保存在 outputs/ 目录下最新的时间戳文件夹中"
else
    echo "实验流程执行失败，请检查日志文件"
    echo "请查看 outputs/ 目录下的日志文件以获取更多信息"
    exit 1
fi