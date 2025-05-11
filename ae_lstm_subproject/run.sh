#!/usr/bin/env bash
# 脚本启动 AE-LSTM 子项目
cd "$(dirname "$0")"
# 运行主流水线，指定配置目录和配置文件名称
python3 ../src/pipeline_ae_lstm.py --config-path conf --config-name config "$@"