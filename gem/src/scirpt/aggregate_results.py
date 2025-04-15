import os
import glob
import argparse
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def aggregate_results(multirun_dir: str, output_csv: str):
    """
    聚合 Hydra 多重运行的评估结果。

    Args:
        multirun_dir: Hydra 多重运行的主输出目录
                      (例如 'outputs/YYYY-MM-DD/HH-MM-SS_multirun')
        output_csv: 保存聚合结果的 CSV 文件路径。
    """
    all_results = []
    run_dirs = sorted(glob.glob(os.path.join(multirun_dir, '[0-9]*'))) # 查找数字命名的子目录

    if not run_dirs:
        logging.warning(f"在 '{multirun_dir}' 中未找到任何运行子目录 (例如 '0', '1', ...)。")
        return

    logging.info(f"找到 {len(run_dirs)} 个运行目录，开始聚合...")

    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            continue

        run_index = os.path.basename(run_dir)
        logging.debug(f"处理运行目录: {run_dir}")

        # --- 1. 加载该运行的配置 ---
        config_path = os.path.join(run_dir, '.hydra', 'config.yaml')
        if not os.path.exists(config_path):
            logging.warning(f"配置文件未找到: {config_path}，跳过运行 {run_index}")
            continue

        try:
            cfg = OmegaConf.load(config_path)
            # 提取你关心的变化参数 (确保路径正确)
            map_size = str(list(cfg.training.som.map_size)) # 转为字符串以便存储
            map_size_obs = str(list(cfg.training.som.map_size_obs)) # 转为字符串
            dr_method = cfg.model.dimensionality_reduction.method
            pred_method = cfg.model.prediction.method
        except Exception as e:
            logging.error(f"加载或解析配置 {config_path} 时出错: {e}，跳过运行 {run_index}")
            continue

        # --- 2. 查找并加载评估结果 ---
        eval_base_dir = os.path.join(run_dir, 'evaluation', f"{dr_method}_{pred_method}")
        if not os.path.isdir(eval_base_dir):
             logging.debug(f"评估目录未找到: {eval_base_dir}，跳过运行 {run_index} 的评估部分")
             continue

        for split in ['train', 'val', 'test']:
            metrics_path = os.path.join(eval_base_dir, split, f"metrics_{split}.npy")
            rmse_map_path = os.path.join(eval_base_dir, split, f"rmse_map_{split}.npy") # 可选

            if os.path.exists(metrics_path):
                try:
                    metrics = np.load(metrics_path, allow_pickle=True).item()
                    logging.debug(f"  加载指标: {metrics_path}")

                    # 准备要添加到列表的数据行
                    result_row = {
                        'run_index': run_index,
                        'map_size': map_size,
                        'map_size_obs': map_size_obs,
                        'dr_method': dr_method,
                        'pred_method': pred_method,
                        'split': split,
                        'mean_rmse': metrics.get('mean_rmse', np.nan),
                        'mean_mae': metrics.get('mean_mae', np.nan),
                        'max_rmse': metrics.get('max_rmse', np.nan), # 从之前的代码看，这个可能在 rmse_map 里
                        'min_rmse': metrics.get('min_rmse', np.nan), # 同上
                        # 可以添加其他你想记录的指标
                        'metrics_path': metrics_path # 记录路径以便追溯
                    }

                    # 如果需要，可以加载并计算 RMSE Map 的统计信息
                    if os.path.exists(rmse_map_path):
                         try:
                             rmse_map = np.load(rmse_map_path)
                             result_row['rmse_map_mean'] = float(np.nanmean(rmse_map)) if np.any(np.isfinite(rmse_map)) else np.nan
                             result_row['rmse_map_max'] = float(np.nanmax(rmse_map)) if np.any(np.isfinite(rmse_map)) else np.nan
                             result_row['rmse_map_min'] = float(np.nanmin(rmse_map)) if np.any(np.isfinite(rmse_map)) else np.nan
                         except Exception as e_map:
                             logging.warning(f"加载或处理 RMSE Map {rmse_map_path} 时出错: {e_map}")


                    all_results.append(result_row)

                except Exception as e:
                    logging.error(f"加载或处理指标文件 {metrics_path} 时出错: {e}")
            else:
                logging.debug(f"  指标文件未找到: {metrics_path}")

    # --- 3. 创建并保存 DataFrame ---
    if not all_results:
        logging.warning("未收集到任何结果。")
        return

    df = pd.DataFrame(all_results)
    # 可以根据需要重新排序列
    cols_order = ['run_index', 'map_size', 'map_size_obs', 'dr_method', 'pred_method', 'split',
                  'mean_rmse', 'mean_mae', 'rmse_map_mean', 'rmse_map_max', 'rmse_map_min', # 如果计算了 map 统计
                  'max_rmse', 'min_rmse', # 如果这些在 metrics.npy 里
                  'metrics_path']
    # 过滤掉 DataFrame 中不存在的列名
    cols_order = [col for col in cols_order if col in df.columns]
    df = df[cols_order]

    try:
        df.to_csv(output_csv, index=False)
        logging.info(f"聚合结果已成功保存到: {output_csv}")
        print(f"\n聚合结果预览:\n{df.head()}")
    except Exception as e:
        logging.error(f"保存结果到 CSV 文件 {output_csv} 时出错: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="聚合 Hydra 多重运行评估结果")
    parser.add_argument("multirun_dir", type=str, help="Hydra 多重运行的主输出目录 (例如 'outputs/YYYY-MM-DD/HH-MM-SS_multirun')")
    parser.add_argument("-o", "--output", type=str, default="aggregated_results.csv", help="输出 CSV 文件的路径 (默认: aggregated_results.csv)")
    args = parser.parse_args()

    aggregate_results(args.multirun_dir, args.output)