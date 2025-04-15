import os
import sys
import hydra
from omegaconf import DictConfig

# Add the project root directory to Python path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Use absolute imports instead of relative imports
from src.data_processing.data_validation import load_salinity_data, load_wind_data, load_river_flow_data
from src.data_processing.data_preprocessing import process_and_save_data
from src.data_processing.data_pair_creation import process_all_pairs
from src.utils.hydra_config import DrprConfig

def preprocess_raw_data(raw_data_dir, processed_data_dir, cfg=None):
    """
    处理原始数据并保存为处理后的数据
    
    Args:
        raw_data_dir: 原始数据目录
        processed_data_dir: 处理后数据保存目录
        cfg: 配置对象，用于获取数据分割比例
    """
    # 创建处理后数据存放目录
    os.makedirs(processed_data_dir, exist_ok=True)

    # 检查是否已有处理好的数据
    try:
        processed_files_exist = all([
            os.path.exists(os.path.join(processed_data_dir, file))
            for file in [
                "train_salinity_processed.npy", 
                "val_salinity_processed.npy", 
                "test_salinity_processed.npy",
                "salinity_scaler.npy", 
                "wind_scaler.npy", 
                "flow_scaler.npy"
            ]
        ])
        
        if processed_files_exist:
            print("检测到已有处理好的数据，跳过数据处理阶段")
        else:
            print("未检测到已处理的数据，开始数据处理...")
            process_and_save_data(raw_data_dir, processed_data_dir, cfg)
    except Exception as e:
        print(f"检查处理数据时出错: {e}")
        print("开始重新处理数据...")
        process_and_save_data(raw_data_dir, processed_data_dir, cfg)

def create_input_output_pairs(processed_data_dir, input_seq_length=6, output_seq_length=1, stride=5):
    """
    创建输入-输出序列对
    
    Args:
        processed_data_dir: 处理后数据目录
        input_seq_length: 输入序列长度
        output_seq_length: 输出序列长度
        stride: 滑动步长
    """
    # 创建一个包含参数信息的子文件夹
    pairs_subfolder = f"seq_in{input_seq_length}_out{output_seq_length}_stride{stride}"
    pairs_output_dir = os.path.join(processed_data_dir, pairs_subfolder)
    os.makedirs(pairs_output_dir, exist_ok=True)
    
    print(f"使用参数: 输入序列长度={input_seq_length}, 输出序列长度={output_seq_length}, 步长={stride}")
    print(f"将数据对保存到: {pairs_output_dir}")
    
    # 检查是否已有输入-输出对
    try:
        pairs_exist = all([
            os.path.exists(os.path.join(pairs_output_dir, file))
            for file in [
                "X_train.npy", "y_train.npy",
                "X_val.npy", "y_val.npy",
                "X_test.npy", "y_test.npy"
            ]
        ])
        
        if pairs_exist:
            print(f"检测到已有输入-输出对（{pairs_subfolder}），跳过输入-输出对创建阶段")
        else:
            print(f"未检测到输入-输出对（{pairs_subfolder}），开始创建输入-输出对...")
            process_all_pairs(processed_data_dir, pairs_output_dir, 
                             input_seq_length=input_seq_length, 
                             output_seq_length=output_seq_length,
                             stride=stride)
    except Exception as e:
        print(f"检查输入-输出对时出错: {e}")
        print(f"开始重新创建输入-输出对（{pairs_subfolder}）...")
        process_all_pairs(processed_data_dir, pairs_output_dir, 
                         input_seq_length=input_seq_length, 
                         output_seq_length=output_seq_length,
                         stride=stride)
    
    return pairs_output_dir

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    """
    主函数，执行完整的数据处理流程
    """
    config = DrprConfig.from_hydra_config(cfg)
    
    # 数据路径
    raw_data_dir = config.paths.raw_data_dir
    processed_data_dir = config.paths.processed_data_dir
    
    print(f"使用配置的数据路径:")
    print(f"  原始数据目录: {raw_data_dir}")
    print(f"  处理后数据目录: {processed_data_dir}")
    
    # 步骤1: 处理原始数据，传递配置对象
    preprocess_raw_data(raw_data_dir, processed_data_dir, cfg)
    
    # # 步骤2: 创建输入-输出序列对
    # input_seq_length = 6
    # output_seq_length = 1
    # stride = 5
    # create_input_output_pairs(processed_data_dir, input_seq_length, output_seq_length, stride)

if __name__ == "__main__":
    main()

