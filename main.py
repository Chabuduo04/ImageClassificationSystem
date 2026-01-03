import re
import pandas as pd
from collections import defaultdict
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore')


def parse_training_log(log_text, y_true=None, y_pred=None):
    """
    解析训练日志文本
    Args:
        log_text: 训练日志文本
        y_true: 真实标签（可选，用于计算指标）
        y_pred: 预测标签（可选，用于计算指标）
    Returns:
        DataFrame: 包含所有epoch结果的DataFrame
    """

    # 定义正则表达式匹配模式
    pattern = r'Epoch \[(\d+)/(\d+)\]\s+Train Loss:\s+([\d.]+)\s*\|\s*Train Acc:\s+([\d.]+)\s+Val Acc:\s+([\d.]+)'

    # 存储解析结果
    results = defaultdict(list)

    # 逐行解析日志
    lines = log_text.strip().split('\n')
    for line in lines:
        match = re.search(pattern, line)
        if match:
            epoch = int(match.group(1))
            total_epochs = int(match.group(2))
            train_loss = float(match.group(3))
            train_acc = float(match.group(4))
            val_acc = float(match.group(5))

            results['Epoch'].append(epoch)
            results['Total_Epochs'].append(total_epochs)
            results['Train_Loss'].append(train_loss)
            results['Train_Accuracy'].append(train_acc)
            results['Val_Accuracy'].append(val_acc)

    # 创建DataFrame
    df = pd.DataFrame(results)

    # 如果有真实标签和预测标签，计算更多指标
    if y_true is not None and y_pred is not None:
        # 转换为numpy数组
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # 计算各项指标
        try:
            # 对于多分类，使用macro平均
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            accuracy = accuracy_score(y_true, y_pred)

            # 为所有行添加这些指标
            df['Precision'] = precision
            df['Recall'] = recall
            df['F1_Score'] = f1
            df['Final_Accuracy'] = accuracy
        except Exception as e:
            print(f"计算指标时出错: {e}")
            # 添加空值
            df['Precision'] = np.nan
            df['Recall'] = np.nan
            df['F1_Score'] = np.nan
            df['Final_Accuracy'] = np.nan

    return df


def save_to_excel(df, filename='training_results.xlsx'):
    """
    将结果保存到Excel文件
    Args:
        df: 包含结果的DataFrame
        filename: Excel文件名
    """
    # 创建Excel写入器
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # 将主要结果保存到第一个sheet
        df.to_excel(writer, sheet_name='Training_Results', index=False)

        # 创建一个总结sheet
        summary_data = {}

        # 计算最佳epoch
        if not df.empty:
            best_epoch = df.loc[df['Val_Accuracy'].idxmax()]
            summary_data['Best_Epoch'] = [int(best_epoch['Epoch'])]
            summary_data['Best_Val_Accuracy'] = [best_epoch['Val_Accuracy']]
            summary_data['Best_Train_Accuracy'] = [best_epoch['Train_Accuracy']]
            summary_data['Best_Train_Loss'] = [best_epoch['Train_Loss']]

            # 最终结果
            if 'Final_Accuracy' in df.columns and not df['Final_Accuracy'].isna().all():
                summary_data['Final_Accuracy'] = [df['Final_Accuracy'].iloc[0]]
                summary_data['Precision'] = [df['Precision'].iloc[0]]
                summary_data['Recall'] = [df['Recall'].iloc[0]]
                summary_data['F1_Score'] = [df['F1_Score'].iloc[0]]

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # 设置列宽
        for column in df:
            column_width = max(df[column].astype(str).map(len).max(), len(column)) + 2
            col_idx = df.columns.get_loc(column)
            writer.sheets['Training_Results'].column_dimensions[chr(65 + col_idx)].width = column_width

    print(f"结果已保存到: {filename}")


def process_log_file(log_file_path, y_true=None, y_pred=None, output_file='training_results.xlsx'):
    """
    处理日志文件
    Args:
        log_file_path: 日志文件路径
        y_true: 真实标签文件路径或数组
        y_pred: 预测标签文件路径或数组
        output_file: 输出Excel文件名
    """
    # 读取日志文件
    with open(log_file_path, 'r', encoding='utf-8') as f:
        log_text = f.read()

    # 如果提供了文件路径，读取标签
    if isinstance(y_true, str):
        y_true = pd.read_csv(y_true) if y_true.endswith('.csv') else np.load(y_true)
    if isinstance(y_pred, str):
        y_pred = pd.read_csv(y_pred) if y_pred.endswith('.csv') else np.load(y_pred)

    # 解析日志
    print("正在解析训练日志...")
    df = parse_training_log(log_text, y_true, y_pred)

    # 显示前几行
    print("\n解析结果:")
    print(df.head())

    # 保存到Excel
    save_to_excel(df, output_file)

    return df


# 示例使用方式1：直接从日志文本处理
if __name__ == "__main__":
    df = process_log_file('training_log.txt', output_file='results.xlsx')