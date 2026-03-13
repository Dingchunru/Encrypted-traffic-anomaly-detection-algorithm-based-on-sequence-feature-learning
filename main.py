#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USTC-TFC2016 模型使用工具 - 批量预测版本
"""

import os
import numpy as np
import torch
import torch.nn as nn
import pickle
from collections import defaultdict, Counter
import time
import warnings
import random
warnings.filterwarnings('ignore')

# ========== 重新定义Config类（与训练时一致） ==========
class Config:
    """模型配置（必须与训练时完全一致）"""
    # 数据路径
    DATA_FILE = r"E:\USTC\preprocess\ustc_processed_data.pkl"
    MODEL_SAVE_DIR = r"E:\USTC\models"
    RESULT_DIR = r"E:\USTC\results"
    
    # 模型参数
    MODEL_TYPE = 'lstm'
    INPUT_SIZE = 3
    SEQUENCE_LENGTH = 100
    NUM_CLASSES = 20
    
    # LSTM参数
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.5
    
    # 设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 类别名称映射
    CLASS_NAMES = {
        0: 'BitTorrent', 1: 'Facetime', 2: 'FTP', 3: 'Gmail', 4: 'MySQL',
        5: 'Outlook', 6: 'Skype', 7: 'SMB', 8: 'Weibo', 9: 'WorldOfWarcraft',
        10: 'Cridex', 11: 'Geodo', 12: 'Htbot', 13: 'Miuref', 14: 'Neris',
        15: 'Nsis-ay', 16: 'Shifu', 17: 'Tinba', 18: 'Virut', 19: 'Zeus'
    }
    
    # 类别类型
    CLASS_TYPES = {
        0: 'benign', 1: 'benign', 2: 'benign', 3: 'benign', 4: 'benign',
        5: 'benign', 6: 'benign', 7: 'benign', 8: 'benign', 9: 'benign',
        10: 'malware', 11: 'malware', 12: 'malware', 13: 'malware', 14: 'malware',
        15: 'malware', 16: 'malware', 17: 'malware', 18: 'malware', 19: 'malware'
    }


# ========== 模型定义（必须与训练时一致） ==========
class LSTMModel(nn.Module):
    """LSTM模型 - 与训练时完全一致"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(128)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        
        hidden = self.fc1(last_output)
        hidden = self.bn(hidden)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)
        
        output = self.fc2(hidden)
        return output


# ========== 流量预测器 ==========
class TrafficPredictor:
    """流量预测器 - 加载模型进行预测"""
    
    def __init__(self, model_path, config=None):
        """
        初始化预测器
        Args:
            model_path: 训练好的模型路径 (.pth文件)
            config: 配置对象（可选）
        """
        if config is None:
            self.config = Config()
        else:
            self.config = config
            
        self.model = self._load_model(model_path)
        self.model.eval()
        print(f"模型加载成功！使用设备: {self.config.DEVICE}")
        print(f"模型将流量分为 {self.config.NUM_CLASSES} 个类别")
        
    def _load_model(self, model_path):
        """加载训练好的模型"""
        # 创建模型实例
        model = LSTMModel(
            self.config.INPUT_SIZE,
            self.config.HIDDEN_SIZE,
            self.config.NUM_LAYERS,
            self.config.NUM_CLASSES,
            self.config.DROPOUT
        )
        
        # 加载checkpoint
        checkpoint = torch.load(model_path, map_location=self.config.DEVICE)
        
        # 尝试多种方式加载
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                try:
                    model.load_state_dict(checkpoint)
                except:
                    # 去除可能的前缀
                    new_state_dict = {}
                    for k, v in checkpoint.items():
                        if k.startswith('module.'):
                            new_state_dict[k[7:]] = v
                        else:
                            new_state_dict[k] = v
                    model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.config.DEVICE)
        return model
    
    def predict_batch(self, sequences):
        """
        批量预测
        Args:
            sequences: numpy array of shape (batch, 100, 3)
        Returns:
            predictions: 预测结果列表
        """
        # 转换为tensor
        if isinstance(sequences, np.ndarray):
            tensor = torch.FloatTensor(sequences).to(self.config.DEVICE)
        else:
            tensor = sequences.to(self.config.DEVICE)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted = outputs.argmax(dim=1)
        
        # 转换为numpy
        pred_classes = predicted.cpu().numpy()
        probs = probabilities.cpu().numpy()
        
        # 构建结果
        results = []
        for i in range(len(pred_classes)):
            # 获取top3
            top3_idx = np.argsort(probs[i])[-3:][::-1]
            top3 = [(self.config.CLASS_NAMES[idx], float(probs[i][idx])) for idx in top3_idx]
            
            results.append({
                'class_id': int(pred_classes[i]),
                'class_name': self.config.CLASS_NAMES[pred_classes[i]],
                'type': self.config.CLASS_TYPES[pred_classes[i]],
                'confidence': float(probs[i][pred_classes[i]]),
                'top3_predictions': top3
            })
        
        return results


# ========== 批量预测函数 ==========
def batch_prediction(num_samples=1000):
    """
    批量预测函数
    Args:
        num_samples: 要预测的样本数量
    """
    print("="*60)
    print(f"USTC-TFC2016 批量预测测试 (样本数: {num_samples})")
    print("="*60)
    
    # 1. 加载模型
    model_path = r"E:\USTC-TFC2016_organized\models\best_model_lstm.pth"
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        return
    
    predictor = TrafficPredictor(model_path)
    
    # 2. 加载数据
    data_file = r"E:\USTC\preprocessed_data\ustc_processed_data.pkl"
    
    if not os.path.exists(data_file):
        print(f"错误: 数据文件不存在 {data_file}")
        return
    
    print(f"\n加载数据文件: {data_file}")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    all_sequences = np.array(data['sequences'])
    all_labels = np.array(data['labels'])
    
    print(f"总样本数: {len(all_sequences)}")
    print(f"数据形状: {all_sequences.shape}")
    
    # 3. 随机选择样本
    print(f"\n随机选择 {num_samples} 个样本...")
    
    # 设置随机种子，确保可重复性
    random.seed(42)
    np.random.seed(42)
    
    # 随机选择索引
    indices = np.random.choice(len(all_sequences), num_samples, replace=False)
    
    test_sequences = all_sequences[indices]
    test_labels = all_labels[indices]
    
    # 统计真实标签分布
    true_distribution = Counter(test_labels)
    
    print("\n真实标签分布:")
    print("-" * 60)
    print(f"{'类别ID':<8} {'类别名称':<20} {'类型':<8} {'样本数':<10} {'占比':<10}")
    print("-" * 60)
    
    for label in sorted(true_distribution.keys()):
        count = true_distribution[label]
        percentage = count / num_samples * 100
        name = predictor.config.CLASS_NAMES[label]
        type_name = predictor.config.CLASS_TYPES[label]
        print(f"{label:<8} {name:<20} {type_name:<8} {count:<10} {percentage:.2f}%")
    
    # 4. 批量预测
    print(f"\n开始批量预测...")
    start_time = time.time()
    
    # 分批处理避免显存溢出
    batch_size = 128
    all_results = []
    
    for i in range(0, len(test_sequences), batch_size):
        batch_sequences = test_sequences[i:i+batch_size]
        batch_results = predictor.predict_batch(batch_sequences)
        all_results.extend(batch_results)
        
        if (i + batch_size) % 500 == 0:
            print(f"  已处理 {min(i+batch_size, num_samples)}/{num_samples} 样本...")
    
    elapsed_time = time.time() - start_time
    print(f"预测完成！耗时: {elapsed_time:.2f} 秒")
    print(f"平均每样本: {elapsed_time/num_samples*1000:.2f} 毫秒")
    
    # 5. 计算准确率
    correct = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    
    for i, result in enumerate(all_results):
        true_label = test_labels[i]
        pred_label = result['class_id']
        
        per_class_total[true_label] += 1
        
        if pred_label == true_label:
            correct += 1
            per_class_correct[true_label] += 1
    
    accuracy = correct / num_samples * 100
    
    # 6. 显示详细结果
    print("\n" + "="*60)
    print("预测结果统计")
    print("="*60)
    print(f"总体准确率: {accuracy:.2f}% ({correct}/{num_samples})")
    
    print("\n各类别准确率:")
    print("-" * 70)
    print(f"{'类别ID':<8} {'类别名称':<20} {'类型':<8} {'准确率':<10} {'正确/总数':<15}")
    print("-" * 70)
    
    for label in sorted(per_class_total.keys()):
        total = per_class_total[label]
        correct_count = per_class_correct[label]
        acc = correct_count / total * 100 if total > 0 else 0
        name = predictor.config.CLASS_NAMES[label]
        type_name = predictor.config.CLASS_TYPES[label]
        
        # 根据准确率显示颜色标记
        if acc >= 90:
            mark = "✅"
        elif acc >= 70:
            mark = "⚠️"
        else:
            mark = "❌"
        
        print(f"{label:<8} {name:<20} {type_name:<8} {acc:>6.2f}%    {mark}  {correct_count:>4}/{total:<4}")
    
    # 7. 混淆分析
    print("\n" + "="*60)
    print("混淆分析 (预测错误的样本)")
    print("="*60)
    
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    for i, result in enumerate(all_results):
        true_label = test_labels[i]
        pred_label = result['class_id']
        
        if true_label != pred_label:
            confusion_matrix[true_label][pred_label] += 1
    
    # 显示最常见的混淆对
    print("\n最常见的混淆对 (真实 → 预测):")
    confusion_pairs = []
    for true_label, pred_dict in confusion_matrix.items():
        for pred_label, count in pred_dict.items():
            confusion_pairs.append((count, true_label, pred_label))
    
    confusion_pairs.sort(reverse=True)
    
    for count, true_label, pred_label in confusion_pairs[:10]:
        true_name = predictor.config.CLASS_NAMES[true_label]
        pred_name = predictor.config.CLASS_NAMES[pred_label]
        print(f"  {true_name:20} → {pred_name:20} : {count:3} 次")
    
    # 8. 置信度分析
    print("\n" + "="*60)
    print("置信度分析")
    print("="*60)
    
    confidences = [r['confidence'] for r in all_results]
    
    print(f"平均置信度: {np.mean(confidences):.2%}")
    print(f"中位数置信度: {np.median(confidences):.2%}")
    print(f"最小置信度: {np.min(confidences):.2%}")
    print(f"最大置信度: {np.max(confidences):.2%}")
    
    # 置信度分布
    bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    hist, edges = np.histogram(confidences, bins=bins)
    
    print("\n置信度分布:")
    for i in range(len(hist)):
        if hist[i] > 0:
            print(f"  {edges[i]:.0%} - {edges[i+1]:.0%}: {hist[i]:4d} 样本 ({hist[i]/num_samples*100:5.2f}%)")
    
    # 9. 保存结果
    result_file = r"E:\USTC\results\batch_prediction_results.txt"
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"USTC-TFC2016 批量预测结果\n")
        f.write(f"样本数量: {num_samples}\n")
        f.write(f"总体准确率: {accuracy:.2f}%\n\n")
        
        f.write("各类别准确率:\n")
        for label in sorted(per_class_total.keys()):
            total = per_class_total[label]
            correct_count = per_class_correct[label]
            acc = correct_count / total * 100 if total > 0 else 0
            name = predictor.config.CLASS_NAMES[label]
            f.write(f"  {name}: {acc:.2f}% ({correct_count}/{total})\n")
    
    print(f"\n详细结果已保存到: {result_file}")
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': num_samples,
        'per_class_accuracy': {predictor.config.CLASS_NAMES[label]: per_class_correct[label]/per_class_total[label]*100 
                               for label in per_class_total.keys()},
        'avg_confidence': np.mean(confidences)
    }


def main():
    """主函数"""
    print("="*60)
    print("USTC-TFC2016 模型批量预测工具")
    print("="*60)
    
    batch_prediction(5000)



if __name__ == "__main__":
    main()