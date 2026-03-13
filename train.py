#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USTC-TFC2016 流量分类模型训练脚本 - 最终修复版
"""

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from pathlib import Path
import warnings
from tqdm import tqdm
import json
from collections import Counter
import argparse
import gc

warnings.filterwarnings('ignore')

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ==================== 设备配置 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 如果使用CUDA，设置调试模式
if device.type == 'cuda':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ==================== 标签重映射 ====================

class LabelMapper:
    """标签映射器：将原始标签映射到连续的0~n-1"""
    
    def __init__(self):
        self.original_to_continuous = {}
        self.continuous_to_original = {}
        self.class_names = {}
        
    def fit(self, labels):
        """根据标签创建映射"""
        unique_labels = sorted(np.unique(labels))
        self.original_to_continuous = {orig: i for i, orig in enumerate(unique_labels)}
        self.continuous_to_original = {i: orig for i, orig in enumerate(unique_labels)}
        print(f"标签映射: {self.original_to_continuous}")
        return self
    
    def transform(self, labels):
        """转换标签到连续值"""
        if isinstance(labels, (list, np.ndarray)):
            return np.array([self.original_to_continuous[l] for l in labels])
        return self.original_to_continuous[labels]
    
    def inverse_transform(self, labels):
        """转换回原始标签"""
        if isinstance(labels, (list, np.ndarray)):
            return np.array([self.continuous_to_original[l] for l in labels])
        return self.continuous_to_original[labels]
    
    @property
    def num_classes(self):
        return len(self.original_to_continuous)

# ==================== 内存优化的数据加载 ====================

class MemoryEfficientUSTCDataset(Dataset):
    """内存高效的USTC数据集类"""
    
    def __init__(self, data_dir, indices=None, label_mapper=None, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.label_mapper = label_mapper
        
        # 使用mmap模式加载
        seq_file = self.data_dir / "ustc_sequences.npy"
        label_file = self.data_dir / "ustc_labels.npy"
        
        print(f"加载序列文件: {seq_file}")
        self.sequences = np.load(seq_file, mmap_mode='r')
        self.original_labels = np.load(label_file)
        
        # 加载类别名称
        data_file = self.data_dir / "ustc_complete_data.pkl"
        if data_file.exists():
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
                self.class_names = data.get('class_names', {})
        else:
            self.class_names = {}
        
        # 应用标签映射
        if label_mapper is not None:
            self.labels = label_mapper.transform(self.original_labels)
        else:
            self.labels = self.original_labels
        
        # 设置索引
        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(len(self.labels))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        
        # 加载单个样本
        seq = np.array(self.sequences[actual_idx], copy=True)
        label = self.labels[actual_idx]
        
        # 转换为tensor
        seq = torch.FloatTensor(seq)
        label = torch.LongTensor([label])[0]
        
        if self.transform:
            seq = self.transform(seq)
        
        return seq, label

def create_data_indices(labels, test_size=0.2, val_size=0.1, random_state=42):
    """创建训练/验证/测试集的索引"""
    np.random.seed(random_state)
    n_samples = len(labels)
    indices = np.arange(n_samples)
    
    # 按类别分层采样
    unique_labels = np.unique(labels)
    train_indices = []
    val_indices = []
    test_indices = []
    
    for label in tqdm(unique_labels, desc="创建数据分割"):
        label_indices = indices[labels == label]
        n_label = len(label_indices)
        
        # 打乱
        np.random.shuffle(label_indices)
        
        # 计算分割点
        n_test = int(n_label * test_size)
        n_val = int(n_label * val_size)
        n_train = n_label - n_test - n_val
        
        # 分配
        train_indices.extend(label_indices[:n_train])
        val_indices.extend(label_indices[n_train:n_train + n_val])
        test_indices.extend(label_indices[n_train + n_val:])
    
    # 再次打乱
    for indices_list in [train_indices, val_indices, test_indices]:
        np.random.shuffle(indices_list)
    
    return np.array(train_indices), np.array(val_indices), np.array(test_indices)

def load_ustc_data_memory_efficient(data_dir, batch_size=32, test_size=0.2, val_size=0.1):
    """内存高效的数据加载"""
    print("="*60)
    print("加载USTC数据（内存优化模式）")
    print("="*60)
    
    # 首先加载原始标签
    label_file = Path(data_dir) / "ustc_labels.npy"
    original_labels = np.load(label_file)
    
    print(f"原始标签范围: {original_labels.min()} - {original_labels.max()}")
    print(f"原始标签分布: {Counter(original_labels)}")
    
    # 创建标签映射器
    label_mapper = LabelMapper()
    label_mapper.fit(original_labels)
    
    # 转换标签
    continuous_labels = label_mapper.transform(original_labels)
    
    print(f"\n映射后标签范围: 0 - {label_mapper.num_classes-1}")
    print(f"映射后标签分布: {Counter(continuous_labels)}")
    
    # 创建数据分割索引
    train_idx, val_idx, test_idx = create_data_indices(
        continuous_labels, test_size=test_size, val_size=val_size
    )
    
    print(f"\n数据分割完成:")
    print(f"  训练集: {len(train_idx)} 样本")
    print(f"  验证集: {len(val_idx)} 样本")
    print(f"  测试集: {len(test_idx)} 样本")
    
    # 创建数据集
    train_dataset = MemoryEfficientUSTCDataset(data_dir, train_idx, label_mapper)
    val_dataset = MemoryEfficientUSTCDataset(data_dir, val_idx, label_mapper)
    test_dataset = MemoryEfficientUSTCDataset(data_dir, test_idx, label_mapper)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 清理内存
    gc.collect()
    
    return train_loader, val_loader, test_loader, label_mapper, train_dataset

# ==================== 模型定义 ====================

class LSTMClassifier(nn.Module):
    """LSTM分类器"""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=17, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # 使用最后一个时间步的输出
        last_out = lstm_out[:, -1, :]
        return self.classifier(last_out)

class HybridCNN_LSTM(nn.Module):
    """CNN + LSTM 混合模型"""
    
    def __init__(self, input_dim, cnn_channels=64, lstm_hidden=128, num_classes=17):
        super(HybridCNN_LSTM, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        
        self.lstm = nn.LSTM(
            cnn_channels, lstm_hidden, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.3,
            num_layers=1
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(lstm_hidden, num_classes)
        )
        
    def forward(self, x):
        # CNN
        x_cnn = x.transpose(1, 2)
        cnn_out = self.cnn(x_cnn)
        cnn_out = cnn_out.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(cnn_out)
        last_out = lstm_out[:, -1, :]
        
        return self.classifier(last_out)

# ==================== 训练器类 ====================

class Trainer:
    """训练器类"""
    
    def __init__(self, model, device, model_name="model"):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def train_epoch(self, train_loader, criterion, optimizer):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"训练 {self.model_name}", leave=False)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            pbar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return total_loss / len(train_loader), 100. * correct / total
    
    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        return total_loss / len(val_loader), 100. * correct / total
    
    def train(self, train_loader, val_loader, epochs=20, lr=0.001, 
              weight_decay=1e-4, patience=8, class_weights=None):
        
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        best_val_acc = 0
        best_model_state = None
        patience_counter = 0
        
        print(f"\n开始训练 {self.model_name}...")
        
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            scheduler.step(val_loss)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"  → 新最佳模型! Val Acc: {val_acc:.2f}%")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\n早停触发! 最佳验证准确率: {best_val_acc:.2f}%")
                break
        
        self.model.load_state_dict(best_model_state)
        return self.model
    
    def evaluate(self, test_loader, label_mapper=None):
        """评估模型"""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="评估"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(all_targets, all_preds)
        f1_macro = f1_score(all_targets, all_preds, average='macro')
        f1_weighted = f1_score(all_targets, all_preds, average='weighted')
        
        print(f"\n{'='*60}")
        print(f"测试集评估结果 - {self.model_name}")
        print('='*60)
        print(f"准确率: {accuracy*100:.2f}%")
        print(f"Macro F1: {f1_macro:.4f}")
        print(f"Weighted F1: {f1_weighted:.4f}")
        
        # 如果有标签映射器，显示原始标签的分类报告
        if label_mapper:
            original_targets = label_mapper.inverse_transform(all_targets)
            original_preds = label_mapper.inverse_transform(all_preds)
            
            print("\n原始标签分类报告:")
            unique_orig = np.unique(original_targets)
            target_names = [f"Class-{l}" for l in unique_orig]
            print(classification_report(original_targets, original_preds, 
                                      target_names=target_names, digits=4))
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'predictions': all_preds,
            'targets': all_targets
        }
    
    def plot_training_history(self, save_path=None):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        axes[0].plot(epochs, self.train_losses, 'b-', label='训练损失')
        axes[0].plot(epochs, self.val_losses, 'r-', label='验证损失')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{self.model_name} - 损失曲线')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(epochs, self.train_accs, 'b-', label='训练准确率')
        axes[1].plot(epochs, self.val_accs, 'r-', label='验证准确率')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title(f'{self.model_name} - 准确率曲线')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练曲线已保存到: {save_path}")
        
        plt.show()

# ==================== 主训练函数 ====================

def train_models_memory_efficient(data_dir, output_dir, batch_size=16, epochs=20, device=device):
    """内存高效的模型训练"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 加载数据
    train_loader, val_loader, test_loader, label_mapper, train_dataset = load_ustc_data_memory_efficient(
        data_dir, batch_size=batch_size
    )
    
    # 获取模型参数
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].shape[2]
    num_classes = label_mapper.num_classes
    seq_len = sample_batch[0].shape[1]
    
    print(f"\n模型配置:")
    print(f"  输入维度: {input_dim}")
    print(f"  序列长度: {seq_len}")
    print(f"  类别数量: {num_classes}")
    
    # 计算类别权重（使用连续标签）
    labels = train_dataset.labels[train_dataset.indices]
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.FloatTensor(class_weights)
    print(f"类别权重: {class_weights}")
    
    # 定义要训练的模型
    models_to_train = {
        'LSTM': LSTMClassifier(input_dim, hidden_dim=128, num_layers=2, 
                               num_classes=num_classes, dropout=0.3),
        'Hybrid_CNN_LSTM': HybridCNN_LSTM(input_dim, num_classes=num_classes)
    }
    
    results = {}
    best_model = None
    best_accuracy = 0
    
    for model_name, model in models_to_train.items():
        print(f"\n{'='*60}")
        print(f"训练模型: {model_name}")
        print('='*60)
        
        # 计算模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {total_params:,}")
        
        # 创建训练器
        trainer = Trainer(model, device, model_name)
        
        # 训练
        trainer.train(
            train_loader, val_loader, 
            epochs=epochs, 
            lr=0.001,
            class_weights=class_weights,
            patience=8
        )
        
        # 绘制训练曲线
        trainer.plot_training_history(output_path / f"{model_name}_training_history.png")
        
        # 评估
        eval_results = trainer.evaluate(test_loader, label_mapper)
        results[model_name] = eval_results
        
        # 保存模型
        model_path = output_path / f"{model_name}_best.pth"
        torch.save({
            'model_state_dict': trainer.model.state_dict(),
            'model_config': {
                'name': model_name,
                'input_dim': input_dim,
                'num_classes': num_classes
            },
            'label_mapper': {
                'original_to_continuous': label_mapper.original_to_continuous,
                'continuous_to_original': label_mapper.continuous_to_original
            },
            'eval_results': eval_results
        }, model_path)
        print(f"模型已保存到: {model_path}")
        
        # 更新最佳模型
        if eval_results['accuracy'] > best_accuracy:
            best_accuracy = eval_results['accuracy']
            best_model = model_name
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # 比较结果
    print("\n" + "="*60)
    print("模型性能比较")
    print("="*60)
    
    comparison_data = []
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  准确率: {metrics['accuracy']*100:.2f}%")
        print(f"  Macro F1: {metrics['f1_macro']:.4f}")
        
        comparison_data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']*100:.2f}%",
            'Macro F1': f"{metrics['f1_macro']:.4f}",
            'Weighted F1': f"{metrics['f1_weighted']:.4f}"
        })
    
    # 保存比较结果
    df = pd.DataFrame(comparison_data)
    df.to_csv(output_path / "model_comparison.csv", index=False)
    print(f"\n比较结果已保存到: {output_path / 'model_comparison.csv'}")
    print(f"\n最佳模型: {best_model}  (准确率: {best_accuracy*100:.2f}%)")
    
    # 保存完整结果
    with open(output_path / "results.json", 'w') as f:
        serializable_results = {
            model_name: {
                'accuracy': float(metrics['accuracy']),
                'f1_macro': float(metrics['f1_macro']),
                'f1_weighted': float(metrics['f1_weighted'])
            }
            for model_name, metrics in results.items()
        }
        json.dump(serializable_results, f, indent=2)
    
    return results, best_model

# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='USTC-TFC2016 流量分类训练（最终修复版）')
    parser.add_argument('--data_dir', type=str,
                       default=r"E:\USTC\preprocessed_data",
                       help='数据目录')
    parser.add_argument('--output_dir', type=str,
                       default=r"E:\USTC\results",
                       help='输出目录')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=20,
                       help='训练轮数')
    
    args = parser.parse_args()
    
    # 获取全局device
    global device
    
    # 创建输出目录
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 保存配置
    config = {
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'device': str(device)
    }
    
    with open(output_path / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # 训练模型
    try:
        results, best_model = train_models_memory_efficient(
            args.data_dir,
            args.output_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            device=device
        )
        print(f"\n训练完成！最佳模型: {best_model}")
    except RuntimeError as e:
        if "CUDA error" in str(e):
            print("\nCUDA错误 detected. 尝试使用CPU运行...")
            # 重新设置device为CPU
            device = torch.device('cpu')
            print(f"切换到设备: {device}")
            
            # 重新训练
            results, best_model = train_models_memory_efficient(
                args.data_dir,
                args.output_dir,
                batch_size=args.batch_size,
                epochs=args.epochs,
                device=device
            )
            print(f"\n训练完成！最佳模型: {best_model}")
        else:
            raise e

if __name__ == "__main__":
    main()