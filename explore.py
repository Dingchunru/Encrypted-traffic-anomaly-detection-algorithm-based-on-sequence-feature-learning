#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USTC-TFC2016 数据探索脚本 - 交互式版本
一张一张展示图表，关闭一张展示下一张
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
from pathlib import Path
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import warnings
import time
warnings.filterwarnings('ignore')

# ========== 美化配置 ==========
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 设置全局字体和样式
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# 颜色方案
COLORS = {
    'benign': '#2ecc71',      # 翠绿
    'malware': '#e74c3c',      # 红色
    'primary': '#3498db',      # 蓝色
    'secondary': '#f39c12',    # 橙色
    'accent': '#9b59b6',       # 紫色
    'background': '#f8f9fa',   # 浅灰
    'text': '#2c3e50',         # 深灰
    'grid': '#bdc3c7'          # 中灰
}

# ========== USTC-TFC2016 类别名称映射 ==========
CLASS_NAMES = {
    # 良性应用 (0-9)
    0: {'en': 'BitTorrent', 'zh': '比特彗星', 'type': 'benign', 'emoji': '📥'},
    1: {'en': 'Facetime', 'zh': 'Facetime', 'type': 'benign', 'emoji': '📱'},
    2: {'en': 'FTP', 'zh': '文件传输', 'type': 'benign', 'emoji': '📂'},
    3: {'en': 'Gmail', 'zh': 'Gmail', 'type': 'benign', 'emoji': '📧'},
    4: {'en': 'MySQL', 'zh': 'MySQL数据库', 'type': 'benign', 'emoji': '🗄️'},
    5: {'en': 'Outlook', 'zh': 'Outlook邮件', 'type': 'benign', 'emoji': '📨'},
    6: {'en': 'Skype', 'zh': 'Skype', 'type': 'benign', 'emoji': '💬'},
    7: {'en': 'SMB', 'zh': '文件共享', 'type': 'benign', 'emoji': '🖥️'},
    8: {'en': 'Weibo', 'zh': '微博', 'type': 'benign', 'emoji': '🐦'},
    9: {'en': 'WorldOfWarcraft', 'zh': '魔兽世界', 'type': 'benign', 'emoji': '🎮'},
    
    # 恶意软件 (10-19)
    10: {'en': 'Cridex', 'zh': 'Cridex木马', 'type': 'malware', 'emoji': '🦠'},
    11: {'en': 'Geodo', 'zh': 'Geodo木马', 'type': 'malware', 'emoji': '🦠'},
    12: {'en': 'Htbot', 'zh': 'Htbot僵尸网络', 'type': 'malware', 'emoji': '🤖'},
    13: {'en': 'Miuref', 'zh': 'Miuref后门', 'type': 'malware', 'emoji': '🚪'},
    14: {'en': 'Neris', 'zh': 'Neris僵尸网络', 'type': 'malware', 'emoji': '🤖'},
    15: {'en': 'Nsis-ay', 'zh': 'Nsis-ay下载器', 'type': 'malware', 'emoji': '⬇️'},
    16: {'en': 'Shifu', 'zh': 'Shifu银行木马', 'type': 'malware', 'emoji': '🏦'},
    17: {'en': 'Tinba', 'zh': 'Tinba银行木马', 'type': 'malware', 'emoji': '🏦'},
    18: {'en': 'Virut', 'zh': 'Virut病毒', 'type': 'malware', 'emoji': '🦠'},
    19: {'en': 'Zeus', 'zh': 'Zeus银行木马', 'type': 'malware', 'emoji': '🏦'},
}

# 中英文切换
USE_ENGLISH = False

def get_class_display(label):
    """获取类别显示信息"""
    label = int(label)
    if label in CLASS_NAMES:
        info = CLASS_NAMES[label]
        if USE_ENGLISH:
            return f"{info['emoji']} {info['en']}"
        else:
            return f"{info['emoji']} {info['zh']}"
    return f"❓ Unknown-{label}"

def get_class_color(label):
    """获取类别颜色"""
    label = int(label)
    if label in CLASS_NAMES:
        return COLORS[CLASS_NAMES[label]['type']]
    return COLORS['primary']

class USTCDataExplorer:
    """USTC数据探索器"""
    
    def __init__(self, data_dir, output_dir):
        """
        初始化探索器
        
        Args:
            data_dir: 数据目录（包含预处理后的文件）
            output_dir: 输出目录
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sequences = None
        self.labels = None
        self.class_names = None
        self.metadata = None
        self.config = None
        
        # 加载数据
        self.load_data()
        
        # 设置中文（如果需要）
        if not USE_ENGLISH:
            self.setup_chinese_font()
    
    def setup_chinese_font(self):
        """设置中文字体"""
        import platform
        system = platform.system()
        
        if system == "Windows":
            font_list = ['Microsoft YaHei', 'SimHei', 'KaiTi']
        elif system == "Linux":
            font_list = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC']
        elif system == "Darwin":
            font_list = ['Arial Unicode MS', 'Heiti SC']
        else:
            font_list = ['sans-serif']
        
        for font in font_list:
            try:
                plt.rcParams['font.sans-serif'] = [font] + plt.rcParams.get('font.sans-serif', [])
                print(f"使用字体: {font}")
                break
            except:
                continue
    
    def load_data(self):
        """加载预处理后的数据"""
        print("="*70)
        print("📊 USTC-TFC2016 数据探索 - 交互式版本")
        print("="*70)
        
        # 尝试加载pickle文件
        data_file = self.data_dir / "ustc_complete_data.pkl"
        
        if not data_file.exists():
            # 尝试加载numpy文件
            seq_file = self.data_dir / "ustc_sequences.npy"
            label_file = self.data_dir / "ustc_labels.npy"
            
            if seq_file.exists() and label_file.exists():
                print("📂 加载numpy格式数据...")
                self.sequences = np.load(seq_file, mmap_mode='r')
                self.labels = np.load(label_file)
                
                # 创建默认的类别名称
                unique_labels = np.unique(self.labels)
                self.class_names = {int(l): f"Class-{l}" for l in unique_labels}
                self.config = {'max_packets_per_flow': self.sequences.shape[1]}
            else:
                raise FileNotFoundError(f"❌ 未找到数据文件: {data_file}")
        else:
            print(f"📂 加载数据: {data_file}")
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            
            self.sequences = np.array(data['sequences'])
            self.labels = np.array(data['labels'])
            self.class_names = data.get('class_names', {})
            self.metadata = data.get('metadata', [])
            self.config = data.get('config', {})
        
        print(f"✅ 数据加载成功!")
        print(f"   📦 序列形状: {self.sequences.shape}")
        print(f"   🏷️  标签形状: {self.labels.shape}")
        print(f"   📊 数据类型: {self.sequences.dtype}")
        print(f"   🔢 标签范围: {self.labels.min()} - {self.labels.max()}")
        print(f"   🎯 类别数量: {len(np.unique(self.labels))}")
        
        if self.config:
            print(f"\n⚙️  配置信息:")
            for key, value in self.config.items():
                print(f"   {key}: {value}")
    
    def plot_data_overview(self):
        """绘制数据概览图"""
        print("\n📊 生成数据概览图...")
        
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], width_ratios=[1, 1])
        
        # 1. 类别分布
        ax1 = plt.subplot(gs[0, :])
        self._plot_class_distribution(ax1)
        
        # 2. 良性vs恶意对比
        ax2 = plt.subplot(gs[1, 0])
        self._plot_benign_malware_comparison(ax2)
        
        # 3. 流长度分布
        ax3 = plt.subplot(gs[1, 1])
        self._plot_flow_length_distribution(ax3)
        
        plt.suptitle('📊 USTC-TFC2016 数据集概览', fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # 保存图片
        output_path = self.output_dir / '01_ustc_data_overview.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✅ 已保存: {output_path.name}")
        
        # 显示图片
        print("   👆 图1: 数据概览 (关闭图片继续)")
        plt.show()
        plt.close(fig)
    
    def _plot_class_distribution(self, ax):
        """绘制类别分布"""
        counter = Counter(self.labels)
        labels_list = sorted(counter.keys())
        counts = [counter[l] for l in labels_list]
        names = [get_class_display(l) for l in labels_list]
        colors = [get_class_color(l) for l in labels_list]
        
        bars = ax.bar(range(len(labels_list)), counts, color=colors, alpha=0.8, 
                      edgecolor='white', linewidth=1)
        
        ax.set_xticks(range(len(labels_list)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('样本数量' if not USE_ENGLISH else 'Number of Samples', fontsize=12)
        ax.set_title('📊 各类别样本分布', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 添加图例
        legend_elements = [
            Patch(facecolor=COLORS['benign'], alpha=0.8, label='良性应用' if not USE_ENGLISH else 'Benign'),
            Patch(facecolor=COLORS['malware'], alpha=0.8, label='恶意软件' if not USE_ENGLISH else 'Malware')
        ]
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    
    def _plot_benign_malware_comparison(self, ax):
        """绘制良性vs恶意对比"""
        benign_count = sum(1 for l in self.labels if int(l) in CLASS_NAMES and CLASS_NAMES[int(l)]['type'] == 'benign')
        malware_count = sum(1 for l in self.labels if int(l) in CLASS_NAMES and CLASS_NAMES[int(l)]['type'] == 'malware')
        total = len(self.labels)
        
        # 饼图
        sizes = [benign_count, malware_count]
        labels_pie = [f'良性应用\n{benign_count:,} ({benign_count/total*100:.1f}%)' if not USE_ENGLISH 
                     else f'Benign\n{benign_count:,} ({benign_count/total*100:.1f}%)',
                     f'恶意软件\n{malware_count:,} ({malware_count/total*100:.1f}%)' if not USE_ENGLISH
                     else f'Malware\n{malware_count:,} ({malware_count/total*100:.1f}%)']
        colors_pie = [COLORS['benign'], COLORS['malware']]
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels_pie, colors=colors_pie,
                                           autopct='', startangle=90, 
                                           textprops={'fontsize': 11, 'fontweight': 'bold'})
        
        # 添加中心圆
        centre_circle = plt.Circle((0,0), 0.70, fc='white', linewidth=2, edgecolor='gray')
        ax.add_artist(centre_circle)
        
        # 添加中心文字
        ax.text(0, 0, f'总计\n{total:,}', ha='center', va='center', 
                fontsize=14, fontweight='bold', color=COLORS['text'])
        
        ax.set_title('🎯 良性 vs 恶意流量分布', fontsize=14, fontweight='bold', pad=20)
    
    def _plot_flow_length_distribution(self, ax):
        """绘制流长度分布"""
        valid_packets = np.sum(self.sequences[:, :, 1] > 0, axis=1)
        
        # 直方图 + KDE
        sns.histplot(valid_packets, bins=50, kde=True, ax=ax, 
                    color=COLORS['primary'], alpha=0.6)
        
        ax.axvline(valid_packets.mean(), color=COLORS['malware'], 
                  linestyle='--', linewidth=2, label=f'均值: {valid_packets.mean():.1f}')
        ax.axvline(np.median(valid_packets), color=COLORS['benign'], 
                  linestyle='--', linewidth=2, label=f'中位数: {np.median(valid_packets):.1f}')
        
        ax.set_xlabel('流长度 (包数)' if not USE_ENGLISH else 'Flow Length (packets)', fontsize=12)
        ax.set_ylabel('频次' if not USE_ENGLISH else 'Frequency', fontsize=12)
        ax.set_title('📈 流长度分布', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    def plot_packet_features(self):
        """绘制包特征分析图"""
        print("\n📦 生成包特征分析图...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 包长度分布
        ax1 = axes[0, 0]
        packet_lengths = self.sequences[:, :, 1][self.sequences[:, :, 1] > 0]
        sns.histplot(packet_lengths, bins=100, ax=ax1, color=COLORS['accent'], alpha=0.6, log_scale=True)
        ax1.set_xlabel('包长度 (字节)' if not USE_ENGLISH else 'Packet Length (bytes)', fontsize=11)
        ax1.set_ylabel('频次' if not USE_ENGLISH else 'Frequency', fontsize=11)
        ax1.set_title('📦 包长度分布 (对数坐标)', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 添加统计信息
        stats_text = f'均值: {packet_lengths.mean():.1f}\n中位数: {np.median(packet_lengths):.1f}\n标准差: {packet_lengths.std():.1f}'
        ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. 时间间隔分布
        ax2 = axes[0, 1]
        time_diffs = self.sequences[:, 1:, 0][self.sequences[:, 1:, 0] > 0]
        time_diffs = time_diffs[time_diffs > 0]  # 只考虑正的时间间隔
        
        if len(time_diffs) > 0:
            sns.histplot(time_diffs, bins=100, ax=ax2, color=COLORS['secondary'], alpha=0.6, log_scale=True)
            ax2.set_xlabel('时间间隔 (秒)' if not USE_ENGLISH else 'Inter-Arrival Time (s)', fontsize=11)
            ax2.set_ylabel('频次' if not USE_ENGLISH else 'Frequency', fontsize=11)
            ax2.set_title('⏱️ 包间时间间隔分布 (对数坐标)', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
        
        # 3. 方向分布
        ax3 = axes[1, 0]
        # 方向特征通常是第10个特征（索引9）
        if self.sequences.shape[2] > 9:
            directions = self.sequences[:, :, 9][self.sequences[:, :, 1] > 0]
            direction_counts = Counter(directions)
            
            labels_dir = ['正向 (0)' if not USE_ENGLISH else 'Forward (0)',
                         '反向 (1)' if not USE_ENGLISH else 'Backward (1)']
            sizes_dir = [direction_counts.get(0, 0), direction_counts.get(1, 0)]
            colors_dir = [COLORS['benign'], COLORS['malware']]
            
            if sum(sizes_dir) > 0:
                wedges, texts, autotexts = ax3.pie(sizes_dir, labels=labels_dir, colors=colors_dir,
                                                   autopct='%1.1f%%', startangle=90,
                                                   textprops={'fontsize': 12})
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
        
        ax3.set_title('🔄 数据包方向分布', fontsize=13, fontweight='bold')
        
        # 4. 各类别包长度对比
        ax4 = axes[1, 1]
        class_packet_lengths = []
        class_names_display = []
        
        # 选择8个主要类别
        counter = Counter(self.labels)
        top_classes = [l for l, _ in counter.most_common(8)]
        
        for label in top_classes:
            mask = self.labels == label
            class_seqs = self.sequences[mask]
            lengths = class_seqs[:, :, 1][class_seqs[:, :, 1] > 0]
            if len(lengths) > 0:
                class_packet_lengths.append(lengths)
                class_names_display.append(get_class_display(label))
        
        if class_packet_lengths:
            bp = ax4.boxplot(class_packet_lengths, labels=class_names_display, patch_artist=True)
            
            # 设置颜色
            for patch, label in zip(bp['boxes'], top_classes[:len(class_packet_lengths)]):
                patch.set_facecolor(get_class_color(label))
                patch.set_alpha(0.6)
            
            for whisker in bp['whiskers']:
                whisker.set_color(COLORS['text'])
            for cap in bp['caps']:
                cap.set_color(COLORS['text'])
            for median in bp['medians']:
                median.set_color('red')
                median.set_linewidth(2)
        
        ax4.set_xlabel('类别' if not USE_ENGLISH else 'Class', fontsize=11)
        ax4.set_ylabel('包长度 (字节)' if not USE_ENGLISH else 'Packet Length (bytes)', fontsize=11)
        ax4.set_title('📊 主要类别包长度分布对比', fontsize=13, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        plt.suptitle('📈 USTC-TFC2016 包特征分析', fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # 保存图片
        output_path = self.output_dir / '02_ustc_packet_features.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✅ 已保存: {output_path.name}")
        
        # 显示图片
        print("   👆 图2: 包特征分析 (关闭图片继续)")
        plt.show()
        plt.close(fig)
    
    def plot_sample_visualization(self, n_samples=6):
        """绘制样本可视化图"""
        print(f"\n📊 生成样本可视化图 (n={n_samples})...")
        
        n_rows = (n_samples + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4*n_rows))
        axes = axes.flatten()
        
        # 随机选择样本
        indices = np.random.choice(len(self.sequences), n_samples, replace=False)
        
        for i, idx in enumerate(indices):
            ax = axes[i]
            sample = self.sequences[idx]
            label = int(self.labels[idx])
            
            # 找到非零包
            non_zero_mask = np.any(sample != 0, axis=1)
            valid_packets = sample[non_zero_mask]
            packet_indices = np.arange(len(valid_packets))
            
            if len(valid_packets) > 0:
                # 绘制包长度序列
                ax.plot(packet_indices, valid_packets[:, 1], 'o-', 
                       color=get_class_color(label), markersize=6, linewidth=2, alpha=0.7)
                
                # 添加颜色填充
                ax.fill_between(packet_indices, 0, valid_packets[:, 1], 
                               alpha=0.2, color=get_class_color(label))
            
            ax.set_xlabel('包序号' if not USE_ENGLISH else 'Packet Index', fontsize=10)
            ax.set_ylabel('包长度 (字节)' if not USE_ENGLISH else 'Packet Length (bytes)', fontsize=10)
            ax.set_title(f'样本 {idx} - {get_class_display(label)}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # 添加统计信息
            stats = f'有效包数: {len(valid_packets)}'
            ax.text(0.02, 0.95, stats, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 隐藏多余的子图
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle('📊 样本流量序列示例', fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # 保存图片
        output_path = self.output_dir / '03_ustc_sample_sequences.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✅ 已保存: {output_path.name}")
        
        # 显示图片
        print("   👆 图3: 样本序列示例 (关闭图片继续)")
        plt.show()
        plt.close(fig)
    
    def plot_feature_correlation(self):
        """绘制特征相关性热力图"""
        print("\n🔗 生成特征相关性热力图...")
        
        # 随机采样以减少计算量
        if len(self.sequences) > 5000:
            indices = np.random.choice(len(self.sequences), 5000, replace=False)
            sample_seqs = self.sequences[indices]
        else:
            sample_seqs = self.sequences
        
        # 重塑数据：将所有包的特征合并
        n_samples, n_packets, n_features = sample_seqs.shape
        flattened = sample_seqs.reshape(-1, n_features)
        
        # 移除全零行（填充的包）
        non_zero_rows = np.any(flattened != 0, axis=1)
        flattened = flattened[non_zero_rows]
        
        if len(flattened) > 0:
            # 计算相关性矩阵
            corr_matrix = np.corrcoef(flattened.T)
            
            # 特征名称
            feature_names = [
                '时间差', '包长度', 'IP头长度', 'TTL', '协议',
                '源端口', '目的端口', 'TCP标志', '窗口大小', '方向', 'Payload长度'
            ]
            if n_features > 11:
                feature_names += [f'P{i}' for i in range(n_features - 11)]
            
            # 绘制热力图
            fig, ax = plt.subplots(figsize=(18, 15))
            
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0,
                       xticklabels=feature_names[:len(corr_matrix)], 
                       yticklabels=feature_names[:len(corr_matrix)],
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
            
            ax.set_title('🔗 特征相关性热力图', fontsize=16, fontweight='bold')
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            
            # 保存图片
            output_path = self.output_dir / '04_ustc_feature_correlation.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"   ✅ 已保存: {output_path.name}")
            
            # 显示图片
            print("   👆 图4: 特征相关性热力图 (关闭图片继续)")
            plt.show()
            plt.close(fig)
        else:
            print("   ⚠️ 没有足够的有效数据计算相关性")
    
    def generate_reports(self):
        """生成CSV和文本报告"""
        print("\n📋 生成分析报告...")
        
        # 基本统计
        total_samples = len(self.sequences)
        n_features = self.sequences.shape[2]
        n_classes = len(np.unique(self.labels))
        
        # 类别统计
        counter = Counter(self.labels)
        benign_count = sum(1 for l in self.labels if int(l) in CLASS_NAMES and CLASS_NAMES[int(l)]['type'] == 'benign')
        malware_count = sum(1 for l in self.labels if int(l) in CLASS_NAMES and CLASS_NAMES[int(l)]['type'] == 'malware')
        
        # 流长度统计
        valid_packets = np.sum(self.sequences[:, :, 1] > 0, axis=1)
        
        # 创建报告DataFrame
        report_data = []
        for label in sorted(counter.keys()):
            count = counter[label]
            percentage = count/total_samples*100
            if int(label) in CLASS_NAMES:
                class_info = CLASS_NAMES[int(label)]
                report_data.append({
                    'Class ID': label,
                    'Name (EN)': class_info['en'],
                    'Name (ZH)': class_info['zh'],
                    'Type': class_info['type'],
                    'Count': count,
                    'Percentage': f"{percentage:.2f}%"
                })
            else:
                report_data.append({
                    'Class ID': label,
                    'Name (EN)': f'Unknown-{label}',
                    'Name (ZH)': f'未知-{label}',
                    'Type': 'unknown',
                    'Count': count,
                    'Percentage': f"{percentage:.2f}%"
                })
        
        df = pd.DataFrame(report_data)
        
        # 保存CSV
        csv_path = self.output_dir / 'class_distribution.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"   ✅ 已保存: {csv_path.name}")
        
        # 生成文本报告
        txt_path = self.output_dir / 'analysis_report.txt'
        self._generate_text_report(df, total_samples, n_features, n_classes,
                                  benign_count, malware_count, valid_packets, txt_path)
        
        # 打印统计摘要
        self._print_statistics_summary(df, total_samples, benign_count, malware_count, valid_packets)
    
    def _generate_text_report(self, df, total_samples, n_features, n_classes,
                              benign_count, malware_count, valid_packets, txt_path):
        """生成文本报告"""
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("USTC-TFC2016 数据集分析报告\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输出目录: {self.output_dir}\n\n")
            
            f.write("📦 数据集大小:\n")
            f.write(f"  总样本数: {total_samples:,}\n")
            f.write(f"  序列形状: {self.sequences.shape}\n")
            f.write(f"  特征维度: {n_features}\n")
            f.write(f"  类别数量: {n_classes}\n\n")
            
            f.write("🎯 良性 vs 恶意:\n")
            f.write(f"  良性应用: {benign_count:,} ({benign_count/total_samples*100:.2f}%)\n")
            f.write(f"  恶意软件: {malware_count:,} ({malware_count/total_samples*100:.2f}%)\n\n")
            
            f.write("📈 流长度统计:\n")
            f.write(f"  平均值: {valid_packets.mean():.2f}\n")
            f.write(f"  中位数: {np.median(valid_packets):.2f}\n")
            f.write(f"  标准差: {valid_packets.std():.2f}\n")
            f.write(f"  最小值: {valid_packets.min()}\n")
            f.write(f"  最大值: {valid_packets.max()}\n\n")
            
            f.write("📋 类别分布:\n")
            f.write("-"*60 + "\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")
            
            f.write("📁 生成的文件列表:\n")
            f.write("  01_ustc_data_overview.png - 数据概览图\n")
            f.write("  02_ustc_packet_features.png - 包特征分析图\n")
            f.write("  03_ustc_sample_sequences.png - 样本序列示例图\n")
            f.write("  04_ustc_feature_correlation.png - 特征相关性热力图\n")
            f.write("  class_distribution.csv - 类别分布CSV\n")
            f.write("  analysis_report.txt - 文本格式报告\n\n")
            
            f.write("="*70 + "\n")
            f.write("报告生成完成\n")
        
        print(f"   ✅ 已保存: {txt_path.name}")
    
    def _print_statistics_summary(self, df, total_samples, benign_count, malware_count, valid_packets):
        """打印统计摘要"""
        print("\n" + "="*70)
        print("📊 统计摘要")
        print("="*70)
        
        print(f"\n📦 数据集大小:")
        print(f"   ├─ 总样本数: {total_samples:,}")
        print(f"   ├─ 良性应用: {benign_count:,} ({benign_count/total_samples*100:.2f}%)")
        print(f"   └─ 恶意软件: {malware_count:,} ({malware_count/total_samples*100:.2f}%)")
        
        print(f"\n📈 流长度统计:")
        print(f"   ├─ 平均值: {valid_packets.mean():.2f}")
        print(f"   ├─ 中位数: {np.median(valid_packets):.2f}")
        print(f"   ├─ 标准差: {valid_packets.std():.2f}")
        print(f"   ├─ 最小值: {valid_packets.min()}")
        print(f"   └─ 最大值: {valid_packets.max()}")
        
        print(f"\n📋 类别分布:")
        print(df[['Class ID', 'Name (EN)', 'Count', 'Percentage']].to_string(index=False))
    
    def run_all_analyses(self, show_correlation=True, n_samples=6):
        """运行所有分析，一张一张展示图表"""
        print("\n" + "="*70)
        print("🚀 开始全面数据分析 - 交互式模式")
        print("📢 图表将一张一张显示，关闭当前图表以查看下一张")
        print("="*70)
        
        start_time = time.time()
        
        # 生成并显示概览图
        self.plot_data_overview()
        
        # 生成并显示特征分析图
        self.plot_packet_features()
        
        # 生成并显示样本可视化
        self.plot_sample_visualization(n_samples)
        
        # 生成并显示特征相关性热力图（可选）
        if show_correlation:
            self.plot_feature_correlation()
        
        # 生成报告（不显示图表）
        self.generate_reports()
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*70)
        print(f"✅ 所有分析完成! 用时: {elapsed_time:.2f} 秒")
        print(f"📁 输出目录: {self.output_dir}")
        print("="*70)
        
        # 列出所有生成的文件
        print("\n📁 生成的文件列表:")
        files = list(self.output_dir.glob("*"))
        for file in sorted(files):
            size = file.stat().st_size / 1024  # KB
            print(f"  📄 {file.name} ({size:.1f} KB)")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='USTC-TFC2016 数据探索 - 交互式版本')
    parser.add_argument('--data_dir', type=str,
                       default=r"E:\USTC\preprocessed_data",
                       help='数据目录')
    parser.add_argument('--output', type=str,
                       default=r"E:\USTC\explore",
                       help='输出目录')
    parser.add_argument('--no_correlation', action='store_true',
                       help='不生成特征相关性热力图')
    parser.add_argument('--samples', type=int, default=6,
                       help='样本可视化数量 (默认: 6)')
    parser.add_argument('--english', action='store_true',
                       help='使用英文显示')
    
    args = parser.parse_args()
    
    # 设置语言
    global USE_ENGLISH
    USE_ENGLISH = args.english
    
    print(f"📁 数据目录: {args.data_dir}")
    print(f"📁 输出目录: {args.output}")
    print(f"🔤 语言: {'英文' if args.english else '中文'}")
    print(f"📊 样本数量: {args.samples}")
    print(f"🔗 相关性热力图: {'跳过' if args.no_correlation else '生成'}")
    
    # 创建探索器
    explorer = USTCDataExplorer(args.data_dir, args.output)
    
    if explorer.sequences is not None:
        explorer.run_all_analyses(
            show_correlation=not args.no_correlation,
            n_samples=args.samples
        )
    else:
        print("❌ 数据加载失败，无法进行分析")

if __name__ == "__main__":
    main()