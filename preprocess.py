#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USTC-TFC2016 数据集预处理脚本 - 完整特征提取版本（修复版）
"""

import os
import numpy as np
import pickle
from collections import defaultdict
from scapy.all import IP, TCP, UDP, Raw
from scapy.utils import PcapReader
import warnings
import time
from tqdm import tqdm
import argparse
from pathlib import Path

warnings.filterwarnings('ignore')

class USTCDataPreprocessor:
    """USTC-TFC2016数据集预处理器"""
    
    def __init__(self, 
                 max_packets_per_flow=100,
                 min_packets_per_flow=3,
                 max_files_per_class=None,
                 payload_bytes=128):
        """
        初始化预处理器
        """
        self.max_packets = max_packets_per_flow
        self.min_packets = min_packets_per_flow
        self.max_files = max_files_per_class
        self.payload_bytes = payload_bytes
        
        # USTC类别映射
        self.class_mapping = {
            # 良性应用 (0-9)
            'bittorrent': 0, 'facetime': 1, 'ftp': 2, 'gmail': 3,
            'mysql': 4, 'outlook': 5, 'skype': 6, 'smb': 7,
            'weibo': 8, 'worldofwarcraft': 9, 'wow': 9,
            
            # 恶意软件 (10-19)
            'cridex': 10, 'geodo': 11, 'htbot': 12, 'miuref': 13,
            'neris': 14, 'nsis-ay': 15, 'nsis': 15, 'shifu': 16,
            'tinba': 17, 'virut': 18, 'zeus': 19,
        }
        
        # 类别名称
        self.class_names = {
            0: 'BitTorrent', 1: 'Facetime', 2: 'FTP', 3: 'Gmail',
            4: 'MySQL', 5: 'Outlook', 6: 'Skype', 7: 'SMB',
            8: 'Weibo', 9: 'WorldOfWarcraft',
            10: 'Cridex', 11: 'Geodo', 12: 'Htbot', 13: 'Miuref',
            14: 'Neris', 15: 'Nsis-ay', 16: 'Shifu', 17: 'Tinba',
            18: 'Virut', 19: 'Zeus'
        }
    
    def safe_get_time(self, packet):
        """安全地获取包的时间戳"""
        try:
            if hasattr(packet, 'time'):
                time_val = packet.time
                # 如果是FlagValue类型，转换为float
                if hasattr(time_val, 'real'):
                    return float(time_val.real)
                return float(time_val)
        except:
            pass
        return 0.0
    
    def safe_get_flags(self, tcp_layer):
        """安全地获取TCP标志"""
        try:
            flags = tcp_layer.flags
            if hasattr(flags, 'value'):
                return int(flags.value)
            return int(flags)
        except:
            return 0
    
    def extract_packet_features(self, packet):
        """
        从数据包中提取完整特征
        """
        features = {
            'timestamp': self.safe_get_time(packet),
            'packet_len': len(packet),
            'ip_header_len': 0,
            'ttl': 0,
            'protocol': 0,
            'src_ip': '',
            'dst_ip': '',
            'src_port': 0,
            'dst_port': 0,
            'tcp_flags': 0,
            'window_size': 0,
            'payload_bytes': np.zeros(self.payload_bytes, dtype=np.uint8),
            'payload_len': 0
        }
        
        # IP层特征
        if IP in packet:
            ip = packet[IP]
            features['ip_header_len'] = ip.ihl * 4
            features['ttl'] = ip.ttl
            features['protocol'] = ip.proto
            features['src_ip'] = ip.src
            features['dst_ip'] = ip.dst
            
            # 传输层特征
            if TCP in packet:
                tcp = packet[TCP]
                features['src_port'] = tcp.sport
                features['dst_port'] = tcp.dport
                features['tcp_flags'] = self.safe_get_flags(tcp)
                features['window_size'] = tcp.window
                
                # 提取payload
                if Raw in tcp:
                    try:
                        payload = bytes(tcp[Raw])
                        features['payload_len'] = len(payload)
                        # 截取或填充payload
                        if len(payload) > self.payload_bytes:
                            features['payload_bytes'] = np.frombuffer(payload[:self.payload_bytes], dtype=np.uint8)
                        else:
                            if len(payload) > 0:
                                payload_array = np.frombuffer(payload, dtype=np.uint8)
                                features['payload_bytes'][:len(payload_array)] = payload_array
                    except:
                        pass
                            
            elif UDP in packet:
                udp = packet[UDP]
                features['src_port'] = udp.sport
                features['dst_port'] = udp.dport
                
                # UDP payload
                if Raw in udp:
                    try:
                        payload = bytes(udp[Raw])
                        features['payload_len'] = len(payload)
                        if len(payload) > self.payload_bytes:
                            features['payload_bytes'] = np.frombuffer(payload[:self.payload_bytes], dtype=np.uint8)
                        else:
                            if len(payload) > 0:
                                payload_array = np.frombuffer(payload, dtype=np.uint8)
                                features['payload_bytes'][:len(payload_array)] = payload_array
                    except:
                        pass
        
        return features
    
    def create_flow_key(self, packet):
        """创建双向流的key"""
        if IP not in packet:
            return None
        
        ip = packet[IP]
        
        # 获取传输层端口
        if TCP in packet:
            proto = 'TCP'
            sport = packet[TCP].sport
            dport = packet[TCP].dport
        elif UDP in packet:
            proto = 'UDP'
            sport = packet[UDP].sport
            dport = packet[UDP].dport
        else:
            return None
        
        # 创建排序后的key以实现双向流
        src = f"{ip.src}:{sport}"
        dst = f"{ip.dst}:{dport}"
        
        if src < dst:
            flow_key = f"{src}|{dst}|{proto}"
            template = (src, dst)
        else:
            flow_key = f"{dst}|{src}|{proto}"
            template = (dst, src)
        
        return flow_key, template
    
    def process_pcap_file(self, pcap_path):
        """
        处理单个pcap文件，提取流序列
        """
        print(f"  处理文件: {Path(pcap_path).name}")
        
        flows = defaultdict(list)
        flow_templates = {}
        packet_count = 0
        
        try:
            # 读取pcap文件
            with PcapReader(str(pcap_path)) as pcap_reader:
                for packet in tqdm(pcap_reader, desc="    读取包", unit="包", leave=False):
                    packet_count += 1
                    
                    # 提取五元组
                    result = self.create_flow_key(packet)
                    if result is None:
                        continue
                    
                    flow_key, template = result
                    
                    # 保存模板
                    if flow_key not in flow_templates:
                        flow_templates[flow_key] = template
                    
                    # 提取包特征
                    features = self.extract_packet_features(packet)
                    flows[flow_key].append(features)
            
            print(f"    读取完成: {packet_count} 个包, {len(flows)} 个流")
            
            # 处理每个流，生成序列
            sequences = []
            metadata = []
            
            for flow_key, packets in flows.items():
                # 按时间排序
                try:
                    packets.sort(key=lambda x: x['timestamp'])
                except Exception as e:
                    print(f"    排序失败: {e}, 跳过该流")
                    continue
                
                # 过滤短流
                if len(packets) < self.min_packets:
                    continue
                
                # 截取或填充到固定长度
                if len(packets) > self.max_packets:
                    packets = packets[:self.max_packets]
                
                # 构建序列特征
                sequence = self.build_sequence_features(packets, flow_templates[flow_key])
                if sequence is not None:
                    sequences.append(sequence)
                    
                    # 记录元数据
                    try:
                        metadata.append({
                            'flow_key': flow_key,
                            'num_packets': len(packets),
                            'duration': packets[-1]['timestamp'] - packets[0]['timestamp'],
                            'total_bytes': sum(p['packet_len'] for p in packets)
                        })
                    except:
                        pass
            
            print(f"    生成 {len(sequences)} 个有效流序列")
            return sequences, metadata
            
        except Exception as e:
            print(f"    处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], []
    
    def build_sequence_features(self, packets, template):
        """
        构建完整的序列特征
        """
        try:
            features_list = []
            
            for i, packet in enumerate(packets):
                # 时间特征
                if i == 0:
                    time_diff = 0.0
                else:
                    time_diff = packet['timestamp'] - packets[i-1]['timestamp']
                    # 处理负数或异常时间差
                    if time_diff < 0:
                        time_diff = 0.0
                
                # 方向特征
                src = f"{packet.get('src_ip', '')}:{packet['src_port']}"
                dst = f"{packet.get('dst_ip', '')}:{packet['dst_port']}"
                direction = 0 if src == template[0] and dst == template[1] else 1
                
                # 基础特征 (11维)
                base_features = [
                    float(time_diff),                          # 包间时间差
                    float(packet['packet_len']),                # 包长度
                    float(packet['ip_header_len']),             # IP头长度
                    float(packet['ttl']),                       # TTL
                    float(packet['protocol']),                   # 协议
                    float(packet['src_port']),                   # 源端口
                    float(packet['dst_port']),                   # 目的端口
                    float(packet['tcp_flags']),                  # TCP标志
                    float(packet['window_size']),                # 窗口大小
                    float(direction),                            # 方向
                    float(packet['payload_len'])                 # payload长度
                ]
                
                # 合并基础特征和payload字节
                payload_norm = packet['payload_bytes'].astype(np.float32) / 255.0
                combined_features = np.concatenate([
                    np.array(base_features, dtype=np.float32),
                    payload_norm
                ])
                
                features_list.append(combined_features)
            
            # 转换为numpy数组
            sequence = np.array(features_list, dtype=np.float32)
            
            # 如果长度不足，填充0
            if len(sequence) < self.max_packets:
                padding = np.zeros((self.max_packets - len(sequence), sequence.shape[1]), dtype=np.float32)
                sequence = np.vstack([sequence, padding])
            
            # 确保没有NaN或Inf
            sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
            
            return sequence
            
        except Exception as e:
            print(f"    构建序列失败: {e}")
            return None
    
    def get_label_from_filename(self, filename):
        """从文件名获取标签"""
        filename_lower = filename.lower()
        
        for key, label in self.class_mapping.items():
            if key in filename_lower:
                return label
        
        return None
    
    def process_directory(self, input_dir, output_dir, class_names=None):
        """
        处理整个数据目录
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取所有类别目录
        if class_names:
            categories = []
            for c in class_names:
                # 尝试匹配完整目录名
                found = False
                for d in input_path.iterdir():
                    if d.is_dir() and c.lower() in d.name.lower():
                        categories.append(d.name)
                        found = True
                        break
                if not found:
                    print(f"警告: 未找到类别 {c}")
        else:
            categories = [d.name for d in input_path.iterdir() 
                         if d.is_dir() and d.name != 'class_mapping.txt']
        
        print("="*60)
        print("USTC-TFC2016 数据集预处理 - 完整特征提取")
        print("="*60)
        print(f"输入目录: {input_dir}")
        print(f"输出目录: {output_dir}")
        print(f"发现 {len(categories)} 个类别")
        
        all_sequences = []
        all_labels = []
        all_metadata = []
        class_stats = {}
        
        # 处理每个类别
        for category_name in categories:
            category_path = input_path / category_name
            
            print(f"\n{'='*50}")
            print(f"处理类别: {category_name}")
            print('='*50)
            
            # 获取所有pcap文件
            pcap_files = list(category_path.glob("*.pcap")) + list(category_path.glob("*.pcapng"))
            
            if self.max_files and len(pcap_files) > self.max_files:
                pcap_files = pcap_files[:self.max_files]
            
            print(f"找到 {len(pcap_files)} 个pcap文件")
            
            category_sequences = []
            category_labels = []
            
            # 处理每个文件
            for pcap_file in pcap_files:
                # 获取标签
                label = self.get_label_from_filename(pcap_file.stem)
                if label is None:
                    label = self.get_label_from_filename(category_name)
                
                if label is None:
                    print(f"  跳过 {pcap_file.name} (无法识别类别)")
                    continue
                
                # 处理pcap文件
                sequences, metadata = self.process_pcap_file(pcap_file)
                
                if sequences:
                    category_sequences.extend(sequences)
                    category_labels.extend([label] * len(sequences))
                    all_metadata.extend(metadata)
            
            # 保存类别数据
            if category_sequences:
                all_sequences.extend(category_sequences)
                all_labels.extend(category_labels)
                class_stats[category_name] = len(category_sequences)
                
                print(f"\n类别 {category_name} 处理完成: {len(category_sequences)} 个序列")
        
        # 保存完整数据集
        if all_sequences:
            print(f"\n{'='*60}")
            print("保存完整数据集")
            print('='*60)
            
            # 转换为numpy数组
            sequences_array = np.array(all_sequences)
            labels_array = np.array(all_labels)
            
            print(f"序列数组形状: {sequences_array.shape}")
            print(f"标签数组形状: {labels_array.shape}")
            
            # 保存完整数据
            complete_data = {
                'sequences': sequences_array,
                'labels': labels_array,
                'class_names': self.class_names,
                'metadata': all_metadata,
                'config': {
                    'max_packets_per_flow': self.max_packets,
                    'min_packets_per_flow': self.min_packets,
                    'payload_bytes': self.payload_bytes,
                    'feature_dim': sequences_array.shape[2] if len(sequences_array) > 0 else 0
                }
            }
            
            # 保存为不同格式
            output_file_pkl = output_path / "ustc_complete_data.pkl"
            with open(output_file_pkl, 'wb') as f:
                pickle.dump(complete_data, f)
            
            output_file_npy = output_path / "ustc_sequences.npy"
            np.save(output_file_npy, sequences_array)
            
            output_file_labels = output_path / "ustc_labels.npy"
            np.save(output_file_labels, labels_array)
            
            print(f"\n数据已保存到: {output_path}")
            print(f"  - 序列文件: ustc_sequences.npy")
            print(f"  - 标签文件: ustc_labels.npy")
            print(f"  - 完整数据: ustc_complete_data.pkl")
            
            # 打印统计信息
            self.print_statistics(sequences_array, labels_array)
            
        else:
            print("\n错误: 没有处理到任何有效数据!")
        
        return all_sequences, all_labels
    
    def print_statistics(self, sequences, labels):
        """打印统计信息"""
        print(f"\n{'='*60}")
        print("数据统计")
        print('='*60)
        
        print(f"总样本数: {len(sequences)}")
        print(f"序列形状: {sequences.shape}")
        print(f"标签分布:")
        
        unique_labels = np.unique(labels)
        for label in sorted(unique_labels):
            count = np.sum(labels == label)
            class_name = self.class_names.get(label, f"Unknown-{label}")
            print(f"  {class_name} ({label}): {count} 个样本 ({count/len(labels)*100:.2f}%)")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='USTC-TFC2016数据集预处理')
    parser.add_argument('--input_dir', type=str, 
                       default=r"E:\USTC\data",
                       help='输入数据目录')
    parser.add_argument('--output_dir', type=str,
                       default=r"E:\USTC\preprocessed_data",
                       help='输出目录')
    parser.add_argument('--max_packets', type=int, default=100,
                       help='每个流最大包数')
    parser.add_argument('--min_packets', type=int, default=3,
                       help='每个流最小包数')
    parser.add_argument('--payload_bytes', type=int, default=64,
                       help='提取的payload字节数（减少以节省内存）')
    parser.add_argument('--max_files_per_class', type=int, default=None,
                       help='每类最大处理的文件数')
    parser.add_argument('--classes', nargs='+', default=None,
                       help='指定要处理的类别')
    
    args = parser.parse_args()
    
    # 创建预处理器
    preprocessor = USTCDataPreprocessor(
        max_packets_per_flow=args.max_packets,
        min_packets_per_flow=args.min_packets,
        max_files_per_class=args.max_files_per_class,
        payload_bytes=args.payload_bytes
    )
    
    # 处理数据
    sequences, labels = preprocessor.process_directory(
        args.input_dir,
        args.output_dir,
        args.classes
    )
    
    print("\n" + "="*60)
    print("数据预处理完成!")
    print("="*60)

if __name__ == "__main__":
    main()