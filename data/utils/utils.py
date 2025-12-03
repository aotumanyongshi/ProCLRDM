import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F

# 辅助函数：将模型输出的张量解码回 DNA 序列字符串
# 这个函数需要根据你的 VAE 模型输出张量形状进行修改
char_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
idx_to_char = {0: 'A', 1: 'C', 2: 'G', 3: 'T'} # <--- 确保这里有正确的映射

def decoder2seq_torch(decoded_tensor, original_seq_len, padding):
    """
    将 VAE 解码器的输出张量转换为 DNA 序列字符串列表。

    Args:
        decoded_tensor (torch.Tensor): VAE 解码器的输出张量，形状 (batch_size, 1, 4, 185)
                                       (通常是 logits 或概率)。
        original_seq_len (int): 原始序列的长度 (例如 165)。
        padding (int): 填充的长度 (例如 10)。

    Returns:
        list: 生成的 DNA 序列字符串列表。
    """
    # 确保输入张量在 CPU 上以便转换为 numpy (如果需要在 numpy 中处理的话)
    # 或者在 GPU 上直接进行张量操作
    decoded_tensor = decoded_tensor.cpu()

    # 1. 如果输入是 logits，应用 Softmax 获取概率
    # VAE 解码器输出通常是 logits，表示碱基的维度是 2 (尺寸为 4)
    # 注意：如果你的 VAE decode 方法已经 apply_sigmoid=True，这里就不需要 Softmax 或 Sigmoid
    # 并且应该直接使用这个概率张量
    # 这里假设输入是 logits，对维度 2 应用 Softmax
    probabilities = F.softmax(decoded_tensor, dim=2) # 在碱基维度上进行 Softmax

    # 2. 在碱基维度上使用 argmax 获取预测的碱基索引
    # 结果形状为 (batch_size, 1, 185)
    predicted_indices = torch.argmax(probabilities, dim=2)

    # 3. 移除不必要的维度，得到形状 (batch_size, 185)
    predicted_indices = predicted_indices.squeeze(1) # 移除维度 1

    # 4. 移除填充部分
    # 序列长度是 original_seq_len + 2 * padding
    # 填充在两端，每端 padding 长度
    # 核心序列的起始索引是 padding，结束索引是 padding + original_seq_len
    start_index = padding
    end_index = padding + original_seq_len

    sequences = []
    # 遍历批次中的每个序列
    for i in range(predicted_indices.size(0)):
        # 获取单个序列的预测索引 (形状: 185,)
        single_sequence_indices = predicted_indices[i]

        # 移除填充部分，获取核心序列索引 (形状: original_seq_len,)
        core_sequence_indices = single_sequence_indices[start_index:end_index]

        # 5. 将索引映射回字符并连接成字符串
        sequence_string = "".join([idx_to_char[idx.item()] for idx in core_sequence_indices])

        sequences.append(sequence_string)

    return sequences
class SequenceDataset(Dataset):
    def __init__(self, file_path, seq_length=165, padding=10):
        """
        自定义 PyTorch Dataset 类，用于加载和预处理 DNA 序列数据。

        Args:
            file_path (str): 包含 DNA 序列的文本文件路径，每行一个序列。
            seq_length (int): 序列的预期长度（这里是 165）。
            padding (int): 在序列两端添加的零填充长度。
        """
        self.file_path = file_path
        self.seq_length = seq_length
        self.padding = padding
        self.total_width = seq_length + 2 * padding
        self.sequences = self._load_sequences()
        self.base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': -1} # 添加 'N' 的处理

        # 检查序列长度是否一致 (可选，但推荐对于固定长度数据集进行检查)
        # for i, seq in enumerate(self.sequences):
        #     if len(seq) != self.seq_length:
        #         print(f"Warning: Sequence at index {i} has length {len(seq)}, expected {self.seq_length}. This sequence will be skipped or truncated/padded.")
        #         # 根据需要处理长度不匹配的序列，这里为了简化，假设所有序列长度都是 self.seq_length

    def _load_sequences(self):
        """从文件中加载序列。"""
        sequences = []
        try:
            with open(self.file_path, 'r') as f:
                for line in f:
                    seq = line.strip().upper() # 读取并转换为大写
                    # 可以添加过滤非 DNA 字符的逻辑
                    sequences.append(seq)
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
            return []
        return sequences

    def __len__(self):
        """返回数据集中的序列数量。"""
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        获取单个序列并进行 One-hot 编码和填充。

        Args:
            idx (int): 序列的索引。

        Returns:
            torch.Tensor: 经过 One-hot 编码和填充后的序列张量。
                          形状为 (1, 4, total_width)。
        """
        seq = self.sequences[idx]

        # 创建一个填充了零的 NumPy 数组，形状为 (4, total_width)
        one_hot_matrix = np.zeros((4, self.total_width), dtype=np.float32)

        # 遍历序列，进行 One-hot 编码并放置到正确位置
        # b+10 是考虑到左侧的 10 个填充位
        for b in range(len(seq)): # 这里假设 len(seq) == self.seq_length == 165
            base = seq[b]
            if base in self.base_dict and self.base_dict[base] != -1:
                # 将对应碱基位置设为 1.0
                one_hot_matrix[self.base_dict[base], b + self.padding] = 1.0
            # 如果是 'N' 或其他未知字符，对应列保持为零，即 [0,0,0,0]

        # 转换为 PyTorch 张量，并添加额外的维度以匹配 VAE 模型期望的形状 (1, 4, total_width)
        # 原始模型期望 (1, 1, 4, 120)，这里形状调整为 (1, 4, 185) 以匹配 PyTorch 的 Conv2d 默认维度顺序 (C, H, W)
        # 或者，如果你的 VAE 模型期望 (1, 1, 4, 185) 的形状，你需要调整这里的 unsqueeze
        # 例如：tensor = torch.tensor(one_hot_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # 形状 (1, 1, 4, 185)
        tensor = torch.tensor(one_hot_matrix, dtype=torch.float32).unsqueeze(0) # 形状 (1, 4, 185)
        # print(f"one_hot_matrix{tensor.shape}")
        return tensor
