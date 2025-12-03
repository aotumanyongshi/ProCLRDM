import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import time
from tqdm import tqdm

# DNA序列处理工具（离散编码：直接用整数表示碱基）
INDEX_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
BASE_TO_INDEX = {v: k for k, v in INDEX_TO_BASE.items()}
NUM_BASES = 4  # 离散类别数：4种碱基


def discrete_to_sequence(discrete_tensor, seq_length):
    """将离散整数张量转换为DNA序列"""
    return "".join([INDEX_TO_BASE.get(i.item(), 'N') for i in discrete_tensor[:seq_length]])


def sequence_to_discrete(seq, seq_length, padding=0):
    """将DNA序列转换为离散整数编码（0-3）"""
    discrete = torch.full((seq_length + 2 * padding,), fill_value=-1, dtype=torch.long)  # 初始化为-1（填充标记）
    for i, c in enumerate(seq):
        discrete[i + padding] = BASE_TO_INDEX.get(c, -1)  # 未知碱基用-1表示
    return discrete


# DNA数据集（离散版本）
class DiscreteDNADataset(Dataset):
    def __init__(self, file_path, seq_length, padding=0):
        self.seq_length = seq_length
        self.padding = padding
        self.padded_length = seq_length + 2 * padding
        self.sequences = self._load_sequences(file_path)

    def _load_sequences(self, file_path):
        seqs = []
        with open(file_path, 'r') as f:
            for line in f:
                seq = line.strip().upper()
                if len(seq) == self.seq_length and all(c in BASE_TO_INDEX for c in seq):
                    discrete = sequence_to_discrete(seq, self.seq_length, self.padding)
                    seqs.append(discrete)
        return seqs

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


# 离散去噪模型（输出碱基概率分布）
class DiscreteDenoiseCNN(nn.Module):
    def __init__(self, seq_length, model_dim=64):
        super().__init__()
        self.seq_length = seq_length
        self.model_dim = model_dim

        # 嵌入层：将离散碱基（0-3）转换为向量
        self.embedding = nn.Embedding(
            num_embeddings=NUM_BASES + 1,  # +1 用于处理填充标记-1（会被映射为0向量）
            embedding_dim=model_dim,
            padding_idx=NUM_BASES  # 填充标记-1会被映射为该索引，对应0向量
        )

        # CNN层：处理序列特征
        self.conv1 = nn.Conv1d(model_dim, model_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(model_dim, model_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(model_dim, model_dim, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(model_dim, NUM_BASES, kernel_size=3, padding=1)  # 输出4种碱基的概率
        self.act = nn.SiLU()
        self.norm1 = nn.BatchNorm1d(model_dim)
        self.norm2 = nn.BatchNorm1d(model_dim)
        self.norm3 = nn.BatchNorm1d(model_dim)

        # 时间嵌入：将时间步转换为特征
        self.time_emb = nn.Sequential(
            nn.Embedding(1000, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim)
        )

    def forward(self, x, t):
        """
        x: 离散整数张量，形状 [B, L]（L为序列长度）
        t: 时间步张量，形状 [B]
        """
        # 嵌入离散碱基和时间步
        x_emb = self.embedding(x)  # [B, L, model_dim]
        x_emb = x_emb.transpose(1, 2)  # [B, model_dim, L]

        t_emb = self.time_emb(t)  # [B, model_dim]
        t_emb = t_emb[:, :, None].repeat(1, 1, x_emb.shape[-1])  # [B, model_dim, L]

        # CNN特征提取
        h = self.act(self.norm1(self.conv1(x_emb) + t_emb))
        h = self.act(self.norm2(self.conv2(h) + t_emb))
        h = self.act(self.norm3(self.conv3(h) + t_emb))
        logits = self.conv4(h)  # [B, NUM_BASES, L]

        # 恢复形状：[B, L, NUM_BASES]，便于计算交叉熵损失
        return logits.transpose(1, 2)


# 离散DDPM核心（直接处理离散碱基）
class DiscreteDDPM(nn.Module):
    def __init__(self, denoise_model, timesteps=100, noise_schedule="linear"):
        super().__init__()
        self.denoise_model = denoise_model
        self.timesteps = timesteps
        self.num_bases = NUM_BASES

        # 离散噪声调度：定义每个时间步的扰动概率（随时间增加）
        if noise_schedule == "linear":
            self.transition_probs = torch.linspace(0.01, 0.99, timesteps)  # 从1%到99%的扰动概率
        else:
            self.transition_probs = torch.cos(torch.linspace(np.pi/2, 0, timesteps)) ** 2 * 0.98 + 0.01  # 余弦调度

        # 将参数移至模型设备
        self._move_parameters_to_device()

    def _move_parameters_to_device(self):
        device = next(self.denoise_model.parameters()).device
        self.transition_probs = self.transition_probs.to(device)

    def q_sample(self, x0, t, noise=None):
        """
        前向扩散：在离散空间中扰动序列
        x0: 原始离散序列 [B, L]
        t: 时间步 [B]
        返回：扰动后的序列 x_t
        """
        device = x0.device
        batch_size, seq_len = x0.shape
        x_t = x0.clone()

        # 对每个样本，根据其时间步t的概率进行扰动
        for i in range(batch_size):
            prob = self.transition_probs[t[i]]  # 当前时间步的扰动概率
            # 生成扰动掩码：哪些位置需要被替换
            mask = torch.rand(seq_len, device=device) < prob
            # 生成随机碱基（0-3）作为噪声
            if noise is None:
                noise = torch.randint(0, self.num_bases, (seq_len,), device=device)
            # 对掩码位置替换为噪声
            x_t[i, mask] = noise[mask]

        return x_t

    def p_losses(self, x0, t):
        """
        训练损失：预测原始序列x0（离散分类损失）
        """
        x_t = self.q_sample(x0, t)  # 生成扰动后的序列
        logits = self.denoise_model(x_t, t)  # 模型预测的碱基概率分布 [B, L, 4]
        # 交叉熵损失：预测分布与真实碱基的差异
        loss = F.cross_entropy(
            logits.transpose(1, 2),  # [B, 4, L]（交叉熵要求类别维度在第二维）
            x0,  # 真实标签 [B, L]
            ignore_index=-1  # 忽略填充标记
        )
        return loss

    @torch.no_grad()
    def p_sample(self, x_t, t):
        """
        反向去噪：从x_t生成x_{t-1}（离散采样）
        """
        batch_size, seq_len = x_t.shape
        device = x_t.device

        # 模型预测当前时间步的碱基概率分布
        logits = self.denoise_model(x_t, t)  # [B, L, 4]
        probs = F.softmax(logits, dim=-1)  # 转换为概率

        # 采样：根据概率分布选择碱基（也可直接取argmax，确定性更强但多样性可能降低）
        x_prev = torch.multinomial(probs.reshape(-1, self.num_bases), num_samples=1).reshape(batch_size, seq_len)

        # 最后一步强制使用argmax，避免采样噪声
        if (t == 0).any():
            deterministic_mask = (t == 0)[:, None].repeat(1, seq_len)
            x_prev[deterministic_mask] = torch.argmax(probs, dim=-1)[deterministic_mask]

        return x_prev

    @torch.no_grad()
    def sample(self, batch_size, seq_length):
        """生成离散DNA序列"""
        device = next(self.denoise_model.parameters()).device
        # 初始序列：完全随机的碱基（0-3）
        x = torch.randint(0, self.num_bases, (batch_size, seq_length), device=device)

        # 反向扩散：从T-1到0
        for t in range(self.timesteps - 1, -1, -1):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_tensor)

        return x


# 训练函数
def train_discrete_ddpm():
    # 配置参数
    config = {
        "DATA_PATH": "/root/autodl-tmp/sjy/dachang1/data/raw_sequences/ecoli_data/ecoli_data.txt",
        "SEQ_LENGTH": 165,
        "PADDING": 0,
        "BATCH_SIZE": 8,
        "LEARNING_RATE": 1e-4,
        "NUM_EPOCHS": 100,
        "TIMESTEPS": 1000,
        "MODEL_DIM": 64,
        "SAVE_MODEL_PATH": "discrete_ddpm_model.pth",
        "GENERATED_DIR": "discrete_generated_sequences",
        "GENERATE_EVERY": 2,
        "FINAL_GENERATE_NUM": 10000
    }

    # 确保目录存在
    os.makedirs(config["GENERATED_DIR"], exist_ok=True)

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据加载（离散编码）
    print("加载数据集...")
    dataset = DiscreteDNADataset(
        file_path=config["DATA_PATH"],
        seq_length=config["SEQ_LENGTH"],
        padding=config["PADDING"]
    )
    dataloader = DataLoader(dataset, batch_size=config["BATCH_SIZE"], shuffle=True)
    print(f"加载完成，样本数: {len(dataset)}, 批次大小: {config['BATCH_SIZE']}")

    # 模型初始化（离散版本）
    denoise_model = DiscreteDenoiseCNN(
        seq_length=config["SEQ_LENGTH"] + 2 * config["PADDING"],
        model_dim=config["MODEL_DIM"]
    ).to(device)
    ddpm = DiscreteDDPM(denoise_model, timesteps=config["TIMESTEPS"]).to(device)

    # 打印模型参数数量
    param_count = sum(p.numel() for p in ddpm.parameters() if p.requires_grad)
    print(f"模型初始化完成，参数数量: {param_count:,}")

    # 优化器
    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=config["LEARNING_RATE"])

    # 训练循环
    print("\n开始训练...")
    for epoch in range(config["NUM_EPOCHS"]):
        epoch_loss = 0
        ddpm.train()

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config['NUM_EPOCHS']}"):
            batch = batch.to(device)  # 离散整数张量 [B, L]
            optimizer.zero_grad()

            # 随机采样时间步
            t = torch.randint(0, config["TIMESTEPS"], (batch.shape[0],), device=device).long()
            loss = ddpm.p_losses(batch, t)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{config['NUM_EPOCHS']}, 平均损失: {avg_loss:.6f}")

        # 生成样本
        if (epoch + 1) % config["GENERATE_EVERY"] == 0:
            ddpm.eval()
            with torch.no_grad():
                generated = ddpm.sample(
                    batch_size=config["BATCH_SIZE"],
                    seq_length=config["SEQ_LENGTH"] + 2 * config["PADDING"]
                )

            # 保存生成的序列
            epoch_save_path = os.path.join(
                config["GENERATED_DIR"],
                f"epoch_{epoch + 1}_generated.txt"
            )
            with open(epoch_save_path, 'w') as f:
                for seq in generated:
                    seq_str = discrete_to_sequence(seq, config["SEQ_LENGTH"])
                    f.write(seq_str + '\n')

            print(f"已保存 {config['BATCH_SIZE']} 条生成序列到 {epoch_save_path}")

    # 保存模型
    torch.save(ddpm.state_dict(), config["SAVE_MODEL_PATH"])
    print(f"\n模型已保存至 {config['SAVE_MODEL_PATH']}")

    # 生成最终序列
    ddpm.eval()
    with torch.no_grad():
        final_generated = ddpm.sample(
            batch_size=config["FINAL_GENERATE_NUM"],
            seq_length=config["SEQ_LENGTH"] + 2 * config["PADDING"]
        )

    final_save_path = os.path.join(
        config["GENERATED_DIR"],
        "final_generated.txt"
    )
    with open(final_save_path, 'w') as f:
        for seq in final_generated:
            seq_str = discrete_to_sequence(seq, config["SEQ_LENGTH"])
            f.write(seq_str + '\n')

    print(f"\n已保存 {config['FINAL_GENERATE_NUM']} 条最终生成序列到 {final_save_path}")


if __name__ == "__main__":
    train_discrete_ddpm()