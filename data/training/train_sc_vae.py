# train_vae_torch.py
from scipy.stats import pearsonr
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import os
# 导入数据加载类
# 从 vae_model_torch.py 中导入 PyTorch VAE 模型类
from data.models.sc_vae_model import VAE # 导入 PyTorch 版本的 VAE 类

from torch.utils.data import DataLoader
# 导入学习率调度器类 (如果使用)
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
# 导入 tqdm 用于显示进度
from tqdm import tqdm
# 导入用于 k-mer 计算的 Counter (如果你在 calculate_kmer_similarity 中使用)
from collections import Counter
# 导入用于 Pearson 相关系数计算的 scipy.stats (如果你在 calculate_kmer_similarity 中使用)


# 确保导入了 SequenceDataset 和 decoder2seq_torch 或你自己的序列转换函数
# decoder2seq_torch 可能用于从潜在空间生成序列，此处评估重构时可能需要另一个从one-hot或概率到序列的函数
from data.utils.utils import SequenceDataset # , decoder2seq_torch # 暂时不确定 decoder2seq_torch 是否用于 One-Hot 转换


INDEX_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'T'} # 示例映射，请检查是否与你的数据一致
# --- END PLACEHOLDER ---



def onehot_to_sequence(one_hot_tensor, original_seq_len, padding):
    """
    PLACEHOLDER: 将单条 One-Hot 编码张量转换为序列字符串 (去除 padding)。
    你需要用你实际的实现替换此函数，确保正确处理输入形状并去除 padding。
    假设输入 one_hot_tensor 是一个单条序列的 One-Hot 张量，形状通常是 (C, H, W) 或 (C, W)。
    例如 (1, 4, 185) 或 (4, 185)。
    """
    seq_len_with_padding = one_hot_tensor.shape[-1] # Assume sequence length is the last dimension
    start_index = padding
    end_index = padding + original_seq_len

    # Adjust shape to be (4, sequence_length_with_padding)
    if one_hot_tensor.ndim == 3: # e.g., (1, 4, 185) or (4, 1, 185)
        if one_hot_tensor.shape[1] == 4: # (1, 4, W)
             one_hot_tensor = one_hot_tensor.squeeze(0) # -> (4, W)
        elif one_hot_tensor.shape[0] == 4: # (4, 1, W)
             one_hot_tensor = one_hot_tensor.squeeze(1) # -> (4, W)
        else:
             print(f"WARNING: Cannot squeeze unexpected 3D shape for onehot_to_sequence: {one_hot_tensor.shape}")
             return "UNEXPECTED_SHAPE_SEQ_ORIG"
    elif one_hot_tensor.ndim != 2 or one_hot_tensor.shape[0] != 4: # Expected (4, W)
         print(f"WARNING: Unexpected shape for onehot_to_sequence: {one_hot_tensor.shape}")
         return "UNEXPECTED_SHAPE_SEQ_ORIG"


    # Now one_hot_tensor should be shape (4, sequence_length_with_padding)
    if one_hot_tensor.shape[0] != 4:
         print(f"WARNING: Shape after squeezing is not (4, W) in onehot_to_sequence: {one_hot_tensor.shape}")
         return "UNEXPECTED_SHAPE_SEQ_ORIG"

    # Find the index with the maximum value along the base dimension (dim=0)
    indices = torch.argmax(one_hot_tensor, dim=0)

    # Remove padding
    core_sequence_indices = indices[start_index:end_index]

    # Convert indices to base characters
    sequence_string = "".join([INDEX_TO_BASE.get(i.item(), 'N') for i in core_sequence_indices]) # Use .get() with default 'N'

    return sequence_string


# --- PLACEHOLDER: 将重构的概率张量转换为序列字符串 ---
# 你需要根据 VAE decoder 的输出格式来实现这个函数
# 它应该接收一个形状类似于 (4, sequence_length) 或 (1, 4, sequence_length) 的概率张量
# 并返回对应的碱基字符串 (核心序列，去除 padding)
def reconstructed_to_sequence(reconstructed_tensor, original_seq_len, padding):
    """
    PLACEHOLDER: 将单条重构概率张量转换为序列字符串 (去除 padding)。
    你需要用你实际的实现替换此函数，确保正确处理输入形状并去除 padding。
    假设输入 reconstructed_tensor 是一个单条序列的概率张量 (sigmoid 激活后)，
    形状通常是 (C, H, W) 或 (C, W)。 例如 (1, 4, 185) 或 (4, 185)。
    """
    seq_len_with_padding = reconstructed_tensor.shape[-1] # Assume sequence length is the last dimension
    start_index = padding
    end_index = padding + original_seq_len

    # Adjust shape to be (4, sequence_length_with_padding)
    if reconstructed_tensor.ndim == 3: # e.g., (1, 4, 185) or (4, 1, 185)
        if reconstructed_tensor.shape[1] == 4: # (1, 4, W)
             reconstructed_tensor = reconstructed_tensor.squeeze(0) # -> (4, W)
        elif reconstructed_tensor.shape[0] == 4: # (4, 1, W)
             reconstructed_tensor = reconstructed_tensor.squeeze(1) # -> (4, W)
        else:
             print(f"WARNING: Cannot squeeze unexpected 3D shape for reconstructed_to_sequence: {reconstructed_tensor.shape}")
             return "UNEXPECTED_SHAPE_SEQ_RECON"
    elif reconstructed_tensor.ndim != 2 or reconstructed_tensor.shape[0] != 4: # Expected (4, W)
        print(f"WARNING: Unexpected shape for reconstructed_to_sequence: {reconstructed_tensor.shape}")
        return "UNEXPECTED_SHAPE_SEQ_RECON"

    # Now reconstructed_tensor should be shape (4, sequence_length_with_padding)
    if reconstructed_tensor.shape[0] != 4:
        print(f"WARNING: Shape after squeezing is not (4, W) in reconstructed_to_sequence: {reconstructed_tensor.shape}")
        return "UNEXPECTED_SHAPE_SEQ_RECON"


    # Find the index with the maximum probability along the base dimension (dim=0)
    indices = torch.argmax(reconstructed_tensor, dim=0)

    # Remove padding
    core_sequence_indices = indices[start_index:end_index]

    # Convert indices to base characters
    sequence_string = "".join([INDEX_TO_BASE.get(i.item(), 'N') for i in core_sequence_indices]) # Use .get() with default 'N'
    print(f"seq{sequence_string}")
    return sequence_string


# --- PLACEHOLDER: 用于计算 k-mer 相似度的函数 ---
# 你需要根据你的 k-mer 计算逻辑来实现这个函数
# 它应该接收两个序列字符串和 k 值
def calculate_kmer_similarity(seq1, seq2, k=6, similarity_metric="pearson"):
    """
    计算两个序列的k-mer相似度，支持多种相似度指标。

    参数:
        seq1 (str): 序列1（仅含A/T/C/G/N，不区分大小写）
        seq2 (str): 序列2（仅含A/T/C/G/N，不区分大小写）
        k (int): k-mer的长度（k≥1）
        similarity_metric (str): 相似度指标，可选 "pearson"（皮尔逊相关系数）或 "cosine"（余弦相似度）

    返回:
        float: 相似度得分（范围因指标而异：皮尔逊[-1,1]，余弦[0,1]）
        np.ecoli_data: 若输入无效（如序列为空、k≤0、序列长度<k、含非法字符）
    """

    # ====================== 输入校验与预处理 ======================
    # 检查k是否为正整数
    if not isinstance(k, int) or k <= 0:
        return np.nan

    # 统一转为大写并检查非法字符
    valid_bases = set("ATCGNatcgn")  # 允许N作为通配符
    for seq in [seq1, seq2]:
        if not seq or any(base not in valid_bases for base in seq):
            return np.nan

    seq1 = seq1.upper()
    seq2 = seq2.upper()

    # 检查序列长度是否足够
    if len(seq1) < k or len(seq2) < k:
        return np.nan

    # ====================== k-mer计数函数（优化版） ======================
    def count_kmers(sequence: str, k: int) -> Counter:
        """高效统计序列中k-mer的出现次数"""
        kmer_counts = Counter()
        max_idx = len(sequence) - k + 1
        for i in range(max_idx):
            kmer = sequence[i:i + k]
            kmer_counts[kmer] += 1
        return kmer_counts

    # ====================== 生成k-mer向量 ======================
    kmer_counts1 = count_kmers(seq1, k)
    kmer_counts2 = count_kmers(seq2, k)

    all_kmers = sorted(set(kmer_counts1.keys()).union(kmer_counts2.keys()))
    if not all_kmers:  # 无有效k-mer（如k=0，但已在前面校验）
        return np.nan

    # 生成频率向量（含出现次数）
    vec1 = np.array([kmer_counts1.get(kmer, 0) for kmer in all_kmers], dtype=np.float64)
    vec2 = np.array([kmer_counts2.get(kmer, 0) for kmer in all_kmers], dtype=np.float64)

    # ====================== 相似度计算 ======================
    if similarity_metric == "pearson":
        # 处理全零向量（相关系数定义为0）
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
        # 计算皮尔逊相关系数（返回值范围[-1, 1]，1为完全正相关）
        corr, _ = pearsonr(vec1, vec2)
        return corr

    elif similarity_metric == "cosine":
        # 余弦相似度公式：(A·B) / (||A|| * ||B||)
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0  # 全零向量相似度定义为0
        return dot_product / (norm1 * norm2)

    else:
        raise ValueError("`similarity_metric` 必须为 'pearson' 或 'cosine'")




# 定义优化器 (PyTorch 版本)
# 这里的 model.parameters() 是在创建 VAE 模型实例后获得的
# optimizer = optim.Adam(model.parameters(), lr=1e-4) # 需要先创建模型实例再定义优化器

# 辅助函数：计算正态分布的对数概率密度 (PyTorch 版本)
def log_normal_pdf_torch(sample, mean, logvar):
    # sample, mean, logvar 应该是 PyTorch 张量
    log2pi = torch.log(torch.tensor(2. * np.pi)).to(sample.device) # 确保常数也在正确的设备上
    return torch.sum(
        -.5 * ((sample - mean) ** 2. * torch.exp(-logvar) + logvar + log2pi),
        dim=1 # 假设 sample, mean, logvar shape is (batch_size, latent_dim)
    )

# 计算 VAE 的损失函数 (PyTorch 版本)
# 修改为返回分量损失以便监控
def compute_loss_torch(model, x, kl_weight=1.0): # 添加 kl_weight 参数
    # x 是经过预处理的输入数据张量 (batch_size, channels, height, width)
    # 例如 (batch_size, 1, 4, 185)
    mean, logvar = model.encode(x) # 确保 encode 返回 (batch_size, latent_dim)
    z = model.reparameterize(mean, logvar) # 确保 z 形状是 (batch_size, latent_dim)
    reconstructed_x_logits = model.decode(z) # 确保 decode 返回 (batch_size, 1, 4, 185)

    # 计算重构损失 (Binary Cross Entropy with Logits)
    reconstruction_loss_elementwise = F.binary_cross_entropy_with_logits(
        reconstructed_x_logits, x, reduction='none'
    )
    # 确保这里的 dim 正确对应你数据的通道、高度和宽度维度 (如果形状是 B, 1, 4, 185，则为 [1, 2, 3])
    reconstruction_loss = torch.sum(reconstruction_loss_elementwise, dim=[1, 2, 3])

    # 计算 KL 散度损失
    logpz = log_normal_pdf_torch(z, torch.tensor(0.0).to(z.device), torch.tensor(0.0).to(z.device))
    logqz_x = log_normal_pdf_torch(z, mean, logvar)
    kl_divergence = logqz_x - logpz # KL(q || p)

    # VAE 损失 = 重构损失 + kl_weight * KL 散度
    total_loss = torch.mean(reconstruction_loss + kl_weight * kl_divergence) # 应用 kl_weight

    return total_loss, torch.mean(reconstruction_loss), torch.mean(kl_divergence) # 返回分量损失


# 定义单步训练过程
def train_step_torch(model, x, optimizer, kl_weight=1.0): # 添加 kl_weight 参数
    model.train() # 设置模型为训练模式
    optimizer.zero_grad() # 梯度清零

    # 前向传播并计算损失
    total_loss, recon_loss, kl_loss = compute_loss_torch(model, x, kl_weight) # 获取分量损失

    # 反向传播计算梯度
    total_loss.backward()

    # 更新模型参数
    optimizer.step()

    return total_loss.item(), recon_loss.item(), kl_loss.item() # 返回损失值

# --- 数据加载和训练循环框架 ---

# 大肠杆菌长度为 165 bp
# 这里酿酒酵母长度为80bp
dataset_path = '/root/autodl-tmp/sjy/dachang/data/raw_sequences/sc_data/SC_gen_1w_seq_short_top50000.txt'
# SequenceDataset 的参数必须与 VAE 编码器期望的输入形状匹配
SEQ_LENGTH_RAW = 80
PADDING_RAW = 0
sequence_dataset = SequenceDataset(dataset_path, seq_length=SEQ_LENGTH_RAW, padding=PADDING_RAW)

# 2. 创建 DataLoader
batch_size = 32 # 设置你的批次大小
train_dataloader = DataLoader(sequence_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True) # 增加 num_workers 和 pin_memory 提高数据加载效率
# 检查数据加载是否成功
if len(sequence_dataset) == 0:
    print("Error: Dataset is empty. Please check your data file path and content.")
    exit()
else:
    print(f"Loaded {len(sequence_dataset)} sequences.")
    # 获取一个批次的形状作为参考，进行输入形状检查
    try:
        sample_batch = next(iter(train_dataloader))
        print(f"Sample batch shape from DataLoader: {sample_batch.shape}")
        # 确保这里的形状与 VAE 编码器期望的输入形状一致
        # 你的日志显示 DataLoader 输出形状是 torch.Size([32, 1, 4, 185])
        expected_shape_vae_input = (batch_size, 1, 4, SEQ_LENGTH_RAW + 2 * PADDING_RAW) # 根据你的日志确定是 (B, C, H, W)
        if sample_batch.shape != expected_shape_vae_input:
             print(f"\n警告: DataLoader 输出的批次形状 {sample_batch.shape} 可能与 VAE 模型期望的输入形状 {expected_shape_vae_input} 不符。")
             print("请检查 SequenceDataset 的实现是否正确返回 (channels, height, width)，以及 DataLoader 如何添加批次维度。")

    except StopIteration:
         print("Error: DataLoader is empty.")
    except Exception as e:
         print(f"检查 DataLoader 样本批次时出错: {e}")


# 3. 设置训练参数
latent_dim = 256 # VAE 的潜在空间维度
epochs = 400 # <--- 减少训练轮数以快速测试，或者保持 10000 并使用检查点
# --- 设置设备 ---
device = torch.device("cuda:1" if torch.cuda.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")
if device.type == 'cuda':
    print(f"使用的 GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")


# 4. 创建 VAE 模型实例
model = VAE(latent_dim)
# 将模型移到指定设备
model.to(device)

# 5. 定义优化器
# *** 调整 VAE 学习率 ***
# 尝试不同的基础学习率，例如 1e-4, 5e-5, 1e-5
LEARNING_RATE_VAE = 1e-4 # <-- 示例：VAE 的基础学习率
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_VAE)

# 6. VAE 学习率调度器 (可选)
USE_LR_SCHEDULER_VAE = True # 是否使用 VAE 学习率调度器
# Step Decay 示例参数
STEP_DECAY_EPOCHS_VAE = 100 # 每隔多少个 epoch 衰减一次
STEP_DECAY_GAMMA_VAE = 0.5  # 每次衰减的比例
# Cosine Annealing 示例参数
T_MAX_COSINE_VAE = epochs # Cosine Annealing 的总周期设置为总轮数
ETA_MIN_COSINE_VAE = 1e-6 # 学习率退火到的最小值

scheduler_vae = None
if USE_LR_SCHEDULER_VAE:
    # 示例：使用 StepLR
    scheduler_vae = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_DECAY_EPOCHS_VAE, gamma=STEP_DECAY_GAMMA_VAE)
    print(f"使用 VAE Step Decay 学习率调度器: 每 {STEP_DECAY_EPOCHS_VAE} 轮学习率乘以 {STEP_DECAY_GAMMA_VAE}")

    # 示例：使用 Cosine Annealing (取消下面注释并注释掉 StepLR 来启用)
    # scheduler_vae = CosineAnnealingLR(optimizer, T_max=T_MAX_COSINE_VAE, eta_min=ETA_MIN_COSINE_VAE)
    # print(f"使用 VAE Cosine Annealing 学习率调度器: T_max={T_MAX_COSINE_VAE}, eta_min={ETA_MIN_COSINE_VAE}")


# 7. VAE 训练循环
print("Starting VAE training...")
# KL 散度权重 Annealing (可选)
START_KL_WEIGHT = 0.0
END_KL_WEIGHT = 1.0
KL_ANNEAL_EPOCHS = 1000 # Set KL Annealing epochs to match total epochs for full annealing
kl_weight = 1.0 # Default KL weight is 1.0

for epoch in range(epochs):
    start_time = time.time()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0

    # 计算当前 epoch 的 KL 权重 (如果使用 Annealing)
    kl_weight = START_KL_WEIGHT + (END_KL_WEIGHT - START_KL_WEIGHT) * min(1.0, epoch / KL_ANNEAL_EPOCHS)


    # 遍历 DataLoader 获取批次数据
    for i, x_batch in enumerate(train_dataloader):
        x_batch=x_batch.to(device)

        # 执行单步训练，获取分量损失
        loss, recon_loss, kl_loss = train_step_torch(model, x_batch, optimizer, kl_weight)

        total_loss += loss
        total_recon_loss += recon_loss
        total_kl_loss += kl_loss


        # 可以打印批次损失 (可选)
        # if (i + 1) % 100 == 0: # 每 100 个批次打印一次
        #     print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_dataloader)}], Batch Total Loss: {loss:.4f}, Recon Loss: {recon_loss:.4f}, KL Loss: {kl_loss:.4f}')


    end_time = time.time()
    avg_total_loss = total_loss / len(train_dataloader)
    avg_recon_loss = total_recon_loss / len(train_dataloader)
    avg_kl_loss = total_kl_loss / len(train_dataloader)

    print(f'Epoch {epoch + 1}/{epochs}, Avg Total Loss: {avg_total_loss:.4f}, Avg Recon Loss: {avg_recon_loss:.4f}, Avg KL Loss: {avg_kl_loss:.4f}, Time: {end_time - start_time:.2f}s')
    print(f'Current KL Weight: {kl_weight:.4f}') # 如果使用 KL Annealing，打印当前权重


    # 学习率调度器步进
    if scheduler_vae is not None:
        scheduler_vae.step()
        # print(f"当前 VAE 学习率: {optimizer.param_groups[0]['lr']:.6f}") # 如果需要在每个 epoch 打印学习率可以取消注释


# 训练完成后打印信息
print("\nVAE 训练完成。")

# 7. 训练完成后，保存模型
save_path = '../my_vae_model_state_dict.pth'  # 根据你的目录结构调整保存路径
os.makedirs(os.path.dirname(save_path), exist_ok=True) # 创建目录如果不存在
torch.save(model.state_dict(), save_path) # 保存模型参数字典
print(f"VAE Model saved to {save_path}")


# --- VAE 重构能力评估部分 ---
print("\n--- 开始评估 VAE 重构能力 ---")

# 加载训练好的 VAE 模型 (如果在上面已经加载并设置 eval 模式，这里可以跳过加载步骤)
# model = VAE(latent_dim=YOUR_LATENT_DIM) # Instantiate VAE with the correct latent_dim
# model.load_state_dict(torch.load(save_path, map_location=device))
model.to(device)
model.eval() # Set model to evaluation mode

# 加载用于评估的真实序列数据 (可以使用训练数据集，但最好是独立的验证集)
# 我们这里为了演示方便，复用 train_dataloader 的 dataset，但建议使用单独的评估数据集和 DataLoader
eval_dataset = sequence_dataset # 复用训练数据集进行评估
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True) # 评估时不打乱数据

all_original_sequences = []
all_reconstructed_sequences = []

# 用于计算平均 k-mer 相似度的列表
k = 6 # 你的 k-mer 大小，根据你需要评估的 k 值修改
kmer_similarities_list = []


with torch.no_grad(): # 评估过程不计算梯度
    for batch_original_data in tqdm(eval_dataloader, desc="评估 VAE 重构"):
        batch_original_data = batch_original_data.to(device) # 移动到设备

        # Encode
        mean, logvar = model.encode(batch_original_data)
        # 在评估重构时，通常直接使用均值作为潜在表示，而不是从分布中采样
        # 如果你特别想评估采样+解码的重构能力，可以使用 reparameterize
        # z = model.reparameterize(mean, logvar)
        z = mean # 使用均值进行确定性解码

        # Decode to get probabilities
        reconstructed_probs = model.decode(z, apply_sigmoid=True) # Shape (batch_size, 1, 4, 185)


        # --- 将原始 One-Hot 编码转换为序列字符串 ---
        # 原始数据形状是 (batch_size, 1, 4, 185)
        # 遍历批次中的每个样本
        for i in range(batch_original_data.size(0)):
            # onehot_to_sequence 接收一个形状类似于 (4, 185) 的张量
            # Squeeze height dimension (dim=1) to get (batch_size, 4, 185)
            # Then take i-th sample, shape (4, 185)
            original_seq_str = onehot_to_sequence(batch_original_data[i].squeeze(0), SEQ_LENGTH_RAW, PADDING_RAW)
            all_original_sequences.append(original_seq_str)


        # --- 将重构概率转换为序列字符串 ---
        # 重构数据形状是 (batch_size, 1, 4, 185)
        # 遍历批次中的每个样本
        for i in range(reconstructed_probs.size(0)):
             # reconstructed_to_sequence 接收一个形状类似于 (4, 185) 的张量
             # Squeeze height dimension (dim=1) to get (batch_size, 4, 185)
             # Then take i-th sample, shape (4, 185)
             reconstructed_seq_str = reconstructed_to_sequence(reconstructed_probs[i].squeeze(0), SEQ_LENGTH_RAW, PADDING_RAW)
             all_reconstructed_sequences.append(reconstructed_seq_str)



# --- 计算并打印最终评估指标 ---
print("\n计算 VAE 重构评估指标...")

# 计算平均 k-mer 相似度 (例如 Pearson 相关系数)
# 在收集所有序列后计算更可靠
if len(all_original_sequences) == len(all_reconstructed_sequences) and len(all_original_sequences) > 0:
    kmer_similarities_list = [] # Reset list to compute based on all collected sequences
    for original_seq, reconstructed_seq in tqdm(zip(all_original_sequences, all_reconstructed_sequences), total=len(all_original_sequences), desc=f"计算 K-mer 相似度 (k={k})"):
        try:
            # 你需要你的函数 calculate_kmer_similarity 来计算相似度
            similarity = calculate_kmer_similarity(original_seq, reconstructed_seq, k=k, similarity_metric="pearson")
            kmer_similarities_list.append(similarity)
        except Exception as e:
            # print(f"Warning: Error calculating k-mer similarity for a sequence pair: {e}. Skipping.")
            kmer_similarities_list.append(np.nan) # Append NaN if calculation fails


average_kmer_similarity = 0.0
if kmer_similarities_list:
    average_kmer_similarity = np.nanmean(kmer_similarities_list) # Use nanmean to ignore potential NaN results

print(f"--- VAE 重构评估结果 ---")
print(f"处理的序列总数: {len(all_original_sequences)}")
print(f"成功/尝试计算 K-mer 相似度的序列对数量: {len(kmer_similarities_list)}") # Changed text
print(f"平均 VAE 重构 K-mer 相似度 (k={k}): {average_kmer_similarity:.4f}")



print("\n--- VAE 重构能力评估结束 ---")
