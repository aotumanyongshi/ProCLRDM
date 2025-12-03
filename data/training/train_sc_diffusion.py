import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适用于生成图像文件
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import shutil
import random

# 导入自定义的模型和工具函数
from data.models.sc_vae_model import VAE
from data.models.sc_diffusion_model  import LatentDiffusionModel, TransformerDenoiseModel
from data.utils.utils import SequenceDataset

# --- 设置设备 --- (提前到导入模块后立即设置)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cuda:1" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")
if device.type == 'cuda':
    print(f"使用的 GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# --- 碱基索引到字符的映射 ---
INDEX_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}


# --- 将 One-Hot 编码张量转换为序列字符串 ---
def onehot_to_sequence(one_hot_tensor, original_seq_len, padding):
    """将单条 One-Hot 编码张量转换为序列字符串 (去除 padding)"""
    if one_hot_tensor.device.type != 'cpu':
        one_hot_tensor = one_hot_tensor.cpu()

    if one_hot_tensor.ndim == 3 and one_hot_tensor.shape[0] == 1:
        one_hot_tensor = one_hot_tensor.squeeze(0)
    if one_hot_tensor.ndim == 2 and one_hot_tensor.shape[0] == 1:
        one_hot_tensor = one_hot_tensor.squeeze(0)

    if one_hot_tensor.ndim != 2 or one_hot_tensor.shape[0] != 4:
        if one_hot_tensor.ndim == 3 and one_hot_tensor.shape[0] == 4 and one_hot_tensor.shape[1] == 1:
            one_hot_tensor = one_hot_tensor.squeeze(1)
            if one_hot_tensor.ndim != 2:
                print(f"ERROR: Shape after handling (4, 1, W) still unexpected: {one_hot_tensor.shape}")
                return "UNEXPECTED_SHAPE_SEQ_ORIG"
        else:
            print(f"ERROR: Unexpected tensor shape in onehot_to_sequence: {one_hot_tensor.shape}")
            return "UNEXPECTED_SHAPE_SEQ_ORIG"

    seq_len_with_padding = one_hot_tensor.shape[-1]
    start_index = padding
    end_index = padding + original_seq_len

    if start_index < 0 or end_index > seq_len_with_padding:
        print(
            f"WARNING: Padding/SeqLen mismatch. Padding: {padding}, Original Seq Len: {original_seq_len}, Padded Seq Len: {seq_len_with_padding}")
        start_index = max(0, start_index)
        end_index = min(seq_len_with_padding, end_index)

    indices = torch.argmax(one_hot_tensor, dim=0)
    core_sequence_indices = indices[start_index:end_index]
    sequence_string = "".join([INDEX_TO_BASE.get(i.item(), 'N') for i in core_sequence_indices])
    return sequence_string


# --- 将重构的概率张量转换为序列字符串 ---
def reconstructed_to_sequence(reconstructed_tensor, original_seq_len, padding):
    """将单条重构概率张量转换为序列字符串 (去除 padding)"""
    if reconstructed_tensor.device.type != 'cpu':
        reconstructed_tensor = reconstructed_tensor.cpu()

    if reconstructed_tensor.ndim == 3 and reconstructed_tensor.shape[0] == 1:
        reconstructed_tensor = reconstructed_tensor.squeeze(0)
    if reconstructed_tensor.ndim == 2 and reconstructed_tensor.shape[0] == 1:
        reconstructed_tensor = reconstructed_tensor.squeeze(0)

    if reconstructed_tensor.ndim != 2 or reconstructed_tensor.shape[0] != 4:
        if reconstructed_tensor.ndim == 3 and reconstructed_tensor.shape[0] == 4 and reconstructed_tensor.shape[1] == 1:
            reconstructed_tensor = reconstructed_tensor.squeeze(1)
            if reconstructed_tensor.ndim != 2:
                print(f"ERROR: Shape after handling (4, 1, W) still unexpected: {reconstructed_tensor.shape}")
                return "UNEXPECTED_SHAPE_SEQ_RECON"
        else:
            print(f"ERROR: Unexpected tensor shape in reconstructed_to_sequence: {reconstructed_tensor.shape}")
            return "UNEXPECTED_SHAPE_SEQ_RECON"

    seq_len_with_padding = reconstructed_tensor.shape[-1]
    start_index = padding
    end_index = padding + original_seq_len

    if start_index < 0 or end_index > seq_len_with_padding:
        print(
            f"WARNING: Padding/SeqLen mismatch in reconstructed_to_sequence. Padding: {padding}, Original Seq Len: {original_seq_len}, Padded Seq Len: {seq_len_with_padding}")
        start_index = max(0, start_index)
        end_index = min(seq_len_with_padding, end_index)

    indices = torch.argmax(reconstructed_tensor, dim=0)
    core_sequence_indices = indices[start_index:end_index]
    sequence_string = "".join([INDEX_TO_BASE.get(i.item(), 'N') for i in core_sequence_indices])
    return sequence_string


# --- 新增K-mer频率计算函数（替换原有逻辑） ---
def calculate_kmer_frequency(sequence, k):
    """计算单条序列的K-mer频率（频率=计数/总K-mer数）"""
    kmer_count = {}
    total_kmers = len(sequence) - k + 1
    if total_kmers <= 0:  # 序列长度小于k，无法计算
        return {}

    for i in range(total_kmers):
        kmer = sequence[i:i+k]
        if kmer in kmer_count:
            kmer_count[kmer] += 1
        else:
            kmer_count[kmer] = 1

    # 计算频率（计数/总K-mer数）
    kmer_frequency = {kmer: count / total_kmers for kmer, count in kmer_count.items()}
    return kmer_frequency


def get_kmer_frequencies_for_list(sequences, k):
    """计算序列列表的总K-mer频率（汇总所有序列的K-mer计数后计算频率）"""
    total_kmer_count = {}
    total_kmers_across_sequences = 0  # 所有序列的总K-mer数量

    for seq in sequences:
        if not isinstance(seq, str) or len(seq) < k:
            continue  # 跳过无效序列
        # 计算单条序列的K-mer计数
        seq_kmer_count = {}
        seq_kmers = len(seq) - k + 1
        total_kmers_across_sequences += seq_kmers  # 累加总K-mer数
        for i in range(seq_kmers):
            kmer = seq[i:i+k]
            if kmer in seq_kmer_count:
                seq_kmer_count[kmer] += 1
            else:
                seq_kmer_count[kmer] = 1
        # 汇总到总计数
        for kmer, count in seq_kmer_count.items():
            if kmer in total_kmer_count:
                total_kmer_count[kmer] += count
            else:
                total_kmer_count[kmer] = count

    if total_kmers_across_sequences == 0:  # 没有有效K-mer
        return {}

    # 计算总频率（总计数/总K-mer数）
    total_kmer_frequency = {
        kmer: count / total_kmers_across_sequences
        for kmer, count in total_kmer_count.items()
    }
    return total_kmer_frequency


def compare_kmer_frequencies(gen_sequences, natural_sequences, k, save_path):
    """对比生成序列和自然序列的K-mer频率，生成散点图并保存"""
    # 计算两组序列的总K-mer频率
    gen_freq = get_kmer_frequencies_for_list(gen_sequences, k)
    natural_freq = get_kmer_frequencies_for_list(natural_sequences, k)

    # 获取所有K-mer的集合（确保覆盖两组序列）
    all_kmers = set(gen_freq.keys()).union(set(natural_freq.keys()))
    if not all_kmers:
        print("WARNING: No valid kmers to compare")
        return

    # 提取频率值
    x = [gen_freq.get(kmer, 0.0) for kmer in all_kmers]  # 生成序列频率
    y = [natural_freq.get(kmer, 0.0) for kmer in all_kmers]  # 自然序列频率

    # 绘制散点图
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, alpha=0.5, s=10)
    plt.xlabel(f'Generated Sequences {k}-mer Frequency')
    plt.ylabel(f'Natural Sequences {k}-mer Frequency')
    plt.title(f'Comparison of {k}-mer Frequencies')

    # 设置横纵轴范围一致
    max_val = max(max(x), max(y)) * 1.1  # 留一点余量
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    # 添加参考线（y=x）
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.8)

    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()  # 关闭图像，避免内存占用


def calculate_kmer_distribution_correlation(seq_list1, seq_list2, k, similarity_metric="pearson"):
    """计算两个序列列表的K-mer频率分布相关性（替换原有函数）"""
    if not seq_list1 or not seq_list2:
        print("Error: One or both sequence lists are empty for K-mer evaluation.")
        return np.nan

    # 计算两组序列的总K-mer频率
    freq1 = get_kmer_frequencies_for_list(seq_list1, k)
    freq2 = get_kmer_frequencies_for_list(seq_list2, k)

    if not freq1 or not freq2:
        print("Warning: No valid k-mer frequencies computed for one or both sequence lists.")
        return np.nan

    # 获取共同K-mer（确保有重叠）
    common_kmers = set(freq1.keys()) & set(freq2.keys())
    if not common_kmers:
        print(f"Warning: No common {k}-mers between the two sequence lists.")
        return np.nan

    # 提取共同K-mer的频率
    x = np.array([freq1[kmer] for kmer in common_kmers])
    y = np.array([freq2[kmer] for kmer in common_kmers])

    # 计算皮尔逊相关系数
    if similarity_metric == "pearson":
        correlation, _ = pearsonr(x, y)
        return correlation
    else:
        raise ValueError(f"Unsupported similarity metric: {similarity_metric}")


# --- 其他原有函数保持不变 ---
def get_condition_strength(current_epoch):
    """计算当前Epoch的条件干预强度"""
    if current_epoch < CONDITION_START_EPOCH:
        return CONDITION_MIN_STRENGTH
    elif current_epoch > CONDITION_FULL_EPOCH:
        return CONDITION_MAX_STRENGTH
    else:
        progress = (current_epoch - CONDITION_START_EPOCH) / (CONDITION_FULL_EPOCH - CONDITION_START_EPOCH)
        return CONDITION_MIN_STRENGTH + progress * (CONDITION_MAX_STRENGTH - CONDITION_MIN_STRENGTH)


# --- 超参数设置（保持不变） ---
# VAE 参数
VAE_LATENT_DIM = 256
VAE_MODEL_PATH = '/root/autodl-tmp/sjy/dachang/data/my_vae_model_state_dict.pth'

# 数据路径
DATA_PATH = '/root/autodl-tmp/sjy/dachang/data/raw_sequences/sc_data/SC_gen_1w_seq_short_top50000.txt'
EXPRESSION_DATA_PATH = '/root/autodl-tmp/sjy/dachang/data/raw_sequences/sc_data/SC_gen_1w_exp_short_expression_top50000.txt'

# 序列参数
SEQ_LENGTH_RAW = 80
PADDING_RAW = 0
SEQ_LENGTH_PADDED = SEQ_LENGTH_RAW + 2 * PADDING_RAW

# Diffusion Model 参数
DIFFUSION_MODEL_DIM = 1024
DIFFUSION_NUM_LAYERS = 6
TIME_EMB_DIM = 128
TIMESTEPS = 1000
BETA_SCHEDULE = 'cosine'
EXPRESSION_DIM = 1
EXPRESSION_STD_TARGET_SCALE = 0.9

# 训练参数
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 500

# 学习率调度器
USE_LR_SCHEDULER = True
T_MAX_COSINE = NUM_EPOCHS
ETA_MIN_COSINE = 6e-7

# 检查点目录
CHECKPOINT_DIR = 'conditional_diffusion_checkpoints_latent_vector_256dim_cfg_tuned_early_stopping'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
# 创建K-mer图像保存子目录
KMER_PLOT_DIR = os.path.join(CHECKPOINT_DIR, 'kmer_plots')
os.makedirs(KMER_PLOT_DIR, exist_ok=True)

# 稳健K-mer评估参数（使用新的K-mer计算方式）
USE_ROBUST_KMER_EVAL = True
K_FOR_ROBUST_EVALUATION = 6
NUM_SAMPLES_FOR_ROBUST_EVAL = 10000
ROBUST_EVAL_FREQUENCY_EPOCHS = 50

# 其他参数
PRINT_GENERATED_SEQUENCES = False
VALIDATION_SPLIT_RATIO = 0.15
EARLY_STOPPING_PATIENCE = 5
CONDITION_START_EPOCH = 100
CONDITION_FULL_EPOCH = 250
CONDITION_MIN_STRENGTH = 0.0
CONDITION_MAX_STRENGTH = 1.0


# --- 加载VAE模型（保持不变） ---
if not os.path.exists(VAE_MODEL_PATH):
    print(f"错误: 训练好的 VAE 模型文件未找到于 {VAE_MODEL_PATH}。")
    print("请先运行 VAE 训练脚本生成此文件。")
    exit()

try:
    vae_model = VAE(latent_dim=VAE_LATENT_DIM).to(device)
    print(f"正在加载 VAE 模型状态字典从 {VAE_MODEL_PATH}...")
    original_state_dict = torch.load(VAE_MODEL_PATH, map_location=device)

    new_state_dict = {}
    prefix = "vae."
    if original_state_dict:
        first_key = next(iter(original_state_dict.keys()))
        if first_key.startswith(prefix):
            for key, value in original_state_dict.items():
                new_state_dict[key[len(prefix):]] = value
        else:
            new_state_dict = original_state_dict

    vae_model.load_state_dict(new_state_dict, strict=True)
    print(f"成功加载 VAE 模型状态字典自 {VAE_MODEL_PATH}")
except RuntimeError as load_error:
    print(f"加载 VAE 状态字典时发生错误: {load_error}")
    exit()
except Exception as e:
    print(f"加载 VAE 模型时发生未知错误: {e}")
    exit()

vae_model.eval()
for param in vae_model.parameters():
    param.requires_grad = False


# --- 数据集加载和分割（保持不变） ---
class RawSequenceExpressionDataset(Dataset):
    def __init__(self, raw_data_path, expression_data_path, seq_length, padding):
        self.seq_length = seq_length
        self.padding = padding
        self.padded_seq_length = seq_length + 2 * padding

        # 加载原始序列
        try:
            self.sequence_dataset = SequenceDataset(raw_data_path, seq_length=seq_length, padding=padding)
            self.raw_sequences_list = []
            print(f"正在将 One-Hot 编码转换回原始序列字符串...")
            temp_dataloader = DataLoader(self.sequence_dataset, batch_size=512, shuffle=False)
            for batch_one_hot in tqdm(temp_dataloader, desc="Converting one-hot to sequences"):
                for one_hot_tensor in batch_one_hot:
                    self.raw_sequences_list.append(onehot_to_sequence(one_hot_tensor, self.seq_length, self.padding))
            print(f"成功转换 {len(self.raw_sequences_list)} 条原始序列字符串。")
        except Exception as e:
            print(f"错误: 加载原始序列数据时发生错误: {e}")
            self.sequence_dataset = []
            self.raw_sequences_list = []
            self.expression_data_tensor = torch.empty(0, 1)
            return

        # 加载表达值
        expression_values = []
        try:
            with open(expression_data_path, 'r') as f:
                for i, line in enumerate(f):
                    try:
                        value = float(line.strip())
                        expression_values.append(torch.tensor([value], dtype=torch.float32))
                    except ValueError:
                        print(f"警告: 无法转换表达值 '{line.strip()}' (行 {i + 1})")
                        continue
            self.expression_data_tensor = torch.stack(expression_values) if expression_values else torch.empty(0, 1)
            print(f"成功加载 {len(self.expression_data_tensor)} 条表达值数据。")
        except FileNotFoundError:
            print(f"错误: 表达值文件未找到于 {expression_data_path}")
            self.sequence_dataset = []
            self.raw_sequences_list = []
            self.expression_data_tensor = torch.empty(0, 1)
            return
        except Exception as e:
            print(f"加载表达值文件时发生错误: {e}")
            self.sequence_dataset = []
            self.raw_sequences_list = []
            self.expression_data_tensor = torch.empty(0, 1)
            return

        # 检查数量匹配
        if len(self.sequence_dataset) != len(self.expression_data_tensor):
            print(f"错误: 序列数量 ({len(self.sequence_dataset)}) 与表达值数量 ({len(self.expression_data_tensor)}) 不匹配。")
            self.sequence_dataset = []
            self.raw_sequences_list = []
            self.expression_data_tensor = torch.empty(0, 1)
            return

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        return self.sequence_dataset[idx], self.expression_data_tensor[idx]


# 加载完整数据集并分割
print("\n正在加载完整原始数据集 (序列和表达值) 进行分割...")
full_raw_dataset = RawSequenceExpressionDataset(DATA_PATH, EXPRESSION_DATA_PATH, SEQ_LENGTH_RAW, PADDING_RAW)

if len(full_raw_dataset) == 0:
    print("\n错误：加载的原始数据集为空。")
    exit()

# 分割训练集和验证集
full_data_size = len(full_raw_dataset)
validation_size = int(full_data_size * VALIDATION_SPLIT_RATIO)
train_size = full_data_size - validation_size
train_subset, validation_subset = random_split(full_raw_dataset, [train_size, validation_size])
print(f"数据集已分割: 训练集 {len(train_subset)} 条, 验证集 {len(validation_subset)} 条。")

# 提取训练集和验证集数据
train_raw_onehot_tensors = [train_subset[i][0] for i in range(len(train_subset))]
train_raw_expressions_tensor = torch.stack([train_subset[i][1] for i in range(len(train_subset))])
train_raw_sequences_list = [full_raw_dataset.raw_sequences_list[i] for i in train_subset.indices]

validation_raw_onehot_tensors = [validation_subset[i][0] for i in range(len(validation_subset))]
validation_raw_expressions_tensor = torch.stack([validation_subset[i][1] for i in range(len(validation_subset))])
validation_original_sequences_for_robust_eval = [full_raw_dataset.raw_sequences_list[i] for i in validation_subset.indices]


# --- VAE编码潜在空间（保持不变） ---
def encode_sequences_to_latent(onehot_tensors_list, vae_model, device, padded_seq_len, batch_size=BATCH_SIZE):
    latent_data = []
    if not onehot_tensors_list: return []

    class TempOneHotDataset(Dataset):
        def __init__(self, tensors): self.tensors = tensors
        def __len__(self): return len(self.tensors)
        def __getitem__(self, idx): return self.tensors[idx]

    temp_dataloader = DataLoader(TempOneHotDataset(onehot_tensors_list), batch_size=batch_size, shuffle=False, num_workers=4)

    vae_model.eval()
    with torch.no_grad():
        for batch_raw_data in tqdm(temp_dataloader, desc="Encoding sequences to latent"):
            batch_raw_data = batch_raw_data.to(device)
            expected_shape = (batch_raw_data.shape[0], 1, 4, padded_seq_len)
            if batch_raw_data.shape != expected_shape:
                print(f"Warning: Unexpected batch shape {batch_raw_data.shape} (expected {expected_shape})")
                continue

            try:
                mean, logvar = vae_model.encode(batch_raw_data)
                mu_batch = mean
                latent_data.extend([mu.cpu() for mu in mu_batch])
            except Exception as e:
                print(f"VAE 编码批次时发生错误: {e}")
                continue

    return latent_data


print("正在使用 VAE 编码器将训练集原始数据编码到潜在空间...")
train_latent_data_list = encode_sequences_to_latent(train_raw_onehot_tensors, vae_model, device, SEQ_LENGTH_PADDED)
print("正在使用 VAE 编码器将验证集原始数据编码到潜在空间...")
validation_latent_data_list = encode_sequences_to_latent(validation_raw_onehot_tensors, vae_model, device, SEQ_LENGTH_PADDED)

# 检查编码后的数据长度
assert len(train_latent_data_list) == len(train_raw_expressions_tensor), "训练集潜在向量与表达值数量不匹配"
assert len(validation_latent_data_list) == len(validation_raw_expressions_tensor), "验证集潜在向量与表达值数量不匹配"


# --- 潜在空间数据集（保持不变） ---
class PreEncodedLatentDataset(Dataset):
    def __init__(self, latent_data_list, expressions_tensor):
        self.latent_data = latent_data_list
        self.expression_data = expressions_tensor
        if len(self.latent_data) != len(self.expression_data):
            raise ValueError(f"Mismatched lengths! latent_data={len(self.latent_data)}, expressions={len(self.expression_data)}")
        if len(self.latent_data) > 0:
            self.latent_vector_dim = self.latent_data[0].shape[-1]
            self.expression_dim = self.expression_data.shape[-1]
        else:
            self.latent_vector_dim = VAE_LATENT_DIM
            self.expression_dim = EXPRESSION_DIM

    def __len__(self):
        return len(self.latent_data)

    def __getitem__(self, idx):
        return self.latent_data[idx], self.expression_data[idx]


# 实例化数据集和DataLoader
train_latent_dataset = PreEncodedLatentDataset(train_latent_data_list, train_raw_expressions_tensor)
validation_latent_dataset = PreEncodedLatentDataset(validation_latent_data_list, validation_raw_expressions_tensor)

train_latent_dataloader = DataLoader(train_latent_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
validation_latent_dataloader = DataLoader(validation_latent_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

ACTUAL_LATENT_VECTOR_DIM = train_latent_dataset.latent_vector_dim
ACTUAL_EXPRESSION_DIM = train_latent_dataset.expression_dim
print(f"\n实际用于训练的潜在向量维度: {ACTUAL_LATENT_VECTOR_DIM}")
print(f"实际用于训练的表达值维度: {ACTUAL_EXPRESSION_DIM}")


# --- 实例化模型和优化器（保持不变） ---
denoise_model = TransformerDenoiseModel(
    latent_dim=ACTUAL_LATENT_VECTOR_DIM,
    expression_dim=ACTUAL_EXPRESSION_DIM,
    model_dim=DIFFUSION_MODEL_DIM,
    time_emb_dim=TIME_EMB_DIM,
    num_layers=DIFFUSION_NUM_LAYERS,
    num_heads=8,
    dropout=0.4
).to(device)

diffusion_model = LatentDiffusionModel(
    denoise_model=denoise_model,
    timesteps=1000,
    beta_schedule='cosine',
).to(device)

optimizer = optim.AdamW(diffusion_model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=ETA_MIN_COSINE) if USE_LR_SCHEDULER else None


# --- 训练循环（核心修改：K-mer评估部分） ---
best_kmer_correlation_overall = -1.0
best_epoch_overall = -1
best_checkpoint_path_overall = None
epochs_since_last_improvement = 0

print(f"\n开始训练 Conditional Latent Vector Diffusion Model...")
print(f"验证集分割比例: {VALIDATION_SPLIT_RATIO}")
print(f"早停耐心: {EARLY_STOPPING_PATIENCE} 个评估周期 ({EARLY_STOPPING_PATIENCE * ROBUST_EVAL_FREQUENCY_EPOCHS} 个 epoch)")

for epoch in range(NUM_EPOCHS):
    current_epoch = epoch + 1
    condition_strength = get_condition_strength(current_epoch)
    print(f"Epoch [{current_epoch}/{NUM_EPOCHS}], 条件干预强度: {condition_strength:.2f}")

    # 训练阶段
    diffusion_model.train()
    total_loss = 0
    num_batches = 0

    for z_0_batch, expression_batch in tqdm(train_latent_dataloader, desc=f"Epoch {current_epoch}/{NUM_EPOCHS}"):
        z_0_batch = z_0_batch.to(device)
        expression_batch = expression_batch.to(device)

        optimizer.zero_grad()
        loss = diffusion_model(z_0_batch, expression_batch, condition_strength=condition_strength)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    print(f"Epoch [{current_epoch}/{NUM_EPOCHS}], 平均训练损失: {avg_loss:.4f}")

    if scheduler is not None:
        scheduler.step()

    # 稳健K-mer评估（使用新的K-mer计算方式）
    if USE_ROBUST_KMER_EVAL and current_epoch % ROBUST_EVAL_FREQUENCY_EPOCHS == 0 and len(validation_latent_dataset) > 0:
        diffusion_model.eval()
        print(f"\n--- 开始稳健 K-mer 评估 (Epoch {current_epoch}, k={K_FOR_ROBUST_EVALUATION}) ---")

        num_samples_to_generate = min(NUM_SAMPLES_FOR_ROBUST_EVAL, len(validation_latent_dataset))
        if num_samples_to_generate <= 0:
            print("  没有足够样本进行评估")
            continue

        # 采样表达值作为生成条件
        sampled_indices = torch.randperm(len(validation_raw_expressions_tensor))[:num_samples_to_generate]
        sampled_eval_expressions = validation_raw_expressions_tensor[sampled_indices]
        print(f"  生成 {num_samples_to_generate} 个样本进行评估...")

        # 生成潜在向量并解码为序列
        eval_batch_size = 64
        num_batches_eval = (num_samples_to_generate + eval_batch_size - 1) // eval_batch_size
        generated_sequences_for_eval = []

        with torch.no_grad():
            for i in tqdm(range(num_batches_eval), desc=f" Generating samples "):
                current_batch_size = min(eval_batch_size, num_samples_to_generate - i * eval_batch_size)
                if current_batch_size <= 0: break

                current_eval_expressions = sampled_eval_expressions[i*eval_batch_size: i*eval_batch_size+current_batch_size].to(device)
                generated_latent_vectors = diffusion_model.sample(
                    batch_size=current_batch_size,
                    expression=current_eval_expressions,
                    disable_tqdm=True
                )

                # 解码并转换为序列
                generated_output_probs = vae_model.decode(generated_latent_vectors, apply_sigmoid=True)
                for j in range(generated_output_probs.size(0)):
                    seq_str = reconstructed_to_sequence(generated_output_probs[j], SEQ_LENGTH_RAW, PADDING_RAW)
                    generated_sequences_for_eval.append(seq_str)

        # K-mer评估（使用新的频率计算和相关性函数）
        if generated_sequences_for_eval and validation_original_sequences_for_robust_eval:
            # 1. 计算K-mer相关性
            kmer_correlation = calculate_kmer_distribution_correlation(
                generated_sequences_for_eval,
                validation_original_sequences_for_robust_eval,
                k=K_FOR_ROBUST_EVALUATION,
                similarity_metric="pearson"
            )
            print(f"  K-mer 频率皮尔逊相关系数 (验证集): {kmer_correlation:.4f}")

            # 2. 生成并保存K-mer频率对比散点图
            plot_save_path = os.path.join(KMER_PLOT_DIR, f'kmer_{K_FOR_ROBUST_EVALUATION}_epoch_{current_epoch}.png')
            compare_kmer_frequencies(
                gen_sequences=generated_sequences_for_eval,
                natural_sequences=validation_original_sequences_for_robust_eval,
                k=K_FOR_ROBUST_EVALUATION,
                save_path=plot_save_path
            )
            print(f"  K-mer 对比图已保存至: {plot_save_path}")

            # 3. 早停判断
            if kmer_correlation > best_kmer_correlation_overall:
                best_kmer_correlation_overall = kmer_correlation
                best_epoch_overall = current_epoch
                checkpoint_name = f'best_model_epoch_{current_epoch}.pth'
                best_checkpoint_path_overall = os.path.join(CHECKPOINT_DIR, checkpoint_name)

                # 保存最佳模型
                try:
                    checkpoint_data = {
                        'model_state_dict': diffusion_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                        'epoch': current_epoch,
                        'best_kmer_correlation_val': kmer_correlation
                    }
                    torch.save(checkpoint_data, best_checkpoint_path_overall)
                    print(f"  >>> 新最佳！保存至 {best_checkpoint_path_overall}")
                    epochs_since_last_improvement = 0
                except Exception as e:
                    print(f"  警告: 保存检查点失败: {e}")
            else:
                epochs_since_last_improvement += 1

        else:
            print(f"  生成序列或自然序列为空，无法完成评估")

        # 检查早停
        if epochs_since_last_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"\n早停触发: 验证集 K-mer 相关性在 {EARLY_STOPPING_PATIENCE} 个评估周期内无提升。")
            print(f"最佳模型: Epoch {best_epoch_overall}, K-mer 相关系数: {best_kmer_correlation_overall:.4f}")
            print(f"最佳模型路径: {best_checkpoint_path_overall}")
            break


# 训练结束总结
print("\n训练结束。")
if best_checkpoint_path_overall:
    print(f"最佳模型保存于: {best_checkpoint_path_overall}")
    print(f"最佳 Epoch: {best_epoch_overall}, 最佳 K-mer 频率相关系数: {best_kmer_correlation_overall:.4f}")