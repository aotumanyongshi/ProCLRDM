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
from data.models.non_con_diffusion_model import LatentDiffusionModel, DenoiseModel  # 使用新的扩散模型
from data.utils.utils import SequenceDataset  # 仅使用序列数据集

# --- 设置设备 ---
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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


# --- 超参数设置 ---
# VAE 参数
VAE_LATENT_DIM = 256
VAE_MODEL_PATH = '/root/autodl-tmp/sjy/dachang1/data/my_vae_model_state_dict.pth'

# 数据路径
DATA_PATH = '/root/autodl-tmp/sjy/dachang1/data/raw_sequences/sc_data/top_10000_sequences.txt'

# 序列参数
SEQ_LENGTH_RAW = 80
PADDING_RAW = 0
SEQ_LENGTH_PADDED = SEQ_LENGTH_RAW + 2 * PADDING_RAW
LATENT_SEQ_LEN = 1  # 潜在序列长度

# Diffusion Model 参数
DIFFUSION_MODEL_DIM = 1024
DIFFUSION_NUM_LAYERS = 6
TIME_EMB_DIM = 128
TIMESTEPS = 1000
BETA_SCHEDULE = 'cosine'

# 训练参数
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 200

# 学习率调度器
USE_LR_SCHEDULER = True
T_MAX_COSINE = NUM_EPOCHS
ETA_MIN_COSINE = 6e-7

# 检查点目录
CHECKPOINT_DIR = 'unconditional_diffusion_checkpoints_latent_vector_256dim'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 其他参数
VALIDATION_SPLIT_RATIO = 0.15


# --- 加载VAE模型 ---
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


# --- 数据集加载和分割 ---
print("\n正在加载序列数据集...")
full_sequence_dataset = SequenceDataset(DATA_PATH, seq_length=SEQ_LENGTH_RAW, padding=PADDING_RAW)

if len(full_sequence_dataset) == 0:
    print("\n错误：加载的序列数据集为空。")
    exit()

# 分割训练集和验证集
full_data_size = len(full_sequence_dataset)
validation_size = int(full_data_size * VALIDATION_SPLIT_RATIO)
train_size = full_data_size - validation_size
train_subset, validation_subset = random_split(full_sequence_dataset, [train_size, validation_size])
print(f"数据集已分割: 训练集 {len(train_subset)} 条, 验证集 {len(validation_subset)} 条。")


# --- VAE编码潜在空间 ---
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
                mu_batch = mean  # 取均值作为潜在向量
                # 调整形状为 [B, L, D] 以匹配扩散模型输入 (L=1)
                mu_batch_reshaped = mu_batch.view(mu_batch.shape[0], LATENT_SEQ_LEN, -1)
                latent_data.extend([mu.cpu() for mu in mu_batch_reshaped])
            except Exception as e:
                print(f"VAE 编码批次时发生错误: {e}")
                continue

    return latent_data


# 编码训练集和验证集数据
print("正在使用 VAE 编码器将训练集原始数据编码到潜在空间...")
train_raw_onehot = [train_subset[i] for i in range(len(train_subset))]
train_latent_data_list = encode_sequences_to_latent(train_raw_onehot, vae_model, device, SEQ_LENGTH_PADDED)

print("正在使用 VAE 编码器将验证集原始数据编码到潜在空间...")
validation_raw_onehot = [validation_subset[i] for i in range(len(validation_subset))]
validation_latent_data_list = encode_sequences_to_latent(validation_raw_onehot, vae_model, device, SEQ_LENGTH_PADDED)

# 检查编码后的数据长度
assert len(train_latent_data_list) == len(train_subset), "训练集潜在向量数量不匹配"
assert len(validation_latent_data_list) == len(validation_subset), "验证集潜在向量数量不匹配"


# --- 潜在空间数据集 ---
class PreEncodedLatentDataset(Dataset):
    def __init__(self, latent_data_list):
        self.latent_data = latent_data_list
        if len(self.latent_data) > 0:
            self.latent_shape = self.latent_data[0].shape  # [L, D]
            self.latent_dim = self.latent_shape[-1]
            self.seq_len = self.latent_shape[0]
        else:
            self.latent_dim = VAE_LATENT_DIM
            self.seq_len = LATENT_SEQ_LEN

    def __len__(self):
        return len(self.latent_data)

    def __getitem__(self, idx):
        return self.latent_data[idx]  # 仅返回潜在向量


# 实例化数据集和DataLoader
train_latent_dataset = PreEncodedLatentDataset(train_latent_data_list)
validation_latent_dataset = PreEncodedLatentDataset(validation_latent_data_list)

train_latent_dataloader = DataLoader(train_latent_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
validation_latent_dataloader = DataLoader(validation_latent_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"\n实际潜在向量形状: {train_latent_dataset.latent_shape} (L={train_latent_dataset.seq_len}, D={train_latent_dataset.latent_dim})")


# --- 实例化模型和优化器 ---
denoise_model = DenoiseModel(
    latent_dim=train_latent_dataset.latent_dim,
    seq_len=train_latent_dataset.seq_len,
    model_dim=DIFFUSION_MODEL_DIM,
    nhead=8,
    num_layers=DIFFUSION_NUM_LAYERS,
    time_emb_dim=TIME_EMB_DIM,
    dropout=0.4
).to(device)

diffusion_model = LatentDiffusionModel(
    denoise_model=denoise_model,
    timesteps=TIMESTEPS,
    beta_schedule=BETA_SCHEDULE,
    latent_shape=(train_latent_dataset.seq_len, train_latent_dataset.latent_dim)
).to(device)

optimizer = optim.AdamW(diffusion_model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=ETA_MIN_COSINE) if USE_LR_SCHEDULER else None


# --- 训练循环（仅保留训练逻辑） ---
print(f"\n开始训练 Unconditional Latent Diffusion Model...")

# 保存每个epoch的损失
train_losses = []
val_losses = []

for epoch in range(NUM_EPOCHS):
    current_epoch = epoch + 1
    print(f"Epoch [{current_epoch}/{NUM_EPOCHS}]")

    # 训练阶段
    diffusion_model.train()
    total_train_loss = 0
    num_train_batches = 0

    for z_0_batch in tqdm(train_latent_dataloader, desc=f"训练中"):
        z_0_batch = z_0_batch.to(device)

        optimizer.zero_grad()
        loss = diffusion_model(z_0_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), max_norm=1.0)
        optimizer.step()

        total_train_loss += loss.item()
        num_train_batches += 1

    avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0.0
    train_losses.append(avg_train_loss)
    print(f"Epoch [{current_epoch}/{NUM_EPOCHS}], 平均训练损失: {avg_train_loss:.4f}")

    # 验证阶段（仅计算损失，不做K-mer评估）
    diffusion_model.eval()
    total_val_loss = 0
    num_val_batches = 0

    with torch.no_grad():
        for z_0_batch in tqdm(validation_latent_dataloader, desc=f"验证中"):
            z_0_batch = z_0_batch.to(device)
            loss = diffusion_model(z_0_batch)
            total_val_loss += loss.item()
            num_val_batches += 1

    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
    val_losses.append(avg_val_loss)
    print(f"Epoch [{current_epoch}/{NUM_EPOCHS}], 平均验证损失: {avg_val_loss:.4f}")

    # 学习率调度
    if scheduler is not None:
        scheduler.step()

    # 每20个epoch保存一次模型
    if current_epoch % 20 == 0:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{current_epoch}.pth")
        torch.save({
            'epoch': current_epoch,
            'model_state_dict': diffusion_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }, checkpoint_path)
        print(f"模型已保存至: {checkpoint_path}")


# 训练结束后保存最终模型和损失曲线
final_model_path = os.path.join(CHECKPOINT_DIR, "final_model.pth")
torch.save({
    'model_state_dict': diffusion_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
    'train_losses': train_losses,
    'val_losses': val_losses
}, final_model_path)
print(f"\n训练结束，最终模型保存至: {final_model_path}")

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, NUM_EPOCHS+1), train_losses, label='训练损失')
plt.plot(range(1, NUM_EPOCHS+1), val_losses, label='验证损失')
plt.xlabel('Epoch')
plt.ylabel('损失值')
plt.title('训练与验证损失曲线')
plt.legend()
plt.savefig(os.path.join(CHECKPOINT_DIR, 'loss_curve.png'), dpi=300)
plt.close()
print(f"损失曲线已保存至: {os.path.join(CHECKPOINT_DIR, 'loss_curve.png')}")