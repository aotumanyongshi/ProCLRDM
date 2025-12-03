import os
import torch
from tqdm import tqdm

# --- 导入模型 ---
from data.models.sc_vae_model import VAE
from data.models.non_con_diffusion_model import LatentDiffusionModel, DenoiseModel  # 使用无条件扩散模型

# --- 设备配置 ---
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"使用 GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# --- VAE 参数 ---
VAE_MODEL_PATH = "/root/autodl-tmp/sjy/dachang1/data/my_vae_model_state_dict.pth"  # 与训练时一致
VAE_LATENT_DIM = 256  # 与训练时一致

# --- 无条件Diffusion模型检查点 ---
DIFFUSION_CHECKPOINT_FILE = "/root/autodl-tmp/sjy/dachang1/data/training/unconditional_diffusion_checkpoints_latent_vector_256dim/final_model.pth"  # 修改为您的检查点路径

# --- 序列参数 ---
SEQ_LENGTH_RAW = 80  # 与训练时一致
PADDING_RAW = 0  # 与训练时一致
SEQ_LENGTH_PADDED = SEQ_LENGTH_RAW + 2 * PADDING_RAW  # 与训练时一致

# --- 生成参数 ---
NUM_SEQUENCES_TO_GENERATE = 10000  # 生成数量
GENERATION_BATCH_SIZE = 64  # 批次大小，可根据GPU内存调整

# --- 碱基映射 ---
INDEX_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}


# --- 辅助函数：重构概率张量转DNA序列 ---
def reconstructed_to_sequence(reconstructed_tensor, original_seq_len, padding):
    """将VAE解码后的张量转换为DNA序列"""
    if reconstructed_tensor.device.type != 'cpu':
        reconstructed_tensor = reconstructed_tensor.cpu()

    # 处理不同可能的张量形状
    if reconstructed_tensor.ndim == 3:
        if reconstructed_tensor.shape[0] == 1:
            reconstructed_tensor = reconstructed_tensor.squeeze(0)
        elif reconstructed_tensor.shape[1] == 1:
            reconstructed_tensor = reconstructed_tensor.squeeze(1)

    # 检查是否为预期形状 (4, seq_len)
    if reconstructed_tensor.ndim != 2 or reconstructed_tensor.shape[0] != 4:
        print(f"错误：张量形状异常 {reconstructed_tensor.shape}，预期(4, seq_len)")
        return "INVALID_SHAPE"

    # 提取有效序列部分（去除padding）
    seq_len_with_padding = reconstructed_tensor.shape[1]
    start_idx = padding
    end_idx = padding + original_seq_len
    start_idx = max(0, start_idx)
    end_idx = min(seq_len_with_padding, end_idx)

    if start_idx >= end_idx:
        print(f"警告：有效序列长度为0（填充/序列长度不匹配）")
        return "EMPTY_SEQUENCE"

    # 转换为碱基序列
    base_indices = torch.argmax(reconstructed_tensor, dim=0)
    core_indices = base_indices[start_idx:end_idx]
    return "".join([INDEX_TO_BASE.get(idx.item(), 'N') for idx in core_indices])


# --- 加载VAE模型 ---
print(f"\n=== 加载VAE模型 ===")
vae_model = VAE(latent_dim=VAE_LATENT_DIM).to(device)
try:
    vae_state_dict = torch.load(VAE_MODEL_PATH, map_location=device)
    # 处理可能的前缀问题
    if any(key.startswith("vae.") for key in vae_state_dict.keys()):
        vae_state_dict = {k[len("vae."):]: v for k, v in vae_state_dict.items()}
    vae_model.load_state_dict(vae_state_dict)
    vae_model.eval()
    print(f"VAE模型加载成功（路径：{VAE_MODEL_PATH}）")
except Exception as e:
    print(f"VAE加载失败：{str(e)}")
    exit()

# --- 加载无条件Diffusion模型 ---
print(f"\n=== 加载无条件Diffusion模型 ===")
try:
    checkpoint = torch.load(DIFFUSION_CHECKPOINT_FILE, map_location=device, weights_only=False)

    # 实例化去噪模型（与训练时参数一致）
    denoise_model = DenoiseModel(
        latent_dim=VAE_LATENT_DIM,
        seq_len=1,  # 与训练时LATENT_SEQ_LEN一致
        model_dim=1024,  # 与训练时DIFFUSION_MODEL_DIM一致
        nhead=8,
        num_layers=6,  # 与训练时DIFFUSION_NUM_LAYERS一致
        time_emb_dim=128,  # 与训练时TIME_EMB_DIM一致
        dropout=0.4
    ).to(device)

    # 实例化扩散模型
    diffusion_model = LatentDiffusionModel(
        denoise_model=denoise_model,
        timesteps=1000,  # 与训练时TIMESTEPS一致
        beta_schedule='cosine'  # 与训练时BETA_SCHEDULE一致
    ).to(device)

    # 加载模型权重
    diffusion_model.load_state_dict(checkpoint['model_state_dict'])
    diffusion_model.eval()
    print(f"Diffusion模型加载成功（路径：{DIFFUSION_CHECKPOINT_FILE}）")
    print(f"模型训练 epoch：{checkpoint.get('epoch', '未知')}")

except FileNotFoundError:
    print(f"错误：Diffusion检查点文件不存在（路径：{DIFFUSION_CHECKPOINT_FILE}）")
    exit()
except Exception as e:
    print(f"Diffusion加载失败：{str(e)}")
    exit()

# --- 生成DNA序列（主逻辑）---
print(f"\n=== 开始生成DNA序列 ===")
print(f"生成数量：{NUM_SEQUENCES_TO_GENERATE}条")
print(f"批次大小：{GENERATION_BATCH_SIZE}")

generated_sequences = []
with torch.no_grad():
    for i in tqdm(range(0, NUM_SEQUENCES_TO_GENERATE, GENERATION_BATCH_SIZE), desc="生成进度"):
        current_batch_size = min(GENERATION_BATCH_SIZE, NUM_SEQUENCES_TO_GENERATE - i)

        # 生成潜在向量
        generated_latents = diffusion_model.sample(
            batch_size=current_batch_size
        )

        # 关键修改：去除冗余的维度（将[B, 1, 256]调整为[B, 256]，匹配VAE解码器输入要求）
        generated_latents = generated_latents.squeeze(1)  # 挤压掉长度为1的维度

        # VAE解码（此时输入形状与解码器匹配）
        decoded_probs = vae_model.decode(generated_latents, apply_sigmoid=True)

        # 转换为DNA序列
        for j in range(current_batch_size):
            dna_seq = reconstructed_to_sequence(
                reconstructed_tensor=decoded_probs[j],
                original_seq_len=SEQ_LENGTH_RAW,
                padding=PADDING_RAW
            )
            generated_sequences.append(dna_seq)

# 保存生成结果
output_filename = "unconditional_generated_dna_sequences.txt"
try:
    with open(output_filename, "w", encoding="utf-8") as f:
        # 写入序列数量
        f.write(f"# 生成DNA序列总数: {len(generated_sequences)}\n")
        # 写入每条序列
        for idx, seq in enumerate(generated_sequences, 1):
            f.write(f"{seq}\n")
    print(f"生成完成！结果已保存至：{output_filename}")
except Exception as e:
    print(f"保存文件失败：{str(e)}")

print("\n=== 所有生成任务完成 ===")