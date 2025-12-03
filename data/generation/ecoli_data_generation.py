import os
import torch
import numpy as np
from tqdm import tqdm

# --- 导入模型 ---
from data.models.sc_vae_model import VAE
from data.models.sc_diffusion_model import LatentDiffusionModel, TransformerDenoiseModel

# --- 设备配置 ---
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"使用 GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# --- VAE 参数 ---
VAE_MODEL_PATH = "/root/autodl-tmp/sjy/dachang/data/my_vae_model_state_dict.pth"
VAE_LATENT_DIM = 256

# --- Diffusion 模型检查点 ---
DIFFUSION_CHECKPOINT_FILE = "/root/autodl-tmp/sjy/dachang1/data/training/conditional_diffusion_checkpoints_latent_vector_256dim_cfg_tuned_early_stopping/best_model_epoch_500.pth"

# --- 序列参数 ---
SEQ_LENGTH_RAW = 165
PADDING_RAW = 10
SEQ_LENGTH_PADDED = SEQ_LENGTH_RAW + 2 * PADDING_RAW

# --- 生成参数 ---
NUM_SEQUENCES_TO_GENERATE = 10000
GENERATION_BATCH_SIZE = 64
MIN_EXPRESSION = 4
MAX_EXPRESSION = 8

# 生成随机目标表达值
TARGET_RAW_EXPRESSION_VALUES = np.random.uniform(
    low=MIN_EXPRESSION,
    high=MAX_EXPRESSION,
    size=NUM_SEQUENCES_TO_GENERATE
).tolist()

# --- 碱基映射 ---
INDEX_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}


# --- 辅助函数：重构概率张量转DNA序列 ---
def reconstructed_to_sequence(reconstructed_tensor, original_seq_len, padding):
    if reconstructed_tensor.device.type != 'cpu':
        reconstructed_tensor = reconstructed_tensor.cpu()

    if reconstructed_tensor.ndim == 3:
        if reconstructed_tensor.shape[0] == 1:
            reconstructed_tensor = reconstructed_tensor.squeeze(0)
        elif reconstructed_tensor.shape[1] == 1:
            reconstructed_tensor = reconstructed_tensor.squeeze(1)

    if reconstructed_tensor.ndim != 2 or reconstructed_tensor.shape[0] != 4:
        print(f"错误：张量形状异常 {reconstructed_tensor.shape}，预期(4, seq_len)")
        return "INVALID_SHAPE"

    seq_len_with_padding = reconstructed_tensor.shape[1]
    start_idx = padding
    end_idx = padding + original_seq_len
    start_idx = max(0, start_idx)
    end_idx = min(seq_len_with_padding, end_idx)

    if start_idx >= end_idx:
        print(f"警告：有效序列长度为0（填充/序列长度不匹配）")
        return "EMPTY_SEQUENCE"

    base_indices = torch.argmax(reconstructed_tensor, dim=0)
    core_indices = base_indices[start_idx:end_idx]
    return "".join([INDEX_TO_BASE.get(idx.item(), 'N') for idx in core_indices])


# --- 加载VAE模型 ---
print(f"\n=== 加载VAE模型 ===")
vae_model = VAE(latent_dim=VAE_LATENT_DIM).to(device)
try:
    vae_state_dict = torch.load(VAE_MODEL_PATH, map_location=device)
    if any(key.startswith("vae.") for key in vae_state_dict.keys()):
        vae_state_dict = {k[len("vae."):]: v for k, v in vae_state_dict.items()}
    vae_model.load_state_dict(vae_state_dict)
    vae_model.eval()
    print(f"VAE模型加载成功（路径：{VAE_MODEL_PATH}）")
except Exception as e:
    print(f"VAE加载失败：{str(e)}")
    exit()

# --- 加载Diffusion模型 ---
print(f"\n=== 加载Diffusion模型 ===")
try:
    checkpoint = torch.load(DIFFUSION_CHECKPOINT_FILE, map_location=device, weights_only=False)

    denoise_model = TransformerDenoiseModel(
        latent_dim=VAE_LATENT_DIM,
        expression_dim=1,
        model_dim=1024,
        time_emb_dim=128,
        num_layers=6,
        num_heads=8,
        dropout=0.4
    ).to(device)

    diffusion_model = LatentDiffusionModel(
        denoise_model=denoise_model,
        timesteps=1000,
        beta_schedule='cosine'
    ).to(device)

    diffusion_model.load_state_dict(checkpoint['model_state_dict'])
    diffusion_model.eval()
    print(f"Diffusion模型加载成功（路径：{DIFFUSION_CHECKPOINT_FILE}）")
    print(f"最佳模型训练 epoch：{checkpoint.get('epoch', '未知')}")

except FileNotFoundError:
    print(f"错误：Diffusion检查点文件不存在（路径：{DIFFUSION_CHECKPOINT_FILE}）")
    exit()
except Exception as e:
    print(f"Diffusion加载失败：{str(e)}")
    exit()

# --- 生成DNA序列（主逻辑）---
print(f"\n=== 开始生成DNA序列 ===")
print(f"生成数量：{NUM_SEQUENCES_TO_GENERATE}条")
print(f"表达值范围：[{MIN_EXPRESSION}, {MAX_EXPRESSION}]")
print(f"批次大小：{GENERATION_BATCH_SIZE}")

# 转换目标表达值为张量
condition_expressions = torch.tensor(
    [[val] for val in TARGET_RAW_EXPRESSION_VALUES],
    dtype=torch.float32
).to(device)

# 移除引导强度循环，直接生成（核心修改）
generated_sequences = []
with torch.no_grad():
    for i in tqdm(range(0, NUM_SEQUENCES_TO_GENERATE, GENERATION_BATCH_SIZE), desc="生成进度"):
        batch_conditions = condition_expressions[i:i + GENERATION_BATCH_SIZE]
        batch_size = batch_conditions.shape[0]

        # 生成潜在向量（删除 guidance_scale 参数）
        generated_latents = diffusion_model.sample(
            batch_size=batch_size,
            expression=batch_conditions,  # 仅传入表达值条件
            disable_tqdm=True
        )

        # VAE解码
        decoded_probs = vae_model.decode(generated_latents, apply_sigmoid=True)

        # 转换为序列
        for j in range(batch_size):
            dna_seq = reconstructed_to_sequence(
                reconstructed_tensor=decoded_probs[j],
                original_seq_len=SEQ_LENGTH_RAW,
                padding=PADDING_RAW
            )
            generated_sequences.append(dna_seq)

# 保存生成结果
output_filename = "generated_dna_sequences.csv"
try:
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("expression_value,dna_sequence\n")
        for expr, seq in zip(TARGET_RAW_EXPRESSION_VALUES, generated_sequences):
            f.write(f"{expr:.4f},{seq}\n")
    print(f"生成完成！结果已保存至：{output_filename}")
except Exception as e:
    print(f"保存文件失败：{str(e)}")

print("\n=== 所有生成任务完成 ===")