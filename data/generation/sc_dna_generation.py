import torch
import os
from tqdm import tqdm
from data.models.sc_vae_model import VAE
from data.models.sc_diffusion_model import LatentDiffusionModel, TransformerDenoiseModel

# --- 全局参数（与训练脚本保持一致） ---
VAE_LATENT_DIM = 256
DIFFUSION_MODEL_DIM = 1024
DIFFUSION_NUM_LAYERS = 6
TIME_EMB_DIM = 128
TIMESTEPS = 1000
BETA_SCHEDULE = 'cosine'
EXPRESSION_DIM = 1

SEQ_LENGTH_RAW = 80
PADDING_RAW = 0
SEQ_LENGTH_PADDED = SEQ_LENGTH_RAW + 2 * PADDING_RAW

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")

INDEX_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}


# --- 序列转换函数 ---
def reconstructed_to_sequence(reconstructed_tensor, original_seq_len, padding):
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
                print(f"ERROR: 维度处理后仍异常: {reconstructed_tensor.shape}")
                return "UNEXPECTED_SHAPE"
        else:
            print(f"ERROR: 张量维度异常: {reconstructed_tensor.shape}")
            return "UNEXPECTED_SHAPE"

    seq_len_with_padding = reconstructed_tensor.shape[-1]
    start_index = padding
    end_index = padding + original_seq_len

    start_index = max(0, start_index)
    end_index = min(seq_len_with_padding, end_index)

    indices = torch.argmax(reconstructed_tensor, dim=0)
    core_indices = indices[start_index:end_index]
    return "".join([INDEX_TO_BASE.get(i.item(), 'N') for i in core_indices])


# --- 加载模型 ---
def load_trained_models(vae_path, diffusion_ckpt_path):
    # 加载VAE
    vae_model = VAE(latent_dim=VAE_LATENT_DIM).to(device)
    vae_state_dict = torch.load(vae_path, map_location=device)
    new_vae_state_dict = {k.replace("vae.", ""): v for k, v in vae_state_dict.items()}
    vae_model.load_state_dict(new_vae_state_dict, strict=True)
    vae_model.eval()
    for param in vae_model.parameters():
        param.requires_grad = False

    # 加载扩散模型
    denoise_model = TransformerDenoiseModel(
        latent_dim=VAE_LATENT_DIM,
        expression_dim=EXPRESSION_DIM,
        model_dim=DIFFUSION_MODEL_DIM,
        time_emb_dim=TIME_EMB_DIM,
        num_layers=DIFFUSION_NUM_LAYERS,
        num_heads=8,
        dropout=0.4
    ).to(device)
    diffusion_model = LatentDiffusionModel(
        denoise_model=denoise_model,
        timesteps=TIMESTEPS,
        beta_schedule=BETA_SCHEDULE,
    ).to(device)

    ckpt = torch.load(diffusion_ckpt_path, map_location=device,weights_only=False)
    diffusion_model.load_state_dict(ckpt['model_state_dict'])
    diffusion_model.eval()
    return vae_model, diffusion_model


# --- 批量生成DNA序列 ---
def generate_dna_sequences(
        diffusion_model,
        vae_model,
        num_samples,
        expression_values,
        device,
        batch_size=32
):
    generated_sequences = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="生成序列"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            batch_expressions = expression_values[start_idx:end_idx].to(device)

            generated_latents = diffusion_model.sample(
                batch_size=batch_expressions.shape[0],
                expression=batch_expressions,
            )

            reconstructed_probs = vae_model.decode(generated_latents, apply_sigmoid=True)

            for j in range(reconstructed_probs.size(0)):
                seq = reconstructed_to_sequence(
                    reconstructed_probs[j],
                    original_seq_len=SEQ_LENGTH_RAW,
                    padding=PADDING_RAW
                )
                generated_sequences.append(seq)

    return generated_sequences


# --- 主流程 ---
if __name__ == "__main__":
    # 配置路径
    VAE_MODEL_PATH = '/root/autodl-tmp/sjy/dachang/data/my_vae_model_state_dict.pth'
    DIFFUSION_CKPT_PATH = '/root/autodl-tmp/sjy/dachang/data/training/conditional_diffusion_checkpoints_latent_vector_256dim_cfg_tuned_early_stopping/best_model_epoch_50.pth'

    # 加载模型
    vae_model, diffusion_model = load_trained_models(VAE_MODEL_PATH, DIFFUSION_CKPT_PATH)

    # 配置表达值范围
    num_samples = 10000  # 生成序列数量
    min_expression = 10.0
    max_expression = 16.0

    # 方法1：均匀分布
    random_expressions = min_expression + (max_expression - min_expression) * torch.rand(num_samples, 1)
    print( random_expressions)
    # 或方法2：截断正态分布
    # mean = (min_expression + max_expression) / 2
    # std = 1.0
    # random_expressions = mean + std * torch.randn(num_samples, 1)
    # random_expressions = torch.clamp(random_expressions, min_expression, max_expression)

    print(f"生成的表达值范围: {random_expressions.min().item():.4f} 到 {random_expressions.max().item():.4f}")

    # 生成序列
    generated_seqs = generate_dna_sequences(
        diffusion_model,
        vae_model,
        num_samples=num_samples,
        expression_values=random_expressions,
        device=device
    )

    # 保存结果
    output_dir = "generated_sequences"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "generated_dna_seqs.txt"), "w") as f:
        for i, seq in enumerate(generated_seqs):
            f.write(f"Sequence {i + 1}: {seq}\n")
    print(f"成功生成 {len(generated_seqs)} 条序列，保存至 {output_dir}/generated_dna_seqs.txt")