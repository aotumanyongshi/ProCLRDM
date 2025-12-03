# File: models/diffusion_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm # Import tqdm for the sampling loop

# --- Positional Encoding ---
# Required for the Transformer DenoiseModel
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:, :d_model // 2] if d_model % 2 != 0 else position * div_term) # Corrected for odd d_model
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim] (batch_first=True)
        """
        seq_len = x.size(1)
        # Make PE match batch_first: [1, max_len, D]
        pe_batch_first = self.pe[:seq_len, :].transpose(0, 1) # Shape [1, seq_len, D]
        x = x + pe_batch_first # Add PE to each element in the sequence
        return self.dropout(x)


# --- 时间步嵌入 ---
# (与 VAE 中的 PositionalEncoding 类似，但用于时间步 t)
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# --- 去噪模型 (核心网络) ---
# 这里使用一个基于 Transformer 的简化结构作为示例
# 实践中常用 U-Net 结构，尤其对于图像潜在空间
# 但对于序列潜在空间，Transformer 也是合理的选择
class DenoiseModel(nn.Module):
    def __init__(self, latent_dim, seq_len, model_dim=256, nhead=8, num_layers=6, time_emb_dim=128, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim # VAE 的 D_latent
        self.seq_len = seq_len       # VAE 的 L
        self.model_dim = model_dim   # Transformer 内部维度

        # 时间嵌入 MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, model_dim * 2), # 扩大维度
            nn.SiLU(), # Swish 激活函数
            nn.Linear(model_dim * 2, model_dim)
        )

        # 输入映射: 将 [B, L, D_latent] 映射到 [B, L, model_dim]
        self.input_proj = nn.Linear(latent_dim, model_dim)

        # 位置编码 (可选但推荐)
        self.pos_encoder = PositionalEncoding(model_dim, dropout, max_len=seq_len + 1)

        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True # 确保输入形状为 [B, L, D]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 输出映射: 将 [B, L, model_dim] 映射回 [B, L, D_latent] 以预测噪声
        self.output_proj = nn.Linear(model_dim, latent_dim)

    def forward(self, x, t):
        """
        Args:
            x: Noisy latent variable [B, L, D_latent]
            t: Timestep tensor [B]
        Returns:
            Predicted noise [B, L, D_latent]
        """
        # 1. 时间嵌入
        time_embedding = self.time_mlp(t) # [B, model_dim]

        # 2. 输入映射
        x = self.input_proj(x) # [B, L, model_dim]

        # 3. 加入位置编码
        x = self.pos_encoder(x) # [B, L, model_dim]


        # 4. 将时间嵌入融入序列 (常见做法是加到每个位置上)
        # 扩展时间嵌入: [B, model_dim] -> [B, 1, model_dim] -> [B, L, model_dim]
        time_embedding_expanded = time_embedding.unsqueeze(1).repeat(1, x.size(1), 1)
        x = x + time_embedding_expanded # [B, L, model_dim]

        # 5. 通过 Transformer 处理
        transformer_output = self.transformer_encoder(x) # [B, L, model_dim]

        # 6. 输出映射，预测噪声
        predicted_noise = self.output_proj(transformer_output) # [B, L, D_latent]

        return predicted_noise

# --- 扩散模型包装器 ---
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    余弦调度，来自 https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def extract(a, t, x_shape):
    """从 a 中提取对应时间步 t 的值，并 reshape 成 x_shape 的形状"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class LatentDiffusionModel(nn.Module):
    def __init__(self, denoise_model, timesteps=1000, beta_schedule='cosine', latent_shape=None):
        """
        Args:
            denoise_model: The network used to predict noise (instance of DenoiseModel).
            timesteps: Number of diffusion steps (T).
            beta_schedule: 'linear' or 'cosine'.
            latent_shape: The shape of the latent variable z, e.g., (L, D_latent). Needed if not inferred.
        """
        super().__init__()
        self.denoise_model = denoise_model
        self.timesteps = timesteps
        self.latent_shape = latent_shape # Example: (seq_len, latent_dim)

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) # alpha_cumprod_{t-1}
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas) # 1 / sqrt(alpha_t)

        # 计算 q(x_t | x_0) 所需常量
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # 计算 p(x_{t-1} | x_t, x_0) 所需常量 (用于采样)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # Clamp variance to avoid potential issues (e.g., log(0))
        posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
        # p(x_{t-1}|x_t, x_0) 的均值系数
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod) # coefficient for x_0 term
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod) # coefficient for x_t term

        # 注册为 buffer，这样它们会被移动到正确的设备并保存在 state_dict 中
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.register_buffer('sqrt_recip_alphas', sqrt_recip_alphas)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', posterior_log_variance_clipped)
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1)
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2)

    # --- 前向过程 q(z_t | z_0) ---
    def q_sample(self, z_0, t, noise=None):
        """
        从 z_0 (clean latent) 采样 z_t (noisy latent)
        z_t = sqrt(alpha_cumprod_t) * z_0 + sqrt(1 - alpha_cumprod_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(z_0)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, z_0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, z_0.shape)

        return sqrt_alphas_cumprod_t * z_0 + sqrt_one_minus_alphas_cumprod_t * noise

    # --- 预测 z_0 ---
    def predict_z0_from_noise(self, z_t, t, noise):
        """
        根据 z_t 和预测的噪声 epsilon，估计原始的 z_0
        z_0 = (z_t - sqrt(1 - alpha_cumprod_t) * noise) / sqrt(alpha_cumprod_t)
        """
        sqrt_recip_alphas_cumprod_t = extract(1. / self.sqrt_alphas_cumprod, t, z_t.shape)
        # sqrt( (1-alpha_cumprod)/alpha_cumprod )
        sqrt_recipm1_alphas_cumprod_t = extract(torch.sqrt(1. / self.alphas_cumprod - 1), t, z_t.shape)

        # 使用另一种等价形式避免除以 sqrt_alphas_cumprod 可能接近0的值
        # z_0 = sqrt(1/alpha_cumprod_t)*z_t - sqrt(1/alpha_cumprod_t - 1)*noise
        pred_z0 = sqrt_recip_alphas_cumprod_t * z_t - sqrt_recipm1_alphas_cumprod_t * noise
        return pred_z0


    # --- 后验分布 q(z_{t-1} | z_t, z_0) 的均值和方差 ---
    def q_posterior_mean_variance(self, z_0, z_t, t):
        """
        计算后验分布 q(z_{t-1} | z_t, z_0) 的均值和对数方差
        """
        assert z_0.shape == z_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, z_t.shape) * z_0 +
            extract(self.posterior_mean_coef2, t, z_t.shape) * z_t
        )
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, z_t.shape)
        return posterior_mean, posterior_log_variance

    # --- 逆向过程 p(z_{t-1} | z_t) ---
    @torch.no_grad()
    def p_sample(self, z_t, t, clip_denoised=True):
        """
        从 z_t 采样 z_{t-1}
        """
        # 1. 使用模型预测噪声
        betas_t = extract(self.betas, t, z_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, z_t.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, z_t.shape)

        # 使用模型预测噪声 epsilon
        predicted_noise = self.denoise_model(z_t, t)

        # Optional: 预测 z_0 并进行 clip
        # pred_z0 = self.predict_z0_from_noise(z_t, t, predicted_noise)
        # if clip_denoised:
        #    pred_z0.clamp_(-1., 1.) # 如果你的潜在空间被归一化到 [-1, 1]

        # 重新计算噪声（如果 pred_z0 被 clip 了）
        # model_mean 不使用 clip 的 z_0
        # 使用 DDPM 论文中的公式 (基于预测的噪声)
        # mean = (1/sqrt(alpha_t)) * (z_t - (beta_t / sqrt(1-alpha_cumprod_t)) * predicted_noise)
        model_mean = sqrt_recip_alphas_t * (z_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)


        # --- 或者，使用 DDIM 的方式 ---
        # # 1. 预测噪声
        # predicted_noise = self.denoise_model(z_t, t)
        # # 2. 预测 z_0
        # pred_z0 = self.predict_z0_from_noise(z_t, t, predicted_noise)
        # # Optional clip
        # if clip_denoised:
        #    pred_z0.clamp_(-1., 1.) # Or other range
        # # 3. 计算均值 (eta=0 for DDIM)
        # alpha_cumprod_prev_t = extract(self.alphas_cumprod_prev, t, z_t.shape)
        # direction_point_to_xt = torch.sqrt(1. - alpha_cumprod_prev_t) * predicted_noise
        # model_mean = torch.sqrt(alpha_cumprod_prev_t) * pred_z0 + direction_point_to_xt


        # --- 计算方差 ---
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, z_t.shape)

        # 采样 z_{t-1}
        noise = torch.randn_like(z_t) if t[0] > 0 else torch.zeros_like(z_t) # no noise at t=0
        z_prev = model_mean + (0.5 * posterior_log_variance).exp() * noise
        return z_prev

    @torch.no_grad()
    def p_sample_loop(self, shape, device):
        """
        完整的采样循环，从 T 到 0
        """
        b = shape[0] # Batch size
        # 从纯噪声 z_T 开始
        z = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.timesteps)), desc='Sampling loop time step', total=self.timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            z = self.p_sample(z, t)

        return z # 返回生成的 z_0

    @torch.no_grad()
    def sample(self, batch_size=1):
        """
        生成指定数量的潜在样本 z_0
        """
        if self.latent_shape is None:
             # 尝试从 denoise_model 推断（如果它存储了）
             # 或者需要用户在初始化时提供
             if hasattr(self.denoise_model, 'seq_len') and hasattr(self.denoise_model, 'latent_dim'):
                 l, d = self.denoise_model.seq_len, self.denoise_model.latent_dim
                 self.latent_shape = (l, d)
             else:
                  raise ValueError("latent_shape must be provided during initialization or inferrable from denoise_model")

        latent_device = next(self.denoise_model.parameters()).device
        sample_shape = (batch_size, *self.latent_shape)
        return self.p_sample_loop(sample_shape, latent_device)

    # --- 训练损失 ---
    def p_losses(self, z_0, t, noise=None):
        """
        计算扩散模型的训练损失 (预测噪声 vs 真实噪声)
        Args:
            z_0: Clean latent variable from VAE encoder [B, L, D_latent]
            t: Sampled timesteps [B]
            noise: Optional pre-sampled noise [B, L, D_latent]
        Returns:
            MSE loss
        """
        if noise is None:
            noise = torch.randn_like(z_0)

        # 1. 计算带噪声的 z_t
        z_t = self.q_sample(z_0, t, noise=noise)

        # 2. 使用模型预测噪声
        predicted_noise = self.denoise_model(z_t, t)

        # 3. 计算损失 (MSE)
        loss = F.mse_loss(predicted_noise, noise) # 比较预测噪声和真实噪声
        return loss

    def forward(self, z_0):
        """
        训练步骤: 计算给定一批 z_0 的损失
        """
        B, L, D = z_0.shape
        # 1. 随机采样时间步 t
        t = torch.randint(0, self.timesteps, (B,), device=z_0.device).long()

        # 2. 计算损失
        return self.p_losses(z_0, t)