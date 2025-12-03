import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import numpy as np


# --- DDPM 基础函数 ---
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# --- 时间步嵌入 ---
class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear_1 = nn.Linear(dim, dim * 4)
        self.linear_2 = nn.Linear(dim * 4, dim)
        self.act = nn.SiLU()

    def forward(self, t):
        # t 形状: (batch_size,)
        t = timestep_embedding(t, self.dim)  # (batch_size, dim)
        t = self.linear_1(t)  # (batch_size, dim*4)
        t = self.act(t)
        t = self.linear_2(t)  # (batch_size, dim)
        return t


def timestep_embedding(t, dim, max_period=10000):
    # 基础正弦嵌入，与OpenAI的实现一致
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# --- 基础DDPM去噪模型 (MLP) ---
class BasicDenoiseMLP(nn.Module):
    def __init__(self, seq_length, latent_dim=4, model_dim=128, time_emb_dim=128):
        super().__init__()
        self.seq_length = seq_length
        self.latent_dim = latent_dim  # DNA One-hot维度 (4)
        self.model_dim = model_dim

        # 输入层：将 [seq_length, 4] 映射到 [seq_length, model_dim]
        self.input_proj = nn.Linear(latent_dim, model_dim)

        # 时间步嵌入
        self.time_emb = TimestepEmbedding(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, model_dim)

        # MLP 主体 (简化的多层感知机)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
        )

        # 输出层：映射回 [seq_length, 4]
        self.output_proj = nn.Linear(model_dim, latent_dim)

    def forward(self, x, t):
        # x 形状: (batch_size, seq_length, 4)
        batch_size, seq_len, _ = x.shape

        # 1. 输入投影
        x = self.input_proj(x)  # (batch_size, seq_len, model_dim)

        # 2. 时间步嵌入
        time_emb = self.time_emb(t)  # (batch_size, model_dim)
        time_emb = self.time_proj(time_emb)  # (batch_size, model_dim)
        time_emb = time_emb.unsqueeze(1)  # (batch_size, 1, model_dim)
        x = x + time_emb  # 广播到序列每个位置

        # 3. MLP处理
        x = self.mlp(x)  # (batch_size, seq_len, model_dim)

        # 4. 输出投影
        noise_pred = self.output_proj(x)  # (batch_size, seq_len, 4)

        return noise_pred


# --- 基础DDPM模型 ---
class BasicDDPM(nn.Module):
    def __init__(self, denoise_model, timesteps=1000, beta_schedule='linear', latent_dim=4):
        super().__init__()
        self.denoise_model = denoise_model
        self.timesteps = timesteps
        self.latent_dim = latent_dim  # DNA One-hot维度

        # 初始化beta调度
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f"仅支持线性beta调度，当前为: {beta_schedule}")

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)  # 前一个时间步的累积alpha

        # 注册为buffer
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # 计算扩散过程中的系数
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # 后验分布参数
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))

        # 后验均值系数
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    # 前向扩散过程
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    # 去噪过程的均值和方差
    def p_mean_variance(self, x, t):
        noise_pred = self.denoise_model(x, t)
        x_0_pred = (extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * noise_pred)

        # 计算后验均值
        model_mean = (extract(self.posterior_mean_coef1, t, x.shape) * x_0_pred +
                      extract(self.posterior_mean_coef2, t, x.shape) * x)

        posterior_variance = extract(self.posterior_variance, t, x.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x.shape)

        return model_mean, posterior_variance, posterior_log_variance

    # 单步反向采样
    def p_sample(self, x, t):
        model_mean, _, model_log_variance = self.p_mean_variance(x, t)
        noise = torch.randn_like(x)
        # 最后一步不需要加噪声
        nonzero_mask = (t != 0).float().reshape(x.shape[0], *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    # 反向采样循环
    @torch.no_grad()
    def p_sample_loop(self, shape):
        batch_size = shape[0]
        device = next(self.parameters()).device

        # 从噪声开始
        x = torch.randn(shape, device=device)
        # 逐步去噪
        for t in tqdm(reversed(range(0, self.timesteps)), desc="采样进度"):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_tensor)
        return x

    # 生成样本
    @torch.no_grad()
    def sample(self, batch_size, seq_length):
        return self.p_sample_loop((batch_size, seq_length, self.latent_dim))

    # 计算训练损失
    def p_losses(self, x_start, t=None):
        batch_size = x_start.shape[0]
        if t is None:
            t = torch.randint(0, self.timesteps, (batch_size,), device=x_start.device).long()

        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        noise_pred = self.denoise_model(x_t, t)

        return F.mse_loss(noise_pred, noise)

    # 前向传播（训练入口）
    def forward(self, x_start):
        return self.p_losses(x_start)