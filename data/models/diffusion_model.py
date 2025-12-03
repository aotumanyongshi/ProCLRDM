import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import numpy as np

# --- DDPM/DDIM 相关的辅助函数 ---
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# --- 时间步嵌入 ---
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

        if emb.shape[-1] != self.dim:
             if emb.shape[-1] > self.dim:
                  emb = emb[:, :self.dim]
             elif emb.shape[-1] < self.dim:
                  padding_needed = self.dim - emb.shape[-1]
                  emb = torch.cat((emb, torch.zeros(x.size(0), padding_needed, device=device)), dim=-1)

        return emb

# --- Adaptive Layer Normalization (AdaLN) 模块 ---
class AdaLN(nn.Module):
    def __init__(self, size, condition_dim):
        super().__init__()
        self.norm = nn.LayerNorm(size, elementwise_affine=False)
        self.projection = nn.Linear(condition_dim, size * 2)

        nn.init.constant_(self.projection.weight, 0)
        if self.projection.bias is not None:
            size_ = size
            nn.init.constant_(self.projection.bias[:size_], 1.0)
            nn.init.constant_(self.projection.bias[size_:], 0.0)

    def forward(self, x, condition):
        if x.ndim == 3 and condition.ndim == 2:
             condition = condition.unsqueeze(1)

        scale_bias = self.projection(condition)
        if scale_bias.ndim == 3:
             scale, bias = scale_bias.chunk(2, dim=-1)
             return self.norm(x) * (scale + 1) + bias
        else:
             scale, bias = scale_bias.chunk(2, dim=-1)
             return self.norm(x) * (scale + 1) + bias


# --- Transformer Block 模块 ---
class TransformerBlock(nn.Module):
    def __init__(self, model_dim, num_heads, dropout, condition_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = AdaLN(size=model_dim, condition_dim=condition_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim)
        )
        self.norm2 = AdaLN(size=model_dim, condition_dim=condition_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, condition):
        # 自注意力模块
        residual = x
        attn_output, _ = self.attention(x, x, x)
        attn_output = self.dropout1(attn_output)
        x = residual + self.norm1(attn_output, condition)

        # 前馈网络模块
        residual = x
        mlp_output = self.mlp(x)
        mlp_output = self.dropout2(mlp_output)
        x = residual + self.norm2(mlp_output, condition)

        return x


# --- Transformer 去噪模型 (核心网络) ---
class TransformerDenoiseModel(nn.Module):
    def __init__(self, latent_dim, expression_dim, model_dim=2048, time_emb_dim=128, num_layers=8, num_heads=8, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.expression_dim = expression_dim
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_p = dropout

        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, model_dim),
            nn.SiLU()
        )

        # 表达值嵌入
        expression_hidden_dim = model_dim // 2
        self.expression_mlp = nn.Sequential(
            nn.Linear(expression_dim, expression_hidden_dim),
            nn.SiLU(),
            nn.Linear(expression_hidden_dim, expression_hidden_dim),
            nn.SiLU(),
            nn.Linear(expression_hidden_dim, model_dim),
            nn.SiLU()
        )

        # 条件组合器
        combined_input_dim = model_dim + model_dim
        combined_emb_dim_for_adaln = model_dim
        self.condition_combiner_mlp = nn.Sequential(
            nn.Linear(combined_input_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, combined_emb_dim_for_adaln),
            nn.SiLU(),
        )

        # 输入投影
        self.input_projection = nn.Linear(1, model_dim)

        # 位置嵌入
        self.positional_embedding = nn.Parameter(torch.randn(1, latent_dim, model_dim))

        # Transformer层
        self.transformer_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.transformer_layers.append(TransformerBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                dropout=dropout,
                condition_dim=combined_emb_dim_for_adaln
            ))

        # 输出投影
        self.output_projection = nn.Linear(model_dim, 1)

        self.print_debug = False

    # --- 关键修改：加入 condition_strength 参数，缩放表达值嵌入 ---
    def forward(self, x, t, expression=None, condition_strength=1.0):
        B, D = x.shape  # x: [B, latent_dim]

        # 输入处理
        x = x.unsqueeze(-1)  # [B, latent_dim, 1]
        x = self.input_projection(x)  # [B, latent_dim, model_dim]
        x = x + self.positional_embedding  # 加位置嵌入

        # 时间嵌入
        time_embedding = self.time_mlp(t)  # [B, model_dim]

        # 表达值嵌入（用 condition_strength 缩放）
        if expression is None:
            expression_embedding = torch.zeros(B, self.model_dim, device=x.device)
        else:
            # 核心：用 condition_strength 动态缩放表达值嵌入的影响
            expression_embedding = self.expression_mlp(expression) * condition_strength

        # 组合条件
        concatenated_condition = torch.cat([time_embedding, expression_embedding], dim=-1)  # [B, model_dim*2]
        combined_condition_for_adaln = self.condition_combiner_mlp(concatenated_condition)  # [B, model_dim]

        # Transformer处理
        for layer in self.transformer_layers:
            x = layer(x, combined_condition_for_adaln)

        # 输出预测噪声
        predicted_noise = self.output_projection(x).squeeze(-1)  # [B, latent_dim]

        return predicted_noise


# --- LatentDiffusionModel 类 (修改为动态条件强度机制) ---
class LatentDiffusionModel(nn.Module):
    def __init__(self, denoise_model, timesteps=1000, beta_schedule='cosine', latent_dim=None, expression_dim=None):
        super().__init__()
        self.denoise_model = denoise_model
        self.timesteps = timesteps

        # 潜在维度和表达值维度
        if latent_dim is None:
             self.latent_dim = denoise_model.latent_dim if hasattr(denoise_model, 'latent_dim') else None
        else:
             self.latent_dim = latent_dim
        if expression_dim is None:
             self.expression_dim = denoise_model.expression_dim if hasattr(denoise_model, 'expression_dim') else None
        else:
             self.expression_dim = expression_dim

        # 扩散过程参数
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

        # 注册缓冲区
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
        if noise is None:
            noise = torch.randn_like(z_0)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, z_0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, z_0.shape)
        return sqrt_alphas_cumprod_t * z_0 + sqrt_one_minus_alphas_cumprod_t * noise

    # --- 预测 z_0 ---
    def predict_z0_from_noise(self, z_t, t, noise):
        sqrt_recip_alphas_cumprod_t = extract(1. / self.sqrt_alphas_cumprod, t, z_t.shape)
        sqrt_recipm1_alphas_cumprod_t = extract(torch.sqrt(1. / self.alphas_cumprod - 1), t, z_t.shape)
        return sqrt_recip_alphas_cumprod_t * z_t - sqrt_recipm1_alphas_cumprod_t * noise

    # --- 后验分布参数 ---
    @torch.no_grad()
    def q_posterior_mean_variance(self, z_0, z_t, t):
        assert z_0.shape == z_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, z_t.shape) * z_0 +
            extract(self.posterior_mean_coef2, t, z_t.shape) * z_t
        )
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, z_t.shape)
        return posterior_mean, posterior_log_variance

    # --- 关键修改：移除 CFG，仅用条件预测噪声 ---
    @torch.no_grad()
    def p_sample(self, z_t, t, expression=None):
        betas_t = extract(self.betas, t, z_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, z_t.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, z_t.shape)

        # 直接用条件预测噪声（无CFG）
        predicted_noise = self.denoise_model(z_t, t, expression)

        model_mean = sqrt_recip_alphas_t * (z_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, z_t.shape)

        noise = torch.randn_like(z_t)
        is_last_step = (t == 0)
        noise = noise.masked_fill(is_last_step.unsqueeze(-1), 0.)

        z_prev = model_mean + (0.5 * posterior_log_variance).exp() * noise
        return z_prev

    # --- 关键修改：采样循环移除 guidance_scale ---
    @torch.no_grad()
    def p_sample_loop(self, shape, device, expression=None, disable_tqdm=False):
        b, d = shape

        if expression is not None:
            if expression.shape[0] != b:
                raise ValueError(f"Expression batch size ({expression.shape[0]}) mismatch with sample batch size ({b})")
            if expression.shape[-1] != self.expression_dim:
                raise ValueError(f"Expression dim ({expression.shape[-1]}) mismatch with model's ({self.expression_dim})")

        z = torch.randn(shape, device=device)  # 从纯噪声开始

        # 采样循环（从T-1到0）
        for i in tqdm(reversed(range(0, self.timesteps)), desc="DDPM Sampling", total=self.timesteps, disable=disable_tqdm):
            t_current = torch.full((b,), i, device=device, dtype=torch.long)
            z = self.p_sample(z, t_current, expression=expression)

        return z

    # --- 关键修改：采样方法移除 guidance_scale ---
    @torch.no_grad()
    def sample(self, batch_size=1, expression=None, disable_tqdm=False):
        if self.latent_dim is None or self.latent_dim <= 0:
            raise ValueError("latent_dim must be valid")
        if expression is not None and expression.shape[0] != batch_size:
            raise ValueError(f"Expression batch size mismatch: {expression.shape[0]} vs {batch_size}")

        latent_device = next(self.denoise_model.parameters()).device
        if expression is not None and expression.device != latent_device:
            expression = expression.to(latent_device)

        return self.p_sample_loop(
            (batch_size, self.latent_dim),
            latent_device,
            expression=expression,
            disable_tqdm=disable_tqdm
        )

    # --- 关键修改：损失函数改为加权MSE（基于表达值和条件强度） ---
    def p_losses(self, z_0, t, expression, condition_strength=1.0, noise=None):
        if noise is None:
            noise = torch.randn_like(z_0)

        z_t = self.q_sample(z_0, t, noise=noise)
        # 用带条件强度的去噪模型预测噪声
        predicted_noise = self.denoise_model(z_t, t, expression, condition_strength=condition_strength)

        # 基础MSE损失
        mse_loss = F.mse_loss(predicted_noise, noise, reduction='none')  # [B, latent_dim]

        # 基于表达值的加权损失（表达值越高，权重越大）
        if expression is not None and condition_strength > 0:
            sample_weights = torch.clamp(expression.squeeze(), min=1e-8)  # [B]（表达值作为权重）
            dynamic_weights = sample_weights * condition_strength  # 受条件强度缩放
            dynamic_weights = dynamic_weights.unsqueeze(1).expand_as(mse_loss)  # [B, latent_dim]
            weighted_loss = (mse_loss * dynamic_weights).mean()
        else:
            weighted_loss = mse_loss.mean()  # 无条件时用普通MSE

        return weighted_loss

    # --- 关键修改：前向方法接收 condition_strength 而非 conditional_dropout_prob ---
    def forward(self, z_0, expression, condition_strength=1.0):
        B, D = z_0.shape
        t = torch.randint(0, self.timesteps, (B,), device=z_0.device).long()
        expression = expression.to(z_0.device)
        return self.p_losses(z_0, t, expression, condition_strength=condition_strength)