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
# 保持不变
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
        # x shape: [B, SeqLen, Features] or [B, Features]
        # condition shape: [B, ConditionDim]
        # If x is [B, SeqLen, Features], we need to broadcast condition to [B, 1, ConditionDim]
        # LayerNorm is applied on the last dimension (Features)
        if x.ndim == 3 and condition.ndim == 2:
            condition = condition.unsqueeze(1)  # [B, 1, ConditionDim] broadcast over SeqLen

        scale_bias = self.projection(condition)
        if scale_bias.ndim == 3:  # If condition was unsqueezed, scale_bias is [B, 1, Size*2]
            scale, bias = scale_bias.chunk(2, dim=-1)  # scale/bias are [B, 1, Size]
            return self.norm(x) * (scale + 1) + bias  # Broadcasting works
        else:  # x is [B, Features], condition is [B, ConditionDim]
            scale, bias = scale_bias.chunk(2, dim=-1)  # scale/bias are [B, Size]
            return self.norm(x) * (scale + 1) + bias


# --- 新增 Transformer Block 模块 ---
class TransformerBlock(nn.Module):
    def __init__(self, model_dim, num_heads, dropout, condition_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Input is [B, SeqLen, Features]
        )
        # Use AdaLN after attention and before the residual connection
        self.norm1 = AdaLN(size=model_dim, condition_dim=condition_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),  # Feed-forward expansion
            nn.GELU(),  # Common activation in Transformers
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim)
        )
        # Use AdaLN after MLP and before the residual connection
        self.norm2 = AdaLN(size=model_dim, condition_dim=condition_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, condition):
        # x is [B, SeqLen, Features] -> [B, latent_dim, model_dim]
        # condition is [B, condition_dim] -> [B, model_dim]

        # Self-Attention Block
        residual = x
        # MultiheadAttention output is (attn_output, attn_output_weights)
        attn_output, _ = self.attention(x, x, x)  # Q, K, V are all x
        attn_output = self.dropout1(attn_output)
        # Apply AdaLN to the output of attention
        # condition shape is [B, model_dim], AdaLN expects [B, ConditionDim] when x is 2D or [B, 1, ConditionDim] when x is 3D for broadcasting
        x = residual + self.norm1(attn_output, condition)  # Pass condition directly, AdaLN handles unsqueeze if needed.

        # Feed-Forward Block
        residual = x
        mlp_output = self.mlp(x)
        mlp_output = self.dropout2(mlp_output)
        # Apply AdaLN to the output of MLP
        x = residual + self.norm2(mlp_output, condition)  # Pass condition directly, AdaLN handles unsqueeze if needed.

        return x  # Output is [B, latent_dim, model_dim]


# --- Transformer 去噪模型 (核心网络) ---
# 替换 VectorDenoiseModel
class TransformerDenoiseModel(nn.Module):
    def __init__(self, latent_dim, expression_dim, model_dim=2048, time_emb_dim=128, num_layers=8, num_heads=8,
                 dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.expression_dim = expression_dim
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_p = dropout

        # Embedding for time and expression conditions (same as before)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, model_dim),
            nn.SiLU()
        )

        expression_hidden_dim = model_dim // 2
        self.expression_mlp = nn.Sequential(
            nn.Linear(expression_dim, expression_hidden_dim),
            nn.SiLU(),
            nn.Linear(expression_hidden_dim, expression_hidden_dim),
            nn.SiLU(),
            nn.Linear(expression_hidden_dim, model_dim),
            nn.SiLU()
        )

        # Combiner for time and expression embeddings (same as before)
        combined_input_dim = model_dim + model_dim
        combined_emb_dim_for_adaln = model_dim  # Condition dimension for AdaLN

        self.condition_combiner_mlp = nn.Sequential(
            nn.Linear(combined_input_dim, model_dim),  # Maps concatenated dim to condition_dim
            nn.SiLU(),
            nn.Linear(model_dim, combined_emb_dim_for_adaln),  # Maps to final condition dim for AdaLN
            nn.SiLU(),
        )

        # Input layer: Project latent vector to sequence of tokens with model_dim features
        # Treat latent_dim as sequence length, input feature dimension is 1.
        # Input: [B, latent_dim] -> unsqueeze -> [B, latent_dim, 1] -> projection -> [B, latent_dim, model_dim]
        self.input_projection = nn.Linear(1, model_dim)

        # Positional encoding for the latent dimensions (sequence length = latent_dim)
        self.positional_embedding = nn.Parameter(torch.randn(1, latent_dim, model_dim))  # Learned positional embedding

        # Transformer Encoder Layers
        self.transformer_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.transformer_layers.append(TransformerBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                dropout=dropout,
                condition_dim=combined_emb_dim_for_adaln  # AdaLN condition dimension
            ))

        # Output layer: Map sequence of tokens back to the latent vector shape
        # Input: [B, latent_dim, model_dim] -> output projection -> [B, latent_dim, 1] -> squeeze -> [B, latent_dim] (predicted noise)
        self.output_projection = nn.Linear(model_dim, 1)

        self.print_debug = False  # Keeping the debug flag

    def forward(self, x, t, expression=None, condition_strength=1.0):
        B, D = x.shape  # x is [B, latent_dim]

        # Reshape input for Transformer: [B, latent_dim, 1]
        x = x.unsqueeze(-1)

        # Project to model_dim features: [B, latent_dim, model_dim]
        x = self.input_projection(x)

        # Add positional embedding
        x = x + self.positional_embedding

        # Compute conditional embedding
        time_embedding = self.time_mlp(t)  # [B, model_dim]

        # 表达值嵌入（用condition_strength缩放）
        if expression is None:
            # 无表达值时，用0填充
            expression_embedding = torch.zeros(B, self.model_dim, device=x.device)
        else:
            # 有表达值时，计算嵌入后用condition_strength缩放
            expression_embedding = self.expression_mlp(expression) * condition_strength

        # 组合时间和表达值嵌入
        concatenated_condition = torch.cat([time_embedding, expression_embedding], dim=-1)  # [B, model_dim * 2]

        # 处理组合条件
        combined_condition_for_adaln = self.condition_combiner_mlp(concatenated_condition)  # [B, model_dim]

        if self.print_debug:
            print(f"Input latent shape: {x.shape}")
            print(f"Combined condition shape for AdaLN: {combined_condition_for_adaln.shape}")
            print(f"Condition strength: {condition_strength}")  # 新增打印条件强度

        # 通过Transformer层处理序列
        for layer in self.transformer_layers:
            x = layer(x, combined_condition_for_adaln)  # x is [B, latent_dim, model_dim]

        # 输出预测噪声
        predicted_noise = self.output_projection(x).squeeze(-1)  # [B, latent_dim]

        if self.print_debug:
            print(f"Predicted noise mean: {predicted_noise.mean().item():.6f}")
            print(f"Predicted noise std: {predicted_noise.std().item():.6f}")

        return predicted_noise


# --- LatentDiffusionModel 类 (移除分阶段条件强化) ---
class LatentDiffusionModel(nn.Module):
    def __init__(self, denoise_model, timesteps=1000, beta_schedule='cosine', latent_dim=None, expression_dim=None):
        super().__init__()
        self.denoise_model = denoise_model
        self.timesteps = timesteps

        if latent_dim is None:
            if hasattr(denoise_model, 'latent_dim'):
                self.latent_dim = denoise_model.latent_dim
            else:
                raise ValueError("latent_dim must be provided during initialization or inferrable from denoise_model")
        else:
            self.latent_dim = latent_dim

        if expression_dim is None:
            if hasattr(denoise_model, 'expression_dim'):
                self.expression_dim = denoise_model.expression_dim
            else:
                raise ValueError(
                    "expression_dim must be provided during initialization or inferrable from denoise_model")
        else:
            self.expression_dim = expression_dim

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

        # calculations for posterior q(z_{t-1} | z_t, z_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

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
        """
        根据 z_t 和预测的噪声 epsilon，估计原始的 z_0
        z_0 = (z_t - sqrt(1 - alpha_cumprod_t) * noise) / sqrt(alpha_cumprod_t)
        """
        sqrt_recip_alphas_cumprod_t = extract(1. / self.sqrt_alphas_cumprod, t, z_t.shape)
        sqrt_recipm1_alphas_cumprod_t = extract(torch.sqrt(1. / self.alphas_cumprod - 1), t, z_t.shape)

        pred_z0 = sqrt_recip_alphas_cumprod_t * z_t - sqrt_recipm1_alphas_cumprod_t * noise
        return pred_z0

    # --- 后验分布 q(z_{t-1} | z_t, z_0) 的均值和方差 (DDPM 采样使用) ---
    @torch.no_grad()
    def q_posterior_mean_variance(self, z_0, z_t, t):
        assert z_0.shape == z_t.shape
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, z_t.shape) * z_0 +
                extract(self.posterior_mean_coef2, t, z_t.shape) * z_t
        )
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, z_t.shape)
        return posterior_mean, posterior_log_variance

    # --- 修改 p_sample 以移除 CFG ---
    @torch.no_grad()
    def p_sample(self, z_t, t, expression=None):
        """
        DDPM 采样一步：从 z_t 采样 z_{t-1}
        """
        betas_t = extract(self.betas, t, z_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, z_t.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, z_t.shape)

        predicted_noise = self.denoise_model(z_t, t, expression)

        model_mean = sqrt_recip_alphas_t * (z_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, z_t.shape)

        noise = torch.randn_like(z_t)
        is_last_step = (t == 0)
        noise = noise.masked_fill(is_last_step.unsqueeze(-1), 0.)

        z_prev = model_mean + (0.5 * posterior_log_variance).exp() * noise

        return z_prev

    # --- 完整的采样循环，移除 CFG 相关 ---
    @torch.no_grad()
    def p_sample_loop(self, shape, device, expression=None, disable_tqdm=False):
        """
        完整的采样循环。
        """
        b, d = shape

        if expression is not None:
            if expression.shape[0] != b:
                raise ValueError(
                    f"Expression batch size ({expression.shape[0]}) must match sampling shape batch size ({b})")
            if expression.shape[-1] != self.expression_dim:
                raise ValueError(
                    f"Expression dimension ({expression.shape[-1]}) must match model's expression_dim ({self.expression_dim}).")

        # Start from pure noise z_T
        z = torch.randn(shape, device=device)

        # Standard DDPM loop iterates from T-1 down to 0
        sample_steps_desc = reversed(range(0, self.timesteps))
        tqdm_desc = "DDPM Sampling"

        for i in tqdm(sample_steps_desc, desc=tqdm_desc, total=self.timesteps, disable=disable_tqdm):
            t_current = torch.full((b,), i, device=device, dtype=torch.long)
            z = self.p_sample(z, t_current, expression=expression)

        return z  # Return the generated z_0

    # --- Sample method (wrapper) - 移除 guidance_scale 参数 ---
    @torch.no_grad()
    def sample(self, batch_size=1, expression=None, disable_tqdm=False):
        """
        生成指定数量的潜在样本 z_0
        """
        if self.latent_dim is None or self.latent_dim <= 0:
            raise ValueError("latent_dim must be provided during initialization or inferrable from denoise_model")

        if expression is not None and expression.shape[0] != batch_size:
            raise ValueError(
                f"Expression batch size ({expression.shape[0]}) must match requested sample batch size ({batch_size}).")

        latent_device = next(self.denoise_model.parameters()).device

        if expression is not None and expression.device != latent_device:
            expression = expression.to(latent_device)

        return self.p_sample_loop(
            (batch_size, self.latent_dim),
            latent_device,
            expression=expression,
            disable_tqdm=disable_tqdm
        )

    # --- 加权训练损失 表达值越高，说明该样本的条件信息越重要，对应的损失权重也就越大。-  ---
        # 新增参数condition_strength，用于动态调整损失权重
    def p_losses(self, z_0, t, expression, condition_strength=1.0, noise=None):
        if noise is None:
            noise = torch.randn_like(z_0)

        z_t = self.q_sample(z_0, t, noise=noise)
        predicted_noise = self.denoise_model(z_t, t, expression)

        # 基础MSE损失（始终计算）
        mse_loss = F.mse_loss(predicted_noise, noise, reduction='none')  # [B, latent_dim]

        # 只有当有表达值且condition_strength>0时，才应用条件加权
        if expression is not None and condition_strength > 0:
            # 计算样本权重（表达值越高，权重越大）
            sample_weights = torch.clamp(expression.squeeze(), min=1e-8)  # [B]
            # 应用动态强度缩放
            dynamic_weights = sample_weights * condition_strength  # [B]
            # 扩展权重以匹配mse_loss的形状
            dynamic_weights = dynamic_weights.unsqueeze(1).expand_as(mse_loss)  # [B, latent_dim]
            # 加权损失
            weighted_loss = (mse_loss * dynamic_weights).mean()
        else:
            # 无条件或strength=0时，使用未加权的MSE损失
            weighted_loss = mse_loss.mean()

        return weighted_loss

    def forward(self, z_0, expression, condition_strength=1.0):  # 新增condition_strength参数
        B, D = z_0.shape
        t = torch.randint(0, self.timesteps, (B,), device=z_0.device).long()
        expression = expression.to(z_0.device)
        # 传递动态强度到损失计算
        return self.p_losses(z_0, t, expression, condition_strength=condition_strength)