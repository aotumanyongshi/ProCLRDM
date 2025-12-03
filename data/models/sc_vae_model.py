import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        # 1. 输入形状修改为 (1,4,80)（对应输入 (batch,1,4,80)）
        input_sample_shape = (1, 4, 80)  # 关键：序列长度从100改为80

        # 2. 辅助函数：计算编码器卷积后展平的特征数（基于新输入形状）
        def _calc_flatten_features(shape):
            # 编码器卷积层需适配80长度，卷积核宽度和padding需对应
            conv_layers = nn.Sequential(
                # 卷积核高度=4（匹配输入高度4），宽度=15（适合80长度）
                nn.Conv2d(in_channels=shape[0], out_channels=16, kernel_size=(shape[1], 15), stride=(1, 1),
                          padding='same' if torch.__version__ >= '1.10' else (0, 7)),  # padding=7（15//2=7，确保same效果）
                nn.ReLU(),
                # 后续卷积核高度=1（不改变高度），宽度=11（适合80长度）
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 11), stride=(1, 1),
                          padding='same' if torch.__version__ >= '1.10' else (0, 5)),  # 11//2=5
                nn.ReLU(),
                # 卷积核宽度=7（适合80长度）
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 7), stride=(1, 1),
                          padding='same' if torch.__version__ >= '1.10' else (0, 3)),  # 7//2=3
                nn.ReLU()
            )
            # 用新输入形状计算特征
            dummy_input = torch.randn(1, *shape)
            with torch.no_grad():
                features = conv_layers(dummy_input)
            return features.numel()  # 输出展平后的特征数

        # 3. 辅助函数：计算编码器卷积后的序列长度（宽度维度）
        def _calc_conv_transpose_input_width(shape):
            conv_layers = nn.Sequential(
                nn.Conv2d(in_channels=shape[0], out_channels=16, kernel_size=(shape[1], 15), stride=(1, 1),
                          padding='same' if torch.__version__ >= '1.10' else (0, 7)),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 11), stride=(1, 1),
                          padding='same' if torch.__version__ >= '1.10' else (0, 5)),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 7), stride=(1, 1),
                          padding='same' if torch.__version__ >= '1.10' else (0, 3)),
                nn.ReLU()
            )
            dummy_input = torch.randn(1, *shape)
            with torch.no_grad():
                features = conv_layers(dummy_input)
            return features.shape[-1]  # 输出卷积后的宽度（应=80，因用了same padding）

        # 4. 基于新输入计算关键尺寸
        flattened_features = _calc_flatten_features(input_sample_shape)  # 基于80长度计算
        conv_output_width = _calc_conv_transpose_input_width(input_sample_shape)  # 应=80
        conv_output_height = 4  # 强制高度=4（对应4种碱基，避免之前的7维问题）

        # 5. 编码器：适配80长度，保持高度=4
        self.encoder = nn.Sequential(
            # 输入: (batch,1,4,80)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, 15), stride=(1, 1),
                      padding='same' if torch.__version__ >= '1.10' else (0, 7)),  # padding=7确保宽度不变
            nn.ReLU(),  # 输出高度=4（因kernel_height=4 + same padding）
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 11), stride=(1, 1),
                      padding='same' if torch.__version__ >= '1.10' else (0, 5)),  # 宽度保持80
            nn.ReLU(),  # 高度保持4（kernel_height=1）
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 7), stride=(1, 1),
                      padding='same' if torch.__version__ >= '1.10' else (0, 3)),  # 宽度保持80
            nn.ReLU(),  # 高度保持4
            nn.Flatten(),
            nn.Linear(in_features=flattened_features, out_features=latent_dim * 2)
        )

        # 6. 解码器：关键调整，确保输出 (batch,1,4,80)
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=flattened_features),
            nn.ReLU(),
            # 还原卷积前的形状：(16,4,80)
            nn.Unflatten(dim=1, unflattened_size=(16, conv_output_height, conv_output_width)),
            # 转置卷积1：保持高度=4，宽度=80
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=(1, 7), stride=(1, 1),
                               padding=(0, 3)),  # 7//2=3，确保宽度不变
            nn.ReLU(),
            # 转置卷积2：保持高度=4，宽度=80
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=(1, 11), stride=(1, 1),
                               padding=(0, 5)),  # 11//2=5，宽度不变
            nn.ReLU(),
            # 转置卷积3：保持高度=4，宽度=80
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=(1, 15), stride=(1, 1),
                               padding=(0, 7)),  # 15//2=7，宽度不变
            nn.ReLU(),
            # 最后一层：输出通道=1，高度=4，宽度=80
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(1, 1), stride=1, padding=(0, 0))
            # kernel_size=(1,1)：不改变高度和宽度，确保输出高度=4，宽度=80
        )

    def encode(self, x):
        x = x.to(next(self.parameters()).device)
        encoded = self.encoder(x)
        mean, logvar = torch.split(encoded, self.latent_dim, dim=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            return torch.sigmoid(logits)
        return logits

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        reconstructed_x_logits = self.decode(z)
        # 打印输出形状，验证是否为 (32,1,4,80)
        print(f"输出形状: {reconstructed_x_logits.shape}")
        return reconstructed_x_logits, mean, logvar

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.latent_dim)
        if next(self.parameters()).is_cuda:
            z = z.to(next(self.parameters()).device)
        generated_output = self.decode(z, apply_sigmoid=True)
        return generated_output