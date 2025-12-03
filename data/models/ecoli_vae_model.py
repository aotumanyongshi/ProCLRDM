# vae_model_torch.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import pandas as pd # 如果你不需要在 ecoli_vae_model.py 中进行 pandas 操作，可以不导入




class VAE(nn.Module): # 继承 torch.nn.Module
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        input_sample_shape = (1, 4, 185)

        # Helper 函数用于计算展平后的特征数量
        def _calc_flatten_features(shape):
            """
            计算经过编码器卷积层后展平的特征数量。
            shape 是单个样本的输入形状 (C, H, W)。
            """
            conv_layers = nn.Sequential(
                nn.Conv2d(in_channels=shape[0], out_channels=16, kernel_size=(shape[1], 35), stride=(1, 1),
                          padding='same' if torch.__version__ >= '1.10' else (0, 0)),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 21), stride=(1, 1),
                          padding='same' if torch.__version__ >= '1.10' else (0, 0)),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 15), stride=(1, 1),
                          padding='same' if torch.__version__ >= '1.10' else (0, 0)),
                nn.ReLU()
            )

            # 创建一个假（dummy）的输入张量，添加一个批次维度
            dummy_input = torch.randn(1, *shape)

            # 通过卷积层部分获取输出张量
            with torch.no_grad():
                features = conv_layers(dummy_input)

            # 返回输出张量的元素总数（展平后的特征数量）
            return features.numel()

        # Helper 函数用于计算转置卷积输入前的宽度维度
        # 这个尺寸应该与编码器 Flatten 前的最后一个卷积层的序列长度维度相匹配
        def _calc_conv_transpose_input_width(shape):
            """
            计算经过编码器卷积层后张量在宽度维度上的尺寸。
            shape 是单个样本的输入形状 (C, H, W)。
            """
            # 创建一个包含编码器卷积部分的临时 Sequential 模型
            # !!! 这里的 Conv2d 层参数 (in_channels, out_channels, kernel_size, stride, padding)
            #     必须与你实际定义的 self.encoder 中前几个 Conv2d 层完全一致 !!!
            conv_layers = nn.Sequential(
                nn.Conv2d(in_channels=shape[0], out_channels=16, kernel_size=(shape[1], 35), stride=(1, 1),
                          padding='same' if torch.__version__ >= '1.10' else (0, 0)),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 21), stride=(1, 1),
                          padding='same' if torch.__version__ >= '1.10' else (0, 0)),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 15), stride=(1, 1),
                          padding='same' if torch.__version__ >= '1.10' else (0, 0)),
                nn.ReLU()
            )

            # 创建一个假（dummy）的输入张量，添加一个批次维度
            dummy_input = torch.randn(1, *shape)

            # 通过卷积层部分获取输出张量
            with torch.no_grad():
                features = conv_layers(dummy_input)

            # 返回输出张量在宽度维度上的大小 (通常是最后一个空间维度)
            return features.shape[-1]

        # --- 计算所需的尺寸 ---
        flattened_features = _calc_flatten_features(input_sample_shape)
        conv_output_width = _calc_conv_transpose_input_width(input_sample_shape)
        conv_output_height = input_sample_shape[1]  # 假设高度始终为 1，如果编码器改变了高度，需要修改这里的计算
        # 编码器定义 - **这里是需要根据 165bp 序列进行重大修改的部分**
        # input_shape 在 PyTorch 的 Sequential 中通常由第一层隐式定义
        # 假设经过 one_hot_encode_torch 后张量形状为 (batch_size, 4, 1, 185)
        self.encoder = nn.Sequential(
            # Input: (batch_size, 1, 4, 185) - 根据你的新输入形状修改注释
            # Conv2d 参数需要根据新的输入形状 (1, 4, 185) 重新计算 kernel_size 和 padding
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, 35), stride=(1, 1),
                      padding='same' if torch.__version__ >= '1.10' else (0, 0)),
            # in_channels 改为 1, kernel_size 改为 (4, 35)
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 21), stride=(1, 1),
                      padding='same' if torch.__version__ >= '1.10' else (0, 0)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 15), stride=(1, 1),
                      padding='same' if torch.__version__ >= '1.10' else (0, 0)),
            nn.ReLU(),
            nn.Flatten(),
            # Dense 层的输入特征数量需要根据新的卷积层输出尺寸重新计算
            nn.Linear(in_features=flattened_features, out_features=latent_dim * 2)  # flattened_features 的计算需要基于新的编码器结构
        )
        # 解码器定义 - **这里同样需要根据 165bp 序列进行重大修改**
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=flattened_features),  # 输出特征数应与编码器展平前匹配
            nn.ReLU(),
            # Unflatten 到卷积层输出前的形状，需要指定正确的尺寸
            nn.Unflatten(dim=1, unflattened_size=(16, conv_output_height, conv_output_width)),
            # (batch_size, 16, 1, 185)
            # Channels=16, Height=conv_output_height, Width=conv_output_width
            # !!! 确保这里的 Channels (16) 与编码器最后一个卷积层的 out_channels 一致 !!!
            # !!! 确保这里的 Height 和 Width 计算正确，与编码器 Flatten 前的张量尺寸完全一致 !!!
            # 转置卷积层参数需要重新计算，以确保能够恢复到原始输入尺寸 (batch_size, 4, 1, 185)
            nn.ConvTranspose2d(in_channels=16,out_channels=16,kernel_size=(3, 15),stride=(1, 1),padding=(1, 7),output_padding=(0, 0)),
            # 这些参数是示例，需要根据你的精确计算来调整!
            # 需要调整 padding 和 output_padding
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16,out_channels=16,kernel_size=(3, 15),stride=(1, 1),padding=(1, 7),output_padding=(0, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16,out_channels=16,kernel_size=(3, 15),stride=(1, 1),padding=(1, 7),output_padding=(0, 0)),
            nn.ReLU(),
            # 最后的转置卷积层输出通道应为 4， kernel_size 需要调整以恢复到 185 的宽度
            # padding 和 output_padding 的设置对于精确恢复尺寸至关重要
            nn.ConvTranspose2d(in_channels=16,out_channels=1,kernel_size=(3, 3),stride=1,padding=(1, 1),output_padding=(0, 0)
            )
            # <<< 修改 kernel_size (高度, 宽度) 和 padding/output_padding 以恢复 1x185
        )

    def encode(self, x):

        # x 应该是经过预处理的输入张量，形状如 (batch_size, 4, 1, 185)

        #print(f"x{x.shape}")

        # !!! 确保 x 在进入编码器之前位于正确的设备上 !!!

        x = x.to(next(self.parameters()).device)

        encoded = self.encoder(x)
        #print(f"encoded{encoded.shape}")

        mean, logvar = torch.split(encoded, self.latent_dim, dim=1)  # 沿特征维度分割

        return mean, logvar

    def reparameterize(self, mean, logvar):

        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)  # 生成与 std 同形状的随机张量

        return mean + eps * std

    def decode(self, z, apply_sigmoid=False):

        # z 是从潜在空间采样的点，形状为 (batch_size, latent_dim)

        logits = self.decoder(z)

        if apply_sigmoid:

        # 通常 VAE 输出层没有激活函数，在这里计算 sigmoid 以获得概率

            return torch.sigmoid(logits)

        return logits

    # PyTorch 中模型的标准前向传播方法
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        reconstructed_x_logits = self.decode(z)
        return reconstructed_x_logits, mean, logvar

    # 用于从潜在空间采样并生成新序列
    def sample(self, num_samples):
        # 从标准正态分布采样潜在向量
        z = torch.randn(num_samples, self.latent_dim)
         # 将采样张量移动到模型所在的设备，以便进行推理
        if next(self.parameters()).is_cuda: # 检查模型是否在 GPU 上
             z = z.to(next(self.parameters()).device)

        # 解码潜在向量以生成序列输出
        # 通常在采样时直接获得概率输出 (apply_sigmoid=True)
        generated_output = self.decode(z, apply_sigmoid=True)
        return generated_output