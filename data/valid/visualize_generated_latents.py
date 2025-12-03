import torch
import numpy as np
import matplotlib
# 必须在导入 plt 前设置 matplotlib 后端，特别是在无 GUI 环境下
from torch.utils.data import DataLoader, Dataset  # 导入 Dataset

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from sklearn.manifold import TSNE
import umap.umap_ as umap
# 导入自定义的模型类 VAE 和 修改后的扩散模型类
# 导入训练时使用的 SequenceDataset 类

from models.vae_model import VAE  # 导入您的 VAE 模型类
# 导入修改后的扩散模型类
from models.diffusion_model import TransformerDenoiseModel, LatentDiffusionModel
# 这里假设 SequenceDataset 可以从其他地方导入，如果没有提供需要补充实现
from utils.utils import SequenceDataset
import torch.nn.functional as F  # 确保 F 也被导入

# 设置支持中文的字体 (如果您的运行环境支持并需要中文显示)
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei', 'Microsoft YaHei', 'Arial Unicode MS',
                                       'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except Exception:
    print("警告: 设置中文字体失败，可能您的环境中没有该字体。图表中文显示可能不正常。")

# --- 设置参数 ---
# 模型路径
# <--- 请务必修改为您的训练好的 VAE 和 Latent Diffusion Model 检查点路径 --->
VAE_MODEL_PATH = '/root/autodl-tmp/sjy/dachang1/my_vae_model_state_dict.pth'  # <-- 使用您最新训练好的 VAE 模型 (确保是训练 LatentDiffusionModel 时使用的那个)
DIFFUSION_MODEL_PATH = '/root/autodl-tmp/sjy/dachang1/best_model_overall.pth'  # <-- 修改为您的训练好的 Latent Diffusion Model 检查点

# 数据路径
DATA_PATH = '/user/data1/sujianye/dachangdiffusion/data/raw_sequences/ecoli_data.txt'  # <-- 原始 DNA 数据文件路径

# 模型架构参数 (必须与您训练时使用的参数完全一致)
# <--- VAE 参数必须与您训练 VAE 时使用的参数完全一致 --->
# VAE 模型构造函数只接受 latent_dim 参数
LATENT_DIM_VAE_CONSTRUCTOR = 256  # <--- 你的 VAE 构造函数中的 latent_dim (例如 2)

# <--- Latent Diffusion Model 参数必须与您训练 Latent Diffusion Model 时使用的参数完全一致 --->
# 根据你修改后的 diffusion_model.py 和 train_ecoli_diffusion.py 来设置这些参数
LATENT_VECTOR_DIM = 256  # <--- VAE encode 返回的潜在向量的维度 (例如 2)
DIFFUSION_MODEL_DIM = 2048  # TransformerDenoiseModel 的内部维度
TIME_EMB_DIM = 128  # TransformerDenoiseModel 的时间嵌入维度
NUM_LAYERS = 8  # TransformerDenoiseModel 的层数
NUM_HEADS = 8  # TransformerDenoiseModel 的头数
DROPOUT = 0.1  # TransformerDenoiseModel 的丢弃率

DIFFUSION_TIMESTEPS = 1000  # <--- 扩散模型的步数
DIFFUSION_BETA_SCHEDULE = 'cosine'  # <--- 扩散模型的 beta 调度策略

# SequenceDataset 的参数 (必须与训练 VAE 时一致)
SEQ_LENGTH_RAW = 165  # <--- 原始序列长度
PADDING_RAW = 10  # <--- 原始序列填充长度

# 可视化参数
NUM_REAL_SAMPLES_TO_VIZ = 1000  # 用于可视化的真实潜在样本数量
NUM_GENERATED_SAMPLES_TO_VIZ = 1000  # 用于可视化的生成潜在样本数量
DIM_REDUCTION_METHOD = 'UMAP'  # 'UMAP' 或 'TSNE'
N_COMPONENTS = 2  # 降维到的维度 (2 或 3)
RANDOM_STATE = 42  # 用于 UMAP/t-SNE 的随机种子

# --- 设置设备 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")
if device.type == 'cuda':
    print(f"使用的 GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# --- 加载训练好的 VAE 模型 ---
if not os.path.exists(VAE_MODEL_PATH):
    print(f"错误: VAE 模型文件未找到于 {VAE_MODEL_PATH}。")
    exit()

try:
    # 使用与训练时完全一致的参数实例化 VAE 模型
    vae_model = VAE(latent_dim=LATENT_DIM_VAE_CONSTRUCTOR).to(device)  # <--- 使用正确参数实例化 VAE

    # 加载状态字典 (考虑包装类前缀问题)
    print(f"正在加载 VAE 模型状态字典从 {VAE_MODEL_PATH}...")
    original_state_dict = torch.load(VAE_MODEL_PATH, map_location=device)
    new_state_dict = {}
    prefix = "vae."
    found_prefix_and_processed = False
    try:
        temp_state_dict = {}
        for key, value in original_state_dict.items():
            if key.startswith(prefix):
                temp_state_dict[key[len(prefix):]] = value
                found_prefix_and_processed = True
            else:
                found_prefix_and_processed = False
                break
        if found_prefix_and_processed:
            new_state_dict = temp_state_dict
    except Exception as e:
        print(f"尝试处理前缀时发生错误: {e}. 将尝试直接加载原始 state_dict.")
        found_prefix_and_processed = False

    if found_prefix_and_processed:
        print("成功处理 state_dict 前缀。尝试加载处理后的 state_dict...")
        vae_model.load_state_dict(new_state_dict, strict=True)
    else:
        print("未检测到或处理前缀成功。尝试直接加载原始 state_dict...")
        vae_model.load_state_dict(original_state_dict, strict=True)

    print(f"成功加载训练好的 VAE 模型自 {VAE_MODEL_PATH}")

except Exception as e:
    print(f"加载 VAE 模型状态字典时发生错误: {e}")
    print(f"请确认 VAE 模型文件 {VAE_MODEL_PATH} 存在且完整，以及 LATENT_DIM_VAE_CONSTRUCTOR 参数与训练时一致。")
    exit()

vae_model.eval()  # 设置为评估模式
print("VAE 模型已设置为评估模式。")

# --- 加载训练好的 Latent Diffusion Model ---
if not os.path.exists(DIFFUSION_MODEL_PATH):
    print(f"错误: Diffusion Model 文件未找到于 {DIFFUSION_MODEL_PATH}.")
    print("请确认模型路径是否正确。")
    exit()

try:
    # 实例化 TransformerDenoiseModel
    denoise_model = TransformerDenoiseModel(
        latent_dim=LATENT_VECTOR_DIM,
        expression_dim=0,  # 如果不需要表达条件，设置为 0
        model_dim=DIFFUSION_MODEL_DIM,
        time_emb_dim=TIME_EMB_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT
    ).to(device)

    # 实例化 LatentDiffusionModel
    diffusion_model = LatentDiffusionModel(
        denoise_model=denoise_model,
        timesteps=DIFFUSION_TIMESTEPS,
        beta_schedule=DIFFUSION_BETA_SCHEDULE,
        latent_dim=LATENT_VECTOR_DIM
    ).to(device)

    # 加载状态字典并打印参数
    print(f"\n正在加载 Latent Diffusion Model 状态字典从 {DIFFUSION_MODEL_PATH}...")
    # 先加载状态字典到临时变量
    diffusion_state_dict = torch.load(DIFFUSION_MODEL_PATH, map_location=device)

    # 打印参数信息
    print("\n### Latent Diffusion Model 检查点参数信息 ###")
    print(f"参数总数: {len(diffusion_state_dict.keys())}")
    print("前10个参数名称及形状:")
    # 遍历前10个参数并打印
    for i, (param_name, param_tensor) in enumerate(diffusion_state_dict.items()):
        if i < 10:
            print(f"参数 {i + 1}: {param_name} | 形状: {param_tensor.shape}")
        else:
            break
    if len(diffusion_state_dict.keys()) > 10:
        print(f"... 省略剩余 {len(diffusion_state_dict.keys()) - 10} 个参数")

    # 加载状态字典到模型
    diffusion_model.load_state_dict(diffusion_state_dict)
    print(f"成功加载训练好的 Latent Diffusion Model 自 {DIFFUSION_MODEL_PATH}")

except Exception as e:
    print(f"加载 Latent Diffusion Model 状态字典时发生错误: {e}")
    print(f"请确认 Diffusion Model 文件 {DIFFUSION_MODEL_PATH} 存在且完整，以及模型参数与训练时一致。")
    exit()

diffusion_model.eval()  # 设置为评估模式
print("Latent Diffusion Model 已设置为评估模式。")

# --- 获取真实潜在数据 (从原始序列编码) ---
if not os.path.exists(DATA_PATH):
    print(f"错误: 数据文件未找到于 {DATA_PATH}.")
    exit()

# 使用 SequenceDataset 加载原始数据
dataset = SequenceDataset(DATA_PATH, seq_length=SEQ_LENGTH_RAW, padding=PADDING_RAW)

if len(dataset) == 0:
    print("\n错误: 数据集为空或没有找到符合条件的序列。无法获取真实潜在数据。")
    exit()

# 确保采样数量不超过数据集大小
num_real_samples = min(NUM_REAL_SAMPLES_TO_VIZ, len(dataset))
if num_real_samples == 0:
    print("\n没有足够的真实样本进行编码可视化。请检查数据集。")
    real_latent_mu = np.array([])  # 空数组
else:
    # 随机采样索引
    real_sample_indices = np.random.choice(len(dataset), size=num_real_samples, replace=False)
    real_sampled_dataset = torch.utils.data.Subset(dataset, real_sample_indices)
    # DataLoader 只需用于批量处理，shuffle=False
    real_dataloader = DataLoader(real_sampled_dataset, batch_size=256, shuffle=False,
                                 num_workers=min(os.cpu_count(), 4))

    print(f"\n从 {len(dataset)} 条原始序列中编码 {num_real_samples} 条以获取真实潜在数据...")
    all_real_latent_mu = []
    with torch.no_grad():
        for batch_data in tqdm(real_dataloader, desc="Encoding real data"):
            # batch_data 形状为 (batch_size, channels, height, width)
            if batch_data is None:
                continue  # 跳过无效 batch
            batch_data = batch_data.to(device)  # 移动到设备

            # 使用 VAE 模型的 encode 方法
            # 它应该返回 mean, logvar，形状都是 (batch_size, latent_vector_dim)
            mean, logvar = vae_model.encode(batch_data)
            mu = mean  # 使用均值，形状 (batch_size, latent_vector_dim)

            # 直接收集 mu 的 numpy 数组，不需要 reshape
            all_real_latent_mu.append(mu.cpu().numpy())

    if all_real_latent_mu:
        real_latent_mu = np.concatenate(all_real_latent_mu, axis=0)
        # 连接后形状: [num_real_samples, latent_vector_dim]
        print(f"获取到真实潜在数据形状: {real_latent_mu.shape}")
    else:
        real_latent_mu = np.array([])
        print("未能成功编码任何真实潜在数据。")

# --- 生成潜在数据 (从 Latent Diffusion Model 采样) ---
print(f"\n使用 Latent Diffusion Model 生成 {NUM_GENERATED_SAMPLES_TO_VIZ} 个潜在样本...")
with torch.no_grad():  # 生成过程不需要梯度
    # diffusion_model.sample() 内部会执行采样循环并处理设备
    # 它应该返回形状为 (batch_size, latent_vector_dim) 的生成潜在数据
    generated_latents = diffusion_model.sample(batch_size=NUM_GENERATED_SAMPLES_TO_VIZ)
    # 预期形状: [NUM_GENERATED_SAMPLES_TO_VIZ, latent_vector_dim]

    # 直接将生成的潜在数据移动到 CPU 并转换为 numpy 数组，不需要 reshape
    generated_latent_numpy = generated_latents.cpu().numpy()

print(f"生成潜在数据形状: {generated_latent_numpy.shape}")

# --- 合并真实和生成的潜在数据并创建标签 ---
if real_latent_mu.shape[0] == 0 and generated_latent_numpy.shape[0] == 0:
    print("\n没有真实或生成的潜在数据，无法进行可视化。")
    exit()
if real_latent_mu.shape[0] == 0:
    print("\n没有真实潜在数据，仅可视化生成数据。")
    combined_latent_data = generated_latent_numpy
    labels = ['Generated'] * generated_latent_numpy.shape[0]
elif generated_latent_numpy.shape[0] == 0:
    print("\n没有生成潜在数据，仅可视化真实数据。")
    combined_latent_data = real_latent_mu
    labels = ['Real'] * real_latent_mu.shape[0]
else:
    combined_latent_data = np.concatenate((real_latent_mu, generated_latent_numpy), axis=0)
    labels = ['Real'] * real_latent_mu.shape[0] + ['Generated'] * generated_latent_numpy.shape[0]

print(f"合并潜在数据形状: {combined_latent_data.shape}")
labels = np.array(labels)  # 将标签列表转换为 numpy 数组

# --- 降维处理 ---
# 检查潜在数据的维度。如果维度已经是 2 或 3 (N_COMPONENTS)，则降维可能不是必需的或 t-SNE/UMAP 参数需要调整。
# 但是为了对比，即使维度是 2，进行一次 UMAP/t-SNE 也是可以的。
print(
    f"\n使用 {DIM_REDUCTION_METHOD} 将 {combined_latent_data.shape[1]} 维数据 ({LATENT_VECTOR_DIM}) 降维到 {N_COMPONENTS} 维...")

try:
    if DIM_REDUCTION_METHOD == 'TSNE':
        # t-SNE 对样本数量和 perplexity 敏感
        # 确保 perplexity 不超过 (样本数 - 1) / 3
        perplexity_val = 30
        if combined_latent_data.shape[0] < perplexity_val * 3:
            perplexity_val = max(1, (combined_latent_data.shape[0] - 1) // 3)
            if perplexity_val == 0 and combined_latent_data.shape[0] > 1:
                perplexity_val = 1
            if combined_latent_data.shape[0] <= 1:
                print("\n样本数量过少，无法进行 t-SNE 降维。")
                exit()
            print(f"警告: 样本数量较少 ({combined_latent_data.shape[0]}), 将 perplexity 调整为 {perplexity_val}")

        print(f"使用 {DIM_REDUCTION_METHOD} perplexity: {perplexity_val}")
        reducer = TSNE(n_components=N_COMPONENTS, random_state=RANDOM_STATE, verbose=1, n_jobs=-1,
                       perplexity=perplexity_val)
    elif DIM_REDUCTION_METHOD == 'UMAP':
        # UMAP 通常比 t-SNE 快且更能保留全局结构
        reducer = umap.UMAP(n_components=N_COMPONENTS, random_state=RANDOM_STATE, verbose=True)
    else:
        raise ValueError(f"不支持的降维方法: {DIM_REDUCTION_METHOD}. 请使用 'UMAP' 或 'TSNE'.")

    # 如果潜在空间维度就是 N_COMPONENTS，fit_transform 可能仍然执行，但不会改变形状
    latent_reduced = reducer.fit_transform(combined_latent_data)  # -> [total_samples, N_COMPONENTS]
    print(f"降维完成。结果形状: {latent_reduced.shape}")

except Exception as e:
    print(f"降维过程中发生错误: {e}")
    print(f"请检查您的 UMAP/t-SNE 参数设置是否合理，以及潜在数据的形状 {combined_latent_data.shape} 是否正确。")
    exit()

# --- 可视化 ---
print("\n绘制真实潜在空间与生成潜在空间对比图...")

plt.figure(figsize=(10, 8))

# 使用 seaborn 的 scatterplot 按标签着色
if N_COMPONENTS == 2:
    sns.scatterplot(x=latent_reduced[:, 0], y=latent_reduced[:, 1],
                    hue=labels,  # 使用 'Real'/'Generated' 标签作为颜色
                    palette="viridis",  # 可以选择其他调色板，例如 'coolwarm'
                    alpha=0.6, s=15)
    plt.xlabel(f'{DIM_REDUCTION_METHOD} Component 1')
    plt.ylabel(f'{DIM_REDUCTION_METHOD} Component 2')
    plt.title('VAE Real vs Latent Diffusion Generated Latent Space (2D)')
elif N_COMPONENTS == 3:
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    # 需要手动按标签分配颜色
    colors = {'Real': 'blue', 'Generated': 'red'}  # 定义颜色映射
    for label in np.unique(labels):
        indices = np.where(labels == label)
        ax.scatter(latent_reduced[indices, 0], latent_reduced[indices, 1], latent_reduced[indices, 2],
                   c=colors[label], label=label, s=15, alpha=0.6)
    ax.set_xlabel(f'{DIM_REDUCTION_METHOD} Component 1')
    ax.set_ylabel(f'{DIM_REDUCTION_METHOD} Component 2')
    ax.set_zlabel(f'{DIM_REDUCTION_METHOD} Component 3')
    plt.title('VAE Real vs Latent Diffusion Generated Latent Space (3D)')
    ax.legend()  # 添加图例

plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()  # 自动调整布局

# 创建保存目录
save_dir = "latent_space_comparisons"
os.makedirs(save_dir, exist_ok=True)

# 保存可视化结果
save_path = os.path.join(save_dir, f'latent_space_comparison_{DIM_REDUCTION_METHOD.lower()}_{N_COMPONENTS}d.png')
plt.savefig(save_path, dpi=300)  # 提高分辨率
plt.close()  # 关闭图形，释放内存

print(f"潜在空间对比可视化结果已保存至: {save_path}")

print("\n可视化完成。请查看生成的图片文件。")
print("在图表中，'Real' 点代表原始数据编码的潜在样本，'Generated' 点代表 Latent Diffusion Model 生成的潜在样本。")
print("理想情况下，'Generated' 点应该与 'Real' 点的分布高度重叠并覆盖相似的区域。")