import torch
import numpy as np
import matplotlib
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

# 必须在导入plt前设置后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from sklearn.manifold import TSNE
import umap.umap_ as umap

# ------------------- 手动加载中文字体（关键：解决服务器无中文字体问题） -------------------
from matplotlib.font_manager import FontProperties

# 替换为你上传的中文字体路径（如 simhei.ttf）
# 示例：若字体文件放在项目目录的 fonts 文件夹，路径为 './fonts/simhei.ttf'
font_path = '/root/autodl-tmp/fonts/simhei.ttf'  # 请根据实际路径修改！！！
try:
    font_prop = FontProperties(fname=font_path)
    plt.rcParams['font.sans-serif'] = [font_prop.get_name()]  # 设置字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
except Exception as e:
    print(f"加载字体失败，将尝试默认字体：{e}")
    # 备用方案：若字体文件不存在，降级使用系统默认字体（可能仍无法显示中文）
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

# ------------------- 导入模型 -------------------
try:
    from data.models.sc_vae_model import VAE
except ImportError as e:
    print(f"错误: 无法导入 VAE 模型: {e}")
    exit()

# ------------------- 数据集类 -------------------
class DnaSequenceDataset:
    def __init__(self, data_path, seq_len=80):
        self.data_path = data_path
        self.seq_len = seq_len
        self.sequences = []
        self.char_to_idx = {'A': 0, 'T': 1, 'C': 2, 'G': 3}  # 与模型训练时的映射保持一致
        self.num_classes = len(self.char_to_idx)

        print(f"加载序列数据（长度={seq_len}）: {data_path}")
        try:
            with open(data_path, 'r') as f:
                for line in tqdm(f, desc="读取序列"):
                    seq = line.strip().upper()
                    if len(seq) == self.seq_len and all(c in self.char_to_idx for c in seq):
                        self.sequences.append(seq)
            print(f"有效序列数量: {len(self.sequences)}")
            if not self.sequences:
                raise ValueError(f"未找到长度为{seq_len}且仅含A/T/C/G的序列")
        except Exception as e:
            print(f"数据加载失败: {e}")
            raise

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        indices = torch.tensor([self.char_to_idx[c] for c in seq], dtype=torch.long)
        # 调整one-hot形状为(1,4,80)以匹配模型输入要求
        one_hot = F.one_hot(indices, num_classes=self.num_classes).float().permute(1, 0).unsqueeze(0)
        return one_hot, indices

# ------------------- 参数配置 -------------------
VAE_MODEL_PATH = '/root/autodl-tmp/sjy/dachang1/data/my_vae_model_state_dict.pth'
DATA_PATH = '/root/autodl-tmp/sjy/dachang/data/raw_sequences/sc_data/SC_gen_1w_seq_short_top50000.txt'

# 模型与数据参数
LATENT_DIM = 256  # 与模型定义一致
SEQ_LENGTH = 80    # 序列长度

# 可视化参数（参考代码中的灵活配置）
NUM_SAMPLES_TO_VIZ = 2000
DIM_REDUCTION_METHOD = 'UMAP'  # 可选 'TSNE'
N_COMPONENTS = 2  # 可选 2 或 3（3D可视化）
RANDOM_STATE = 42

# 颜色映射参数（参考代码中的多样选择）
COLOR_BY = 'gc_content'  # 可选: 'gc_content', 'random', 'custom'
CUSTOM_COLORS = None  # 若COLOR_BY='custom'，需提供与样本数匹配的颜色列表

# ------------------- 设备设置 -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")

# ------------------- 加载模型 -------------------
if not os.path.exists(VAE_MODEL_PATH):
    print(f"错误: 模型文件不存在: {VAE_MODEL_PATH}")
    exit()

try:
    checkpoint = torch.load(VAE_MODEL_PATH, map_location=device)
    print(f"模型检查点键值: {list(checkpoint.keys())[:5]}...")

    # 严格按照模型定义初始化
    model = VAE(LATENT_DIM).to(device)

    # 处理分布式训练的参数前缀
    if any(k.startswith('module.') for k in checkpoint.keys()):
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    print("模型参数加载成功")
except Exception as e:
    print(f"模型加载失败: {e}")
    exit()

model.eval()
print("模型已切换至评估模式")

# ------------------- 提取潜在特征 -------------------
dataset = DnaSequenceDataset(DATA_PATH, seq_len=SEQ_LENGTH)
if len(dataset) == 0:
    print("错误: 数据集为空，无法可视化")
    exit()

# 随机采样样本
sample_indices = np.random.choice(len(dataset), min(NUM_SAMPLES_TO_VIZ, len(dataset)), replace=False)
sampled_dataset = Subset(dataset, sample_indices)
sampled_dataloader = DataLoader(sampled_dataset, batch_size=256, shuffle=False, num_workers=4)

all_latent_mu = []
all_indices = []

with torch.no_grad():
    for batch_onehot, batch_indices in tqdm(sampled_dataloader, desc="提取潜在特征"):
        batch_onehot = batch_onehot.to(device)
        mu, logvar = model.encode(batch_onehot)
        all_latent_mu.append(mu.cpu().numpy())
        all_indices.append(batch_indices.numpy())

all_latent_mu = np.concatenate(all_latent_mu, axis=0)
all_indices = np.concatenate(all_indices, axis=0)
print(f"潜在特征形状: {all_latent_mu.shape}（样本数 × 潜在维度）")

# ------------------- 计算颜色映射（参考代码中的多样逻辑） -------------------
print(f"计算颜色映射值 (基于 {COLOR_BY})...")
if COLOR_BY == 'gc_content':
    color_values = []
    for seq_indices in all_indices:
        # 统计C（2）和G（3）的数量
        gc_count = np.sum((seq_indices == 2) | (seq_indices == 3))
        color_values.append(gc_count / len(seq_indices))  # GC含量比例
    cmap = 'coolwarm'
    colorbar_label = 'GC含量'

elif COLOR_BY == 'random':
    # 随机颜色（用于验证聚类效果）
    color_values = np.random.randint(0, 5, size=len(all_indices))
    cmap = 'tab10'
    colorbar_label = '随机类别'

elif COLOR_BY == 'custom' and CUSTOM_COLORS is not None:
    if len(CUSTOM_COLORS) != len(all_indices):
        print(f"警告: 自定义颜色数量与样本数不匹配，使用随机颜色替代")
        color_values = np.random.randint(0, 5, size=len(all_indices))
        cmap = 'tab10'
        colorbar_label = '随机类别'
    else:
        color_values = CUSTOM_COLORS
        cmap = None  # 自定义颜色无需映射

else:
    print(f"不支持的颜色映射方式: {COLOR_BY}，使用GC含量替代")
    color_values = []
    for seq_indices in all_indices:
        gc_count = np.sum((seq_indices == 2) | (seq_indices == 3))
        color_values.append(gc_count / len(seq_indices))
    cmap = 'coolwarm'
    colorbar_label = 'GC含量'

# ------------------- 降维与可视化（支持3D） -------------------
print(f"使用{DIM_REDUCTION_METHOD}将{all_latent_mu.shape[1]}维数据降维到{N_COMPONENTS}维...")
try:
    if DIM_REDUCTION_METHOD == 'UMAP':
        reducer = umap.UMAP(n_components=N_COMPONENTS, random_state=RANDOM_STATE, verbose=True)
    elif DIM_REDUCTION_METHOD == 'TSNE':
        reducer = TSNE(n_components=N_COMPONENTS, random_state=RANDOM_STATE, verbose=1)
    else:
        raise ValueError(f"不支持的降维方法: {DIM_REDUCTION_METHOD}")

    latent_reduced = reducer.fit_transform(all_latent_mu)
    print(f"降维后形状: {latent_reduced.shape}")
except Exception as e:
    print(f"降维失败: {e}")
    exit()

# 绘制可视化结果（参考代码中的2D/3D支持）
plt.figure(figsize=(10, 8))

if N_COMPONENTS == 2:
    scatter = plt.scatter(
        latent_reduced[:, 0],
        latent_reduced[:, 1],
        c=color_values,
        cmap=cmap,
        alpha=0.7,
        s=15,
        edgecolors='none'
    )
    plt.xlabel(f'{DIM_REDUCTION_METHOD}  Component 1', fontsize=12)
    plt.ylabel(f'{DIM_REDUCTION_METHOD} Component 2', fontsize=12)

elif N_COMPONENTS == 3:
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.figure(figsize=(12, 10)).add_subplot(111, projection='3d')
    scatter = ax.scatter(
        latent_reduced[:, 0],
        latent_reduced[:, 1],
        latent_reduced[:, 2],
        c=color_values,
        cmap=cmap,
        alpha=0.7,
        s=15,
        edgecolors='none'
    )
    ax.set_xlabel(f'{DIM_REDUCTION_METHOD} 分量 1', fontsize=12)
    ax.set_ylabel(f'{DIM_REDUCTION_METHOD} 分量 2', fontsize=12)
    ax.set_zlabel(f'{DIM_REDUCTION_METHOD} 分量 3', fontsize=12)

# 添加颜色条和标题
if cmap is not None:
    cbar = plt.colorbar(scatter)
    cbar.set_label(colorbar_label, fontsize=12)

plt.title(f'VAE Latent Space Visualization (2D) - Colored by GC Content', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# 保存图片（包含颜色映射信息）
save_path = f'vae_latent_{DIM_REDUCTION_METHOD.lower()}_{N_COMPONENTS}d_colored_by_{COLOR_BY}.png'
plt.savefig(save_path, dpi=300)
print(f"可视化结果已保存至: {save_path}")