import matplotlib
matplotlib.use('Agg')  # 设置为非交互式后端

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def calculate_gc_content(sequences):
    gc_contents = []
    for seq in sequences:
        gc_count = seq.count('G') + seq.count('C')
        total_count = len(seq)
        gc_content = gc_count / total_count if total_count > 0 else 0
        gc_contents.append(gc_content)
    return gc_contents

def read_sequences_from_file(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

# 文件路径
natural_file = "/root/autodl-tmp/sjy/dachang1/data/raw_sequences/sc_data/SC_gen_1w_seq_short.txt"
random_file = "/root/autodl-tmp/sjy/dachang1/data/valid/random_dna_sequences.txt"
generated_file = "/root/autodl-tmp/sjy/dachang1/data/raw_sequences/sc_data/second_sc_data.txt"

# 读取序列
generated_sequences = read_sequences_from_file(generated_file)
natural_sequences = read_sequences_from_file(natural_file)
random_sequences = read_sequences_from_file(random_file)

# 计算 GC 含量
gc_generated = calculate_gc_content(generated_sequences)
gc_natural = calculate_gc_content(natural_sequences)
gc_random = calculate_gc_content(random_sequences)

# 准备数据用于小提琴图
data = {
    'Generated': gc_generated,
    'Natural': gc_natural,
    'Random': gc_random
}

# 绘制小提琴图，移除showextrema参数
plt.figure(figsize=(10, 6))
sns.violinplot(
    data=list(data.values()),
    inner=None,           # 不显示内部箱线图
    bw_method='scott',    # 替代弃用的bw参数
    density_norm='width', # 替代弃用的scale参数
    alpha=0.9
)

# 计算中位数和四分位数
medians = [np.median(gc) for gc in data.values()]
q1 = [np.percentile(gc, 25) for gc in data.values()]
q3 = [np.percentile(gc, 75) for gc in data.values()]

# 绘制中位数和四分位线
ind = np.arange(len(data))

plt.vlines(ind, q1, q3, color='black', lw=5)
plt.scatter(ind, medians, color='white', marker='o', s=100, zorder=3)

# 添加箱线图作为补充
sns.boxplot(
    data=list(data.values()),
    color='black',
    boxprops=dict(edgecolor='black', linewidth=1.5),
    whiskerprops=dict(color='black'),
    capprops=dict(color='black'),
    medianprops=dict(color='black'),
    width=0.1,  # 减小箱线图的宽度
    fliersize=2
)

# 设置 x 轴标签和标题
plt.xticks(ticks=ind, labels=data.keys(), fontsize=20)
plt.ylabel('GC Content', fontsize=20)
plt.tick_params(axis='both', labelsize=24)  # 增加坐标轴数值字体大小

# 增加坐标轴粗细
for axis in ['top', 'bottom', 'left', 'right']:
    plt.gca().spines[axis].set_linewidth(4)

# 保存和显示图形
output_path = '/root/autodl-tmp/sjy/dachang1/data/valid/GC/gc_SC.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"图表已保存至: {output_path}")