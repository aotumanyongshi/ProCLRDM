import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import stats

# 必须在导入pyplot之前设置后端
matplotlib.use('Agg')


def calculate_kmer_frequency(sequence, k):
    """计算指定长度k的kmer频率"""
    kmer_count = {}
    total_kmers = 0

    if k == 4:
        for i in range(len(sequence) - 3):
            kmer = sequence[i:i + 4]
            if kmer in kmer_count:
                kmer_count[kmer] += 1
            else:
                kmer_count[kmer] = 1
            total_kmers += 1
    elif k == 5:
        for i in range(len(sequence) - 4):
            kmer = sequence[i:i + 5]
            if kmer in kmer_count:
                kmer_count[kmer] += 1
            else:
                kmer_count[kmer] = 1
            total_kmers += 1
    elif k == 6:
        for i in range(len(sequence) - 5):
            kmer = sequence[i:i + 6]
            if kmer in kmer_count:
                kmer_count[kmer] += 1
            else:
                kmer_count[kmer] = 1
            total_kmers += 1
    else:
        raise ValueError("k必须为4、5或6")

    kmer_frequency = {}
    for kmer, count in kmer_count.items():
        frequency = count / total_kmers
        kmer_frequency[kmer] = frequency

    return kmer_frequency


def compare_kmer_frequencies(generated_sequence, natural_sequence, k, output_path):
    """比较两个序列在指定k值下的kmer频率"""
    generated_freq = calculate_kmer_frequency(generated_sequence, k)
    natural_freq = calculate_kmer_frequency(natural_sequence, k)

    x = []
    y = []

    for kmer, freq in generated_freq.items():
        x.append(freq)
        y.append(natural_freq.get(kmer, 0))  # 如果自然序列中没有该k-mer，设频率为0

    # 线性回归分析
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * np.array(x) + intercept

    # 绘制散点图（保持原有样式）
    plt.scatter(x, y)  # 散点图全部
    # plt.plot(x, line, color='lightcoral')  # 趋势线（原代码注释掉了，如需显示可取消注释）

    plt.xlabel('Generated', fontsize=20)
    plt.ylabel('Natural', fontsize=20)
    plt.title(f'All region:{k}-mer', fontsize=20)
    plt.tick_params(axis='both', labelsize=18)  # 增加坐标轴数值字体大小

    # 关键：保存图表而非显示
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图表释放资源
    print(f"图表已保存至: {output_path}")


def calculate_pearson_correlation(seq1, seq2, k=5):
    """计算两个序列在指定k值下的Pearson相关性"""
    freq1 = calculate_kmer_frequency(seq1, k)
    freq2 = calculate_kmer_frequency(seq2, k)

    common_kmers = set(freq1.keys()) & set(freq2.keys())

    x = np.array([freq1[kmer] for kmer in common_kmers])
    y = np.array([freq2[kmer] for kmer in common_kmers])

    correlation = np.corrcoef(x, y)[0, 1]

    return correlation


# 从文件中读取生成的序列和自然序列
with open("/root/autodl-tmp/sjy/dachang1/data/raw_sequences/vae+con+diffusion/select_gen.txt", 'r') as gen_file, open(
        "/root/autodl-tmp/sjy/dachang1/data/raw_sequences/ecoli_data/ecoli_data.txt", 'r') as natural_file:
    gen_sequence = gen_file.read().strip()
    natural_sequence = natural_file.read().strip()

# 灵活测试不同长度的kmer
k_values = [4, 5, 6]  # 可修改为需要测试的kmer长度
for k in k_values:
    output_path = f'/root/autodl-tmp/sjy/dachang/data/valid/kmer_{k}mer_comparison.png'
    compare_kmer_frequencies(gen_sequence, natural_sequence, k, output_path)

    # 计算相似性
    correlation = calculate_pearson_correlation(gen_sequence, natural_sequence, k)
    print(f"{k}-mer Pearson Correlation: {correlation}")