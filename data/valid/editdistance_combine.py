import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import matplotlib
# 强制设置为 Agg 后端（非交互式，适合批量生成图片）
matplotlib.use('Agg')

# 编辑距离计算函数
def gene_edit_distance(seq1, seq2):
    m = len(seq1)
    n = len(seq2)

    dp = np.zeros((m + 1, n + 1))

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            else:
                cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return dp[m][n]


# 读取文件中的序列并随机抽样
def read_sequences_from_file(file_path, sample_size=100):
    with open(file_path, 'r') as file:
        sequences = [line.strip() for line in file.readlines()]

    if len(sequences) > sample_size:
        sequences = random.sample(sequences, sample_size)

    print(f"Read {len(sequences)} sequences from {file_path}")
    return sequences


# 计算组内的编辑距离
def calculate_intragroup_edit_distances(sequences):
    num_sequences = len(sequences)
    distances = np.zeros((num_sequences, num_sequences))

    for i in range(num_sequences):
        for j in range(i + 1, num_sequences):
            distance = gene_edit_distance(sequences[i], sequences[j])
            distances[i][j] = distance
            distances[j][i] = distance

    print(f"Calculated intragroup distances for {num_sequences} sequences.")
    return distances.flatten()


# 计算组间的编辑距离
def calculate_intergroup_edit_distances(group1, group2):
    distances = []

    for seq1 in group1:
        for seq2 in group2:
            distance = gene_edit_distance(seq1, seq2)
            distances.append(distance)

    print(f"Calculated intergroup distances between {len(group1)} and {len(group2)} sequences.")
    return distances


# 绘制组内编辑距离的直方图和累积分布曲线
def plot_intragroup_edit_distances(natural_sequences, gen1_sequences, gen2_sequences):
    # 新建独立画布
    fig, ax = plt.subplots(figsize=(12, 8))
    # 计算组内编辑距离
    natural_distances = calculate_intragroup_edit_distances(natural_sequences)
    random_distances = calculate_intragroup_edit_distances(gen1_sequences)
    gen_distances = calculate_intragroup_edit_distances(gen2_sequences)

    # 绘制组内编辑距离的直方图
    plt.hist(natural_distances, bins=20, density=True, alpha=1, label='Nat', color='#4573AD')
    plt.hist(gen_distances, bins=20, density=True, alpha=0.8, label='Rand', color='#C0B4DA')
    plt.hist(random_distances, bins=20, density=True, alpha=1, label='Gen', color='#D2B88F')
    plt.xlabel('Edit Distance', fontsize=20)
    plt.ylabel('Probability Density', fontsize=20)
    plt.legend(loc='upper left', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # 添加紧凑布局调整以防止裁剪
    plt.tight_layout()
    # plt.savefig('/home/gyj/Work/first/experiments/experiments0/edit_histogram.svg', format='svg')
    plt.savefig('/root/autodl-tmp/sjy/dachang1/data/valid/editdistance/edit_histogram.png')
    plt.close(fig)  # 关闭当前画布，避免影响后续绘图

    # # 绘制组内编辑距离的累积分布曲线
    # plt.plot(np.sort(natural_distances), np.linspace(0, 1, len(natural_distances), endpoint=False),
    #          label='Nat', color='#4370B4', linewidth=4)
    # plt.plot(np.sort(random_distances), np.linspace(0, 1, len(random_distances), endpoint=False),
    #          label='Random', color='#C0B4DA', linewidth=4)
    # plt.plot(np.sort(gen_distances), np.linspace(0, 1, len(gen_distances), endpoint=False),
    #          label='Gen', color='#D2B88F', linewidth=4)
    # plt.xlabel('Edit Distance', fontsize=16)
    # plt.ylabel('Cumulative Probability', fontsize=16)
    # plt.legend(loc='upper left', fontsize=14)
    # # plt.title('Edit Distance Distribution (Intragroup)', fontsize=16)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.savefig('/home/gyj/Work/first/experiments/experiments0/edit_curve.png')
    # plt.show()


# 绘制组间编辑距离的曲线图
def plot_intergroup_edit_distances(natural_sequences, random_sequences, gen_sequences):
    # 新建独立画布
    fig, ax = plt.subplots(figsize=(12, 8))
    # 计算组间编辑距离
    natural_distances = calculate_intergroup_edit_distances(natural_sequences, random_sequences + gen_sequences)
    random_distances = calculate_intergroup_edit_distances(random_sequences, natural_sequences + gen_sequences)
    gen_distances = calculate_intergroup_edit_distances(gen_sequences, natural_sequences + random_sequences)
    # 使用 KDE 曲线图展示
    sns.kdeplot(natural_distances, label='Nat', color='green', linewidth=4, bw_adjust=1.5)
    sns.kdeplot(random_distances, label='Rand', color='orange', linewidth=4, bw_adjust=1.5)
    sns.kdeplot(gen_distances, label='Gen', color='blue', linewidth=4, bw_adjust=1.5)
    plt.xlabel('Edit Distance', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.legend(loc='upper left', fontsize=18)
    # plt.title('Edit Distance Distribution (groups)', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.savefig('/home/gyj/Work/first/experiments/experiments0/edit_distance_groups.svg', format='svg')
    plt.savefig('/root/autodl-tmp/sjy/dachang1/data/valid/editdistance/ecilo_edit_distance_groups.png')
    plt.close(fig)  # 关闭当前画布

# 主函数
def main():
    # 文件路径（替换为实际路径）
    natural_file_path = '/root/autodl-tmp/sjy/dachang1/data/raw_sequences/sc_data/SC_gen_1w_exp_short_A.txt'
    random_file_path = '/root/autodl-tmp/sjy/dachang1/data/raw_sequences/sc_data/random_80bp_dna_sequences.txt'
    gen_file_path = '/root/autodl-tmp/sjy/dachang1/data/raw_sequences/sc_data/clean_sequences.txt'

    # natural_file_path = '/home/gyj/Work/first/diffusiongan/data/SC/SC_gen_1w_seq_short_2w.txt'
    # random_file_path = '/home/gyj/Work/first/diffusiongan/data/SC/SC_Random.txt'
    # gen_file_path = '/home/gyj/Work/first/diffusiongan/optimizer/results_opt/SC_80bp/high_final_sequences_SC.txt'

    # 读取序列
    natural_sequences = read_sequences_from_file(natural_file_path)
    random_sequences = read_sequences_from_file(random_file_path)
    gen_sequences = read_sequences_from_file(gen_file_path)

    # 调用组内编辑距离的绘制函数
    plot_intragroup_edit_distances(natural_sequences, random_sequences, gen_sequences)

    # 调用组间编辑距离的绘制函数
    plot_intergroup_edit_distances(natural_sequences, random_sequences, gen_sequences)


# 执行主程序
if __name__ == "__main__":
    main()
