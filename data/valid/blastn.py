import os
import matplotlib
matplotlib.use('Agg')  # 非图形环境后端
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def open_BLAST(file_name):
    """解析BLAST结果，提取E值（支持表格行和Score行）"""
    e_values = []
    with open(file_name, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    # 阶段1：解析"Sequences producing significant alignments"表格
    table_header_idx = None
    for i, line in enumerate(lines):
        if "E Value" in line and "Scientific Name" in line:
            table_header_idx = i
            break

    if table_header_idx is not None:
        for line in lines[table_header_idx + 1:]:
            if line.startswith("Alignments:"):
                break
            if not line:
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            # 从标题行找E值列索引
            header_parts = lines[table_header_idx].split()
            try:
                e_col_idx = header_parts.index("E")
                if e_col_idx < len(parts):
                    e_value = float(parts[e_col_idx])
                    e_values.append(e_value)
            except (ValueError, IndexError):
                # 手动遍历列找E值
                for i in range(5, 10):
                    if i < len(parts):
                        try:
                            e_value = float(parts[i])
                            e_values.append(e_value)
                            break
                        except ValueError:
                            continue

    # 阶段2：解析Score行的E值
    for line in lines:
        if line.startswith("Score:") and "Expect:" in line:
            try:
                expect_part = line.split("Expect:")[1].split(",")[0].strip()
                e_value = float(expect_part)
                e_values.append(e_value)
            except (IndexError, ValueError):
                print(f"跳过无效行: {line}")

    # 去重、过滤负数
    e_values = list(set(e_values))
    e_values = [e for e in e_values if e >= 0]

    # 输出日志
    print(f"文件 {file_name} 解析出 {len(e_values)} 个E值: {e_values[:10] if e_values else '无'}")
    return e_values, np.mean(e_values) if e_values else 0.0


def log_density(record):
    """计算E值的对数密度（带噪声处理避免奇异矩阵）"""
    if not record:
        return 0, 0, None
    record = np.array(record)
    # 处理全重复数据
    if np.allclose(record, record[0], atol=1e-9):
        jitter = np.random.normal(0, abs(record[0]) * 1e-10, size=len(record))
        record = record + jitter
    # 对数转换
    record = np.log10(np.maximum(record, 1e-300))
    # 核密度估计
    try:
        density = gaussian_kde(record)
        density.covariance_factor = lambda: 0.15
        density._compute_covariance()
        return min(record), max(record), density
    except np.linalg.LinAlgError:
        print("KDE计算失败，返回空值")
        return min(record), max(record), None


def set_default():
    """设置绘图样式（边框、字体等）"""
    fig, ax = plt.subplots()
    # 显示四周边框并设置样式
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)
    # 坐标轴字体
    plt.tick_params(labelsize=12, width=1, length=5)
    plt.grid(False)  # 移除网格
    return fig, ax


def plot_bar(global_min, global_max, color, save_path, **datasets):
    """绘制多数据集E值分布直方图"""
    fig, ax = set_default()
    # 动态调整bins
    bins = np.linspace(global_min, global_max, 50)  # 更密集的bins
    for i, (name, record) in enumerate(datasets.items()):
        if not record:
            continue  # 跳过空数据集
        blast_log = np.log10(np.maximum(record, 1e-300))
        plt.hist(blast_log, alpha=0.7, color=color[i], density=True,
                 label=name, bins=bins, edgecolor='white')  # 增加白色边框区分柱子

    plt.xlabel('$\\mathregular{log_{10}(e-value)}$', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(loc='best', prop={'size': 12})
    # 保存图片
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, 'blastn_log10_evalue_barplot.png')
    plt.savefig(save_file, format='png', bbox_inches='tight')
    print(f"图片已保存至: {save_file}")
    plt.close(fig)


def blastn_evaluation(rand_file, nat_file,  gen_file, save_path):
    """评估BLAST结果并绘图（区分自然序列、iGEM、生成序列）"""
    # 解析所有数据集
    rand_record, _ = open_BLAST(rand_file)
    nat_record, _ = open_BLAST(nat_file)   # 自然序列（原Nat.txt）
    gen_record, _ = open_BLAST(gen_file)

    # 计算密度范围
    min_rand, max_rand, _ = log_density(rand_record)
    min_nat, max_nat, _ = log_density(nat_record)
    min_gen, max_gen, _ = log_density(gen_record)

    # 全局范围（包含所有数据集）
    global_min = min(min_rand, min_nat, min_gen)
    global_max = max(max_rand, max_nat,  max_gen)

    # 绘图（传递所有数据集）
    plot_bar(global_min, global_max,
             color=['#d95f02', '#1b9e77', '#7570b3', '#ffd700'],  # 颜色对应 Ran, Nat, iGEM, Gen
             save_path=save_path,
             Ran=rand_record,
             iGEM=nat_record,   # 自然序列独立标签
             Gen=gen_record)


if __name__ == "__main__":
    # 数据集路径（需确保文件存在）
    rand_file = "/root/autodl-tmp/sjy/dachang1/data/raw_sequences/sc_blast/sc_nan.txt"
    nat_file  = "/root/autodl-tmp/sjy/dachang1/data/raw_sequences/ecoli_blast/Nat.txt"  # 自然序列
    gen_file  = "/root/autodl-tmp/sjy/dachang1/data/raw_sequences/sc_blast/sc_gen.txt"
    save_path = "/root/autodl-tmp/sjy/dachang1/data/valid/ecoli_blast"

    os.makedirs(save_path, exist_ok=True)
    # 调用时传入自然序列文件
    blastn_evaluation(rand_file, nat_file, gen_file, save_path)