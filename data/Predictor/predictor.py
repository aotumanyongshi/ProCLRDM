import os
import csv
from data.Predictor.cnn_k15 import CNN_K15_language

# 步骤1: 初始化模型
sequence_length = 165  # 确保与训练时一致
predictor = CNN_K15_language(length=sequence_length)

# 步骤2: 指定训练好的模型路径
model_path = '/root/autodl-tmp/sjy/dachang1/data/training/ecilo_trained_predictorcnn_k15/checkpoint.pth'

# 步骤3: 指定待预测的DNA序列文件路径
data_path = '/root/autodl-tmp/sjy/dachang1/data/raw_sequences/vae+con+diffusion/gen_sequences_165bp.txt'


predictor.predict(model_path=model_path, data_path=data_path)

# 步骤5: 读取生成的两个文件
model_dir = os.path.dirname(model_path)
seq_file = os.path.join(model_dir, 'seqs.txt')
pred_file = os.path.join(model_dir, 'nature_preds.txt')

# 读取DNA序列
with open(seq_file, 'r') as f:
    sequences = [line.strip() for line in f if line.strip()]

# 读取表达值
with open(pred_file, 'r') as f:
    predictions = [line.strip() for line in f if line.strip()]

# 步骤6: 将结果合并保存到CSV文件
csv_file = os.path.join(model_dir, 'dna_predictions.csv')

with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # 写入表头
    writer.writerow(['index', 'dna_sequence', 'expression_value'])

    # 写入数据行（确保序列和预测值数量一致）
    min_length = min(len(sequences), len(predictions))
    for i in range(min_length):
        writer.writerow([i, sequences[i], predictions[i]])

print(f"预测的DNA序列和表达值已保存到: {csv_file}")
