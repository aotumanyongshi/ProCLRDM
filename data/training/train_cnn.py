import os
from data.Predictor.cnn_k15 import CNN_K15_language  # 确保正确导入你的类和函数

# --- 数据和模型配置 ---
sequence_file = '/root/autodl-tmp/sjy/dachang1/data/raw_sequences/ecoli_data/ecoli_data.txt'  # 替换为你的序列文件路径
expression_file = '/root/autodl-tmp/sjy/dachang1/data/raw_sequences/ecoli_data/exp.txt'  # 替换为你的表达值文件路径
output_dir = 'ecilo_trained_predictor'  # 模型和日志保存的目录
sequence_length = 165  # 根据你的数据调整序列长度
batch_size = 64
epochs = 200
patience = 50
log_steps = 10
save_steps = 20
exp_mode = "direct"

# --- 创建输出目录 ---
os.makedirs(output_dir, exist_ok=True)

# --- 初始化模型训练器 ---
trainer = CNN_K15_language(
    length=sequence_length,
    batch_size=batch_size,
    epoch=epochs,
    patience=patience,
    log_steps=log_steps,
    save_steps=save_steps,
    exp_mode=exp_mode
)

# --- 开始训练 ---
print("开始训练预测器模型...")
trainer.train(dataset=sequence_file, labels=expression_file, savepath=output_dir)

print("预测器模型训练完成！模型和日志保存在:", output_dir)