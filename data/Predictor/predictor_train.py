# predictor_train.py

import glob
import os
import time
from torch.nn.functional import pad
from torch.utils.data import TensorDataset, DataLoader
import re
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from data.Predictor.pytorchtools import EarlyStopping
from data.Predictor.predictor_models import *


torch.nn.utils.clip_grad_norm_


class PREDICT():
    def __init__(self,run_name='predictor_',dataset='SC_short', model_name = 'LSTMModel',
                 train_data_path = '/root/autodl-tmp/sjy/dachang1/data/raw_sequences/ecoli_data/pred_train_exp_short.csv',
                 test_data_path = '/root/autodl-tmp/sjy/dachang1/data/raw_sequences/ecoli_data/pred_test_exp_short.csv'):
        self.model_name = model_name
        self.patience = 20
        self.val_acc_list = []
        self.save_path = '/root/autodl-tmp/sjy/dachang1/data/Predictor/Predictor/results/model_ecilo_short/'
        self.dataset = dataset
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.seq1, self.exp = self.data_load(self.train_data_path)
        self.seq = self.seq_onehot(self.seq1)
        input_size = self.seq.shape[-1]
        self.input_size = input_size
        self.batch_size = 256
        # self.batch_size = 16
        self.hidden_size = 256
        self.conv_hidden = 128
         # self.seq_len = 156
        self.output_size = 1
        self.lambda_l2 = 0.001
        self.dropout_rate = 0.2
        self.r = 161273
        # self.r = 12575
        torch.cuda.set_device(1)
        self.use_gpu = True if torch.cuda.is_available() else False
        self.build_model()
        self.checkpoint_dir = './checkpoint/' + run_name + '/'
        if not os.path.exists(self.checkpoint_dir): os.makedirs(self.checkpoint_dir)


    def data_load(self, data_path):
        data = open(data_path, 'r')
        next(data)
        seq = []
        exp = []
        for item in data:
            item = item.split(",")
            seq.append(item[0])
            exp.append(item[1])
        data.close()

        # Convert the expression data into array format
        expression = np.zeros((len(exp), 1))
        for i in range(len(exp)):
            expression[i] = float(exp[i])

        return seq, expression

    def data_load1(self, data_path):
        data = open(data_path, 'r')
        next(data)
        seq = []
        exp = []
        for item in data:
            item = item.split(",")
            seq.append(item[0])
            exp.append(item[1])
        data.close()

        expression = np.zeros((len(exp), 1))
        for i in range(len(exp)):
            expression[i] = float(exp[i])

        return seq, expression

    def string_to_array(self, my_string):
        my_string = my_string.lower()
        my_string = re.sub('[^acgt]', 'z', my_string)
        my_array = np.array(list(my_string))
        return my_array

    def one_hot_encode(self, my_array):
        label_encoder = LabelEncoder()
        label_encoder.fit(np.array(['a', 'c', 'g', 't', 'z']))
        integer_encoded = label_encoder.transform(my_array)
        onehot_encoder = OneHotEncoder(dtype=int)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        # 将稀疏矩阵转换为密集数组
        onehot_encoded = onehot_encoded.toarray()

        #print("onehot_encoded shape:", onehot_encoded.shape)
        #print("onehot_encoded content:", onehot_encoded)

        if onehot_encoded.ndim >= 2:
            onehot_encoded = np.delete(onehot_encoded, -1, 1)
        else:
            onehot_encoded = onehot_encoded.reshape(1, -1)
        return onehot_encoded

    def seq_onehot(self, seq):
        onehot_seq = [torch.tensor(self.one_hot_encode(self.string_to_array(s)), dtype=torch.float32) for s in seq]

        # Determine the maximum length
        max_length = max(matrix.shape[0] for matrix in onehot_seq)

        padded_tensor_list = []
        for matrix in onehot_seq:
            padding_length = max_length - matrix.shape[0]
            padded_tensor = pad(matrix, (0, 0, 0, padding_length), value=0)
            padded_tensor_list.append(padded_tensor)

        onehot_seq = torch.stack(padded_tensor_list, dim=0)

        return onehot_seq

    def build_model(self):
        if self.model_name == 'OnlyCNNModel':
            self.model = OnlyCNNModel(self.input_size, self.hidden_size, self.output_size, self.dropout_rate, self.lambda_l2)
        elif self.model_name == 'LSTMModel':
            self.model = LSTMModel(self.input_size, self.hidden_size, self.output_size, self.dropout_rate,
                                      self.lambda_l2)

        if self.use_gpu:
            self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-3,weight_decay=self.lambda_l2)
        self.criterion = nn.MSELoss()

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), self.checkpoint_dir + "model_weights_{}.pth".format(epoch))

    def load_model(self):
        '''
            Load model parameters from most recent epoch
        '''
        list_model = glob.glob(self.checkpoint_dir + "model*.pth")
        if len(list_model) == 0:
            print("[*] Checkpoint not found! Starting from scratch.")
            return 1 #file is not there
        chk_file = max(list_model, key=os.path.getctime)
        epoch_found = int( (chk_file.split('_')[-1]).split('.')[0])
        print("[*] Checkpoint {} found!".format(epoch_found))
        self.model.load_state_dict(torch.load(chk_file))
        return epoch_found

    def evaluate(self):
        """
        评估模型性能：计算皮尔逊相关系数(rho)、斯皮尔曼相关系数(cor)、均方误差(mse)
        逻辑：从训练数据中拆分验证集（或加载测试集），用模型预测后计算指标
        """
        # -------------------------- 1. 准备评估数据（这里用训练集拆分出验证集，也可加载测试集）
        # 方案1：从已有训练数据中拆分 20% 作为验证集（推荐，无需额外加载数据）
        total_size = len(self.seq)
        val_size = int(total_size * 0.2)  # 20% 作为验证集
        # 拆分验证集特征和标签（用索引切片，避免数据打乱）
        val_feature = self.seq[total_size - val_size:]  # 后20%作为验证集
        val_label = self.exp[total_size - val_size:]
        # 转为TensorDataset和DataLoader（与训练集一致的batch_size）
        val_data = TensorDataset(val_feature, torch.Tensor(val_label))
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)  # 验证集不打乱

        # 方案2：加载测试集作为验证集（若想直接用测试集评估，取消下面注释并注释方案1）
        # val_seq1, val_exp = self.data_load1(self.test_data_path)  # 用已有的data_load1方法加载测试集
        # val_seq = self.seq_onehot(val_seq1)  # 测试集序列one-hot编码
        # val_data = TensorDataset(val_seq, torch.Tensor(val_exp))
        # val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)

        # -------------------------- 2. 模型评估模式（关闭 dropout 等训练特有的层）
        self.model.eval()
        all_preds = []  # 存储所有预测值
        all_labels = []  # 存储所有真实标签

        # -------------------------- 3. 无梯度预测（避免计算梯度消耗资源）
        with torch.no_grad():
            for batch_feature, batch_label in val_loader:
                # 数据送GPU（与训练一致）
                if self.use_gpu:
                    batch_feature = batch_feature.cuda()
                    batch_label = batch_label.cuda()

                # 模型预测
                output = self.model(batch_feature)

                # 收集预测值和真实值（转CPU后存列表，避免GPU内存占用）
                all_preds.extend(output.cpu().numpy().flatten())  # 展平为1维列表
                all_labels.extend(batch_label.cpu().numpy().flatten())

        # -------------------------- 4. 计算评估指标
        # 转为numpy数组（方便计算统计指标）
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # 1. 皮尔逊相关系数（rho）：衡量线性相关程度（-1~1，越接近1越好）
        rho, _ = pearsonr(all_labels, all_preds)  # _ 是p值，不需要则忽略
        # 2. 斯皮尔曼相关系数（cor）：衡量单调相关程度（-1~1，对异常值更稳健）
        cor, _ = spearmanr(all_labels, all_preds)
        # 3. 均方误差（mse）：衡量预测误差（值越小越好）
        mse = mean_squared_error(all_labels, all_preds)

        # -------------------------- 5. 恢复模型训练模式（评估后回到训练模式）
        self.model.train()

        # 返回与train方法中匹配的三个指标
        return rho, cor, mse

    def train(self):

        # Split training/validation and testing set
        expression = self.exp
        onehot_seq = self.seq

        seq = onehot_seq
        r = self.r
        train_feature = seq[0:r]
        train_label = expression[0:r]

        train_feature = torch.Tensor(train_feature)
        train_label = torch.Tensor(train_label)


        train_data = TensorDataset(train_feature, train_label)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)


        # train model
        num_epochs = 100
        loss_total = 0.0
        best_loss = float('inf')

        clip_value = 1.0
        early_stopping = EarlyStopping(patience=self.patience, verbose=True,
                                       path=self.save_path + self.model_name + '_' + self.dataset + '.pth', stop_order='max')

        weighted_loss_function = WeightedMSELoss(weight=torch.Tensor([10.0]).cuda())

        with open('/root/autodl-tmp/sjy/dachang1/data/Predictor/Predictor/results/onlyGRUaccuracy.txt', 'w') as f:
            for epoch in range(num_epochs):
                # self.model.train()
                epoch_loss = 0.0
                for batch_idx, (batch_feature, batch_label) in enumerate(train_loader):
                    batch_feature = batch_feature.cuda()
                    batch_label = batch_label.cuda()

                    self.optimizer.zero_grad()

                    with torch.cuda.amp.autocast():
                        output = self.model(batch_feature)
                        # loss = self.criterion(output, batch_label)
                        loss = weighted_loss_function(output, batch_label) # Use a weighted loss function

                    loss.backward()

                    nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)

                    self.optimizer.step()

                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(train_loader)
                print("Epoch:", epoch, "Loss:", avg_loss)
                if(avg_loss < best_loss):
                    best_loss = avg_loss
                    self.save_model(epoch)
                loss_total += avg_loss


                rho, cor, mse = self.evaluate()
                # f.write(f"Epoch {epoch}: Accuracy {rho}\n")
                f.write(f"rho:{rho},cor:{cor},mse:{mse}\n")
                early_stopping(val_loss=rho, model=self.model)
                if early_stopping.early_stop:
                    print('Early Stopping......')
                    break

    # Predict the entire dataset file
    def valdata1(self):
        self.model.load_state_dict(torch.load("/root/autodl-tmp/sjy/dachang1/data/Predictor/Predictor/results/model_ecilo_short/LSTMModel_SC_short.pth"))

        with open("/root/autodl-tmp/sjy/dachang1/data/raw_sequences/vae+con+diffusion/select_gen.txt", 'r') as f:
            next(f)
            inputseq = [line.strip() for line in f]


        valseq_onehot = self.seq_onehot(inputseq)
        valseq = torch.Tensor(valseq_onehot).cuda()

        with torch.no_grad():
            val_output = self.model(valseq)

        val_output = val_output.cpu()
        val_pred = val_output.numpy()

        # Write the results to a file
        with open("/root/autodl-tmp/sjy/dachang1/data/Predictor/Predictor/results/gen_pre.csv", "w") as f:
            for i, pred in enumerate(val_pred):
                f.write(inputseq[i] + "," + str(pred[0]) + "\n")
        return val_pred

if __name__ == '__main__':
    time_start = time.time()  # time record

    predict = PREDICT()
    #predict.train()
    predict.valdata1()

    time_end = time.time()
    time_sum = time_end - time_start
    print(time_sum)