import torch.optim as optim
from model import *
import util
class trainer():
    # scaler: 用于数据标准化
    # in_dim: 输入的数据维度，默认是2，即包含每个监测器的速度和time_in_day
    # seq_length: 输出的序列长度，默认12
    # num_nodes: 速度监测器数量
    # nhid: ***********************************************
    # dropout: dropout率
    # lrate: 学习率
    # wdecay: weight_decay L2范数正则化又叫权重衰减。权重衰减通过惩罚绝对值较大的模型参数为需要学习的模型增加了限制，这可能对过拟合有效
    # device: 设备
    # supports_len: 存邻接矩阵（转移矩阵）的列表长度
    # gcn_bool: 是否包含gcn网络
    # addaptadj: 是否使用自适应邻接矩阵
    # aptinit: 自适应邻接矩阵初始化方式（随机初始化或根据转移矩阵Pf初始化）
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports_len,
                 gcn_bool, addaptadj, aptinit, predict_type, binary_threshold):
        self.model = gwnet(device, num_nodes, dropout, supports_len=supports_len, gcn_bool=gcn_bool,
                           addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim, out_dim=seq_length,
                           residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8,
                           end_channels=nhid * 16, predict_type=predict_type)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)  # 优化器
        if predict_type == 'activity':
            self.loss = util.masked_mae  # 损失函数
        else:
            self.loss = util.masked_bce  # 二分类交叉熵损失函数
        self.binary_threshold = binary_threshold
        self.scaler = scaler
        self.predict_type = predict_type
        self.clip = 5  # 用于梯度裁剪的梯度范数上限

    # 用于训练过程中执行一个batch的梯度计算和参数更新，并返回loss值和mape、rmse值
    # input: (batch_size, input_dim, num_nodes, input_length)
    # real_val: (batch_size, num_nodes, output_length)
    def train(self, input, supports, real_val, mask=None):
        if self.predict_type=='churn' and mask is None:
            print('ERROR:\'mask\' cannot be None when predict_type=\'churn\'!')
            return
        self.model.train()
        self.optimizer.zero_grad()  # 将梯度初始化为0
        input = nn.functional.pad(input,(1,0,0,0))  # 对最后一个进行维度左填充（用0填充）,模型默认的感受野等于13
        output = self.model(input,supports)
        # output = [batch_size,18,num_nodes,1]   即(batch_size,output_length,num_nodes,output_dim) output_dim = 1 *****************
        output = output.transpose(1,3)
        # output = [batch_size,1,num_nodes,18]

        # 传入的real_val是三维张量，需要在dim=1处进行维度扩充
        real = torch.unsqueeze(real_val,dim=1)  # torch.unsqueeze: 对数据维度进行扩充,给指定位置加上维数为1的维度
        # real: (batch_size, 1, num_nodes, output_length)

        if self.predict_type == 'activity':
            predict = self.scaler.inverse_transform(output)
            loss = self.loss(predict, real, 0.0)
        else:
            predict = output
            loss = self.loss(predict, real, mask, True)

        loss.backward()  # 根据loss来计算网络参数的梯度
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)  # 进行梯度裁剪，防止梯度爆炸
        self.optimizer.step()  # 优化器对所有参数进行更新
        if self.predict_type == 'activity':
            mape = util.masked_mape(predict,real,0.0).item()
            rmse = util.masked_rmse(predict,real,0.0).item()
            return loss.item(),mape,rmse
        else:
            acc = util.masked_acc(predict,real,mask=mask,threshold=self.binary_threshold)
            precision = util.masked_precision(predict,real,mask=mask,threshold=self.binary_threshold)
            recall = util.masked_recall(predict,real,mask=mask,threshold=self.binary_threshold)
            return loss.item(),acc,precision,recall

    def eval(self, input,supports, real_val, mask=None):
        if self.predict_type=='churn' and mask is None:
            print('ERROR:\'mask\' cannot be None when predict_type=\'churn\'!')
            return
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input,supports)
        # output = [batch_size,12,num_nodes,1]
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        if self.predict_type == 'activity':
            predict = self.scaler.inverse_transform(output)
            loss = self.loss(predict, real, 0.0)
            mape = util.masked_mape(predict, real, 0.0).item()
            rmse = util.masked_rmse(predict, real, 0.0).item()
            return loss.item(), mape, rmse
        else:
            predict = output
            loss = self.loss(predict, real, mask, True)
            acc = util.masked_acc(predict, real, mask=mask,threshold=self.binary_threshold)
            precision = util.masked_precision(predict, real, mask=mask,threshold=self.binary_threshold)
            recall = util.masked_recall(predict, real, mask=mask,threshold=self.binary_threshold)
            return loss.item(), acc, precision, recall
