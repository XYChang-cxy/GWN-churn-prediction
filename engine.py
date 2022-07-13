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
    # supports: 存邻接矩阵（转移矩阵）的列表
    # gcn_bool: 是否包含gcn网络
    # addaptadj: 是否使用自适应邻接矩阵
    # aptinit: 自适应邻接矩阵初始化方式（随机初始化或根据转移矩阵Pf初始化）
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit):
        self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)  # 优化器
        self.loss = util.masked_mae  # 损失函数
        self.scaler = scaler
        self.clip = 5  # 用于梯度裁剪的梯度范数上限

    # 用于训练过程中执行一个batch的梯度计算和参数更新，并返回loss值和mape、rmse值
    # input: (batch_size, input_dim, num_nodes, input_length)
    # real_val: (batch_size, num_nodes, output_length)
    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()  # 将梯度初始化为0
        input = nn.functional.pad(input,(1,0,0,0))  # 对最后一个进行维度左填充（用0填充）,模型默认的感受野等于13
        output = self.model(input)
        # output = [batch_size,12,num_nodes,1]   即(batch_size,output_length,num_nodes,output_dim) output_dim = 1 *****************
        output = output.transpose(1,3)
        # output = [batch_size,1,num_nodes,12]

        # 传入的real_val是三维张量，需要在dim=1处进行维度扩充
        real = torch.unsqueeze(real_val,dim=1)  # torch.unsqueeze: 对数据维度进行扩充,给指定位置加上维数为1的维度
        # real: (batch_size, 1, num_nodes, output_length)

        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real, 0.0)
        loss.backward()  # 根据loss来计算网络参数的梯度
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)  # 进行梯度裁剪，防止梯度爆炸
        self.optimizer.step()  # 优化器对所有参数进行更新
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        # output = [batch_size,12,num_nodes,1]
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse
