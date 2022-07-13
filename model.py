import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))  # 爱因斯坦求和简记法 \sum_v x_{ncvl}A_{vw} = o_{ncwl}, 即A^Tx  ??????????????????
        return x.contiguous() # torch.contiguous()方法首先拷贝了一份张量在内存中的地址，然后将地址按照形状改变后的张量的语义进行排列。

class linear(nn.Module):
    def __init__(self,c_in,c_out):  # c_in: in_channels; c_out:out_channels
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)


# 疑问：out的各项应该进行累加（沿dim=1方向），这一步如何操作的？是通过self.mlp吗？
class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        # 若x.shape=(n,c,v,l),则c_in=c
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order  # order即式子中的K

    def forward(self,x,support):  # x: 图信号
        out = [x]
        for a in support:  # out = [x,P_fx,P_f^2x,...,P_f^K,P_bx,P_b^2x,...,P_b^Kx,A_{adp}x,A_{adp}^2x,...,A_{adp}^Kx]
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)  # 沿dim=1进行拼接，若x.shape=(n,c,v,l),则h.shape=(n,(order*support_len+1)*c_in,v,l)
        h = self.mlp(h)  # 该步骤相当于为Z=\sum_{k=0}^K P_f^kXW_{k1}+P_b^kXW_{k2}+\tilde{A}^k_{adp}XW_{k3}的每项都乘以Wk,输出通道数等于c_out
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(nn.Module):
    # device: 训练设备
    # num_nodes: 节点数
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks  # 块数，一个块有多个layers
        self.layers = layers  # 层数
        self.gcn_bool = gcn_bool  # 是否使用GCN
        self.addaptadj = addaptadj  # 是否使用自适应邻接矩阵

        self.filter_convs = nn.ModuleList()  # TCN-a
        self.gate_convs = nn.ModuleList()  # TCN-b
        self.residual_convs = nn.ModuleList()  # residual connections
        self.skip_convs = nn.ModuleList()  # skip connections
        self.bn = nn.ModuleList()  # BatchNorm2d,在卷积层之后添加BatchNorm2d进行数据归一化处理，使数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        self.gconv = nn.ModuleList()  # GCN

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))  # 初始Linear层
        self.supports = supports

        receptive_field = 1  # 感受野

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                # 论文中的E1和E2，用于自适应邻接矩阵的初始化
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)  # source node embedding
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)  # target node embedding
                self.supports_len +=1  # 使用自适应邻接矩阵，转移矩阵数+1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))######################################
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1




        for b in range(blocks):#######################################
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope  ############################不同block的感受野如何增长？？？
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field  # 感受野


    # input.shape = (n,c,v,l),n是批处理数量,c是in_channels,v是节点数,l是输入数据的时间长度##########################
    def forward(self, input):
        in_len = input.size(3)
        if in_len<self.receptive_field:  # 输入的长度小于感受野时，对其进行左填充
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            # 扩张卷积，每个block的不同layers的扩张因子dilation不同，且以指数增长；感受野也随着层数和块数增加而不断扩大
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]  #############################？？？？？？
            except:
                skip = 0
            skip = s + skip


            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]  ####################################？？？？？


            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x





