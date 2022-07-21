import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')  # cuda:3时报错，只有一个GPU
parser.add_argument('--data',type=str,default='train_data/REPO-8649239/input_data',help='data path')
# parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='transition',help='adj type')  # 开发者协作网络是无向图，默认一个转移矩阵
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
# parser.add_argument('--seq_length_x',type=int,default=12,help='')  # 默认以12周的数据作为输入
parser.add_argument('--seq_length_y',type=int,default=18,help='')  # 等于流失期限的周数
parser.add_argument('--in_dim',type=int,default=8,help='inputs dimension')  # 5个count、2个received count、1个month
parser.add_argument('--num_nodes',type=int,default=506,help='number of nodes')  ############################
parser.add_argument('--nhid',type=int,default=32,help='')  # 调节模型的参数
parser.add_argument('--batch_size',type=int,default=16,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=1,help='')
#parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='./garage/mindspore',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')

args = parser.parse_args()




def main():
    #set seed
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #load data
    device = torch.device(args.device)

    # util.load_dataset: 加载数据集，返回字典，包含train_loader,val_loader,test_loader（DataLoader类型）和scaler（StandardScaler类型）
    # args.data: 存储数据的文件夹，默认METR-LA；args.batch_size: 训练/验证/测试集的批处理大小，默认64
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    # scaler_list = dataloader['scaler_list']  # 根据训练集的mean和std确定的scaler，用于统一标准化和逆标准化
    activity_scaler = dataloader['activity_scaler']
    # supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)

    # if args.randomadj:  # 随机初始化自适应邻接矩阵，adjinit即gwnet的初始化参数aptinit
    #     adjinit = None
    # else:
    #     adjinit = supports[0]
    # if args.aptonly:  # 如果路网信息未知，仅使用自适应邻接矩阵
    #     supports = None

    adjinit = None
    if args.adjtype == "doubletransition":
        supports_len = 2
    else:
        supports_len = 1

    engine = trainer(activity_scaler, args.in_dim, args.seq_length_y, args.num_nodes, args.nhid, args.dropout,
                     args.learning_rate, args.weight_decay, device, supports_len, args.gcn_bool, args.addaptadj,
                     adjinit)


    print("start training...",flush=True)
    his_loss =[]  # 存储验证集的平均loss
    val_time = []  # 存储每个epoch中验证集评估的时间
    train_time = []  # 存储每个epoch中训练集训练的时间

    # 两层嵌套的循环，外层是epochs轮，每轮有若干batch
    for i in range(1,args.epochs+1):
        #if i % 10 == 0:
            #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
            #for g in engine.optimizer.param_groups:
                #g['lr'] = lr
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据下标和数据
        for iter, (x, y, p) in enumerate(dataloader['train_loader'].get_iterator()):  # (x,y) 是一个batch的数据
            # x: (batch_size, input_length, num_nodes, input_dim)
            # y: (batch_size, output_length, num_nodes, output_dim)
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            # trainx: (batch_size, input_dim, num_nodes, input_length)   即 (n,c,v,l)
            # trainy: (batch_size, output_dim, num_nodes, output_length)
            trainp = torch.Tensor(p).to(device)
            supports = [trainp]
            metrics = engine.train(trainx,supports,trainy[:,0,:,:])  # 预测时仅预测活跃度！trainy[:,0,:,:]: (batch_size, num_nodes, output_length)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []


        s1 = time.time()
        for iter, (x, y, p) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            testp = torch.Tensor(p).to(device)
            supports = [testp]
            metrics = engine.eval(testx, supports, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
        torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    #testing
    bestid = np.argmin(his_loss)  # 获取loss值最小时对应的下标
    engine.model.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))


    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]  # realy: (num_samples, num_nodes, output_length)

    for iter, (x, y, p) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        testp = torch.Tensor(p).to(device)
        supports = [testp]
        with torch.no_grad():  # 在该模块下，所有计算得出的tensor的requires_grad都自动设置为False,反向传播时就不会自动求导了，因此大大节约了显存或者说内存
            preds = engine.model(testx,supports).transpose(1,3)
        outputs.append(preds.squeeze())  # squeeze: 删除张量中维数为1的维度，preds.squeeze(): (batch_size, num_nodes, output_length)

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]  # yhat: (num_samples,num_nodes,output_length)


    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))


    amae = []
    amape = []
    armse = []
    for i in range(args.seq_length_y):  # 12: seq_length_y
        pred = activity_scaler.inverse_transform(yhat[:,:,i])
        real = realy[:,:,i]
        metrics = util.metric(pred,real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
    torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")



if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
