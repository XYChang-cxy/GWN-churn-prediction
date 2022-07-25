import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from torch import nn
from scipy.sparse import linalg
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score


class DataLoader(object):
    def __init__(self, xs, ys, adjs, mask, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0  # 当前batch的索引index
        if pad_with_last_sample:  # 用最后一个样本填充，使样本可以被batch_size整除
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            adjs_padding = np.repeat(adjs[-1:], num_padding, axis=0)
            mask_padding = np.repeat(mask[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            adjs = np.concatenate([adjs, adjs_padding],axis=0)
            mask = np.concatenate([mask, mask_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.adjs = adjs
        self.mask = mask

    def shuffle(self):  # 打乱样本
        permutation = np.random.permutation(self.size)  # 打乱下标
        xs, ys, adjs, mask = self.xs[permutation], self.ys[permutation], self.adjs[permutation], self.mask[permutation]
        self.xs = xs
        self.ys = ys
        self.adjs = adjs
        self.mask = mask

    def get_iterator(self):  # 返回迭代器，每调用一次迭代器返回新的一个batch的数据
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                adj_i = self.adjs[start_ind: end_ind, ...]
                mask_i = self.mask[start_ind: end_ind, ...]
                yield (x_i, y_i, adj_i, mask_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean



def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1), dtype=np.float32).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.  # numpy 除法，分母为0时自动处理为inf
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename, adjtype):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    if adjtype == "scalap":  # ??????????????
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":  # 标准化的拉普拉斯矩阵 L = I - D^-1/2 A D^-1/2
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":  # 对称的标准化的邻接矩阵
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":  # 无向图：转移矩阵P
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":  # 有向图：前向转移矩阵Pf和后向转移矩阵Pb
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":  # 单位矩阵，无路网信息
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj


# def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
#     data = {}
#     for category in ['train', 'val', 'test']:
#         cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
#         data['x_' + category] = cat_data['x']
#         data['y_' + category] = cat_data['y']
#     scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
#     # Data format
#     for category in ['train', 'val', 'test']:
#         data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
#     data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
#     data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
#     data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
#     data['scaler'] = scaler
#     return data

# y_mode: 'activity'/'churn'
def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None, y_mode='activity'):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_activity_' + category] = cat_data['y_activity']
        data['y_churn_' + category] = cat_data['y_churn']
        data['p_'+category] = cat_data['p']  #  转移矩阵
        data['y_activity_mask_' + category] = cat_data['y_activity_mask']
        data['y_churn_mask_' + category] = cat_data['y_churn_mask']
    scaler_list = []
    for i in range(data['x_train'].shape[-1]):
        scaler = StandardScaler(mean=data['x_train'][..., i].mean(), std=data['x_train'][..., i].std())
        scaler_list.append(scaler)
    activity_array = 2 * data['x_train'][...,0] + data['x_train'][...,1] + 3 * data['x_train'][...,2] + \
                     5 * data['x_train'][...,3] + 4 * data['x_train'][...,4]
    activity_scaler = StandardScaler(mean=activity_array.mean(), std=activity_array.std())

    # Data format
    for category in ['train', 'val', 'test']:
        for i in range(data['x_train'].shape[-1]):
            data['x_' + category][..., i] = scaler_list[i].transform(data['x_' + category][..., i])
    data['scaler_list'] = scaler_list
    if y_mode == 'activity':
        data['train_loader'] = DataLoader(data['x_train'], data['y_activity_train'],data['p_train'],data['y_activity_mask_train'], batch_size)
        data['val_loader'] = DataLoader(data['x_val'], data['y_activity_val'], data['p_val'],data['y_activity_mask_val'], valid_batch_size)
        data['test_loader'] = DataLoader(data['x_test'], data['y_activity_test'], data['p_test'],data['y_activity_mask_test'], test_batch_size)
        data['y_scaler'] = activity_scaler
    else: # y_mode == 'churn'
        data['train_loader'] = DataLoader(data['x_train'], data['y_churn_train'], data['p_train'],data['y_churn_mask_train'], batch_size)
        data['val_loader'] = DataLoader(data['x_val'], data['y_churn_val'], data['p_val'], data['y_churn_mask_val'], valid_batch_size)
        data['test_loader'] = DataLoader(data['x_test'], data['y_churn_test'], data['p_test'], data['y_churn_mask_test'], test_batch_size)
        data['y_scaler'] = None
    return data

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    # 此时 torch.mean是由0和1组成的矩阵
    mask /= torch.mean((mask))
    # 此时 torch.mean(mask) = 1.0
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)  # 将mask中的nan替换为0
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse


# 二分类交叉熵损失函数
def masked_bce(preds, labels, mask=None, is_masked=True):
    loss = nn.BCELoss(reduction='mean')
    if not is_masked:
        return loss(preds, labels)
    else:
        if mask is None:
            print('ERROR: \'masks\' must be given if \'is_masked=True\'!')
            return
        mask = (mask > 0.0)
        mask = mask.float()
        masked_preds = preds * mask
        masked_labels = labels * mask
        masked_preds = torch.where(torch.isnan(masked_preds), torch.zeros_like(masked_preds), masked_preds)
        masked_labels = torch.where(torch.isnan(masked_labels), torch.zeros_like(masked_labels), masked_labels)
        return loss(masked_preds, masked_labels) / torch.mean(mask)


def masked_acc(preds, labels, threshold=0.5, mask=None, is_masked=True):
    preds = preds >= threshold
    preds = preds.float()
    if not is_masked:
        preds = torch.squeeze(preds).cpu().numpy()
        labels = torch.squeeze(labels).cpu().numpy()
        sum = 0.0
        for i in range(preds.shape[0]):
            sum += accuracy_score(labels[i],preds[i])
        return sum / preds.shape[0]
    else:
        if mask is None:
            print('ERROR: \'masks\' must be given if \'is_masked=True\'!')
            return
        mask = (mask > 0.0)
        mask = mask.float()
        masked_preds = preds * mask
        masked_labels = labels * mask
        masked_preds = torch.squeeze(masked_preds).cpu().numpy()
        masked_labels = torch.squeeze(masked_labels).cpu().numpy()
        # sum0 = 0.0
        sum = 0.0
        valid_index = int(masked_labels.shape[1] * torch.mean(mask))
        for i in range(masked_preds.shape[0]):
            # sum0 += accuracy_score(masked_labels[i], masked_preds[i])
            sum += accuracy_score(masked_labels[i][:valid_index], masked_preds[i][:valid_index])
        return sum / masked_preds.shape[0]


def masked_precision(preds, labels, threshold=0.5, mask=None, is_masked=True):
    preds = preds >= threshold
    preds = preds.float()
    if not is_masked:
        preds = torch.squeeze(preds).cpu().numpy()
        labels = torch.squeeze(labels).cpu().numpy()
        sum = 0.0
        for i in range(preds.shape[0]):
            sum += precision_score(labels[i], preds[i], zero_division=0)
        return sum / preds.shape[0]
    else:
        if mask is None:
            print('ERROR: \'masks\' must be given if \'is_masked=True\'!')
            return
        mask = (mask > 0.0)
        mask = mask.float()
        masked_preds = preds * mask
        masked_labels = labels * mask
        masked_preds = torch.squeeze(masked_preds).cpu().numpy()
        masked_labels = torch.squeeze(masked_labels).cpu().numpy()
        sum = 0.0
        valid_index = int(masked_labels.shape[1] * torch.mean(mask))
        for i in range(masked_preds.shape[0]):
            sum += precision_score(masked_labels[i][:valid_index], masked_preds[i][:valid_index], zero_division=0)
        return sum / masked_preds.shape[0]


def masked_recall(preds, labels, threshold=0.5, mask=None, is_masked=True):
    preds = preds >= threshold
    preds = preds.float()
    if not is_masked:
        preds = torch.squeeze(preds).cpu().numpy()
        labels = torch.squeeze(labels).cpu().numpy()
        sum = 0.0
        for i in range(preds.shape[0]):
            sum += recall_score(labels[i], preds[i], zero_division=0)
        return sum / preds.shape[0]
    else:
        if mask is None:
            print('ERROR: \'masks\' must be given if \'is_masked=True\'!')
            return
        mask = (mask > 0.0)
        mask = mask.float()
        masked_preds = preds * mask
        masked_labels = labels * mask
        masked_preds = torch.squeeze(masked_preds).cpu().numpy()
        masked_labels = torch.squeeze(masked_labels).cpu().numpy()
        sum = 0.0
        valid_index = int(masked_labels.shape[1] * torch.mean(mask))
        for i in range(masked_preds.shape[0]):
            sum += recall_score(masked_labels[i][:valid_index], masked_preds[i][:valid_index], zero_division=0)
        return sum / masked_preds.shape[0]


def masked_f1(preds, labels, threshold=0.5, mask=None, is_masked=True):
    preds = preds >= threshold
    preds = preds.float()
    if not is_masked:
        preds = torch.squeeze(preds).cpu().numpy()
        labels = torch.squeeze(labels).cpu().numpy()
        sum = 0.0
        for i in range(preds.shape[0]):
            sum += f1_score(labels[i], preds[i], zero_division=0)
        return sum / preds.shape[0]
    else:
        if mask is None:
            print('ERROR: \'masks\' must be given if \'is_masked=True\'!')
            return
        mask = (mask > 0.0)
        mask = mask.float()
        masked_preds = preds * mask
        masked_labels = labels * mask
        masked_preds = torch.squeeze(masked_preds).cpu().numpy()
        masked_labels = torch.squeeze(masked_labels).cpu().numpy()
        sum = 0.0
        valid_index = int(masked_labels.shape[1] * torch.mean(mask))
        for i in range(masked_preds.shape[0]):
            sum += f1_score(masked_labels[i][:valid_index], masked_preds[i][:valid_index], zero_division=0)
        return sum / masked_preds.shape[0]


def masked_auc(preds, labels, threshold=0.5, mask=None, is_masked=True):
    preds = preds >= threshold
    preds = preds.float()
    if not is_masked:
        preds = torch.squeeze(preds).cpu().numpy()
        labels = torch.squeeze(labels).cpu().numpy()
        sum = 0.0
        for i in range(preds.shape[0]):
            sum += roc_auc_score(labels[i], preds[i])
        return sum / preds.shape[0]
    else:
        if mask is None:
            print('ERROR: \'masks\' must be given if \'is_masked=True\'!')
            return
        mask = (mask > 0.0)
        mask = mask.float()
        masked_preds = preds * mask
        masked_labels = labels * mask
        masked_preds = torch.squeeze(masked_preds).cpu().numpy()
        masked_labels = torch.squeeze(masked_labels).cpu().numpy()
        sum = 0.0
        valid_index = int(masked_labels.shape[1] * torch.mean(mask))
        for i in range(masked_preds.shape[0]):
            sum += roc_auc_score(masked_labels[i][:valid_index], masked_preds[i][:valid_index])
        return sum / masked_preds.shape[0]


def binary_metric(pred, real, threshold=0.5, mask=None, is_masked=True):
    acc = masked_acc(pred,real,threshold,mask,is_masked)
    precision = masked_precision(pred,real,threshold,mask,is_masked)
    recall = masked_recall(pred,real,threshold,mask,is_masked)
    f1 = masked_f1(pred,real,threshold,mask,is_masked)
    auc = masked_auc(pred,real,threshold,mask,is_masked)
    return acc, precision, recall, f1, auc
