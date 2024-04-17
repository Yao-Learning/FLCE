import os
import pandas as pd
import pickle
import json
import random
import numpy as np
import math
import fmodule
import itertools
import copy
from scipy.special import comb
from itertools import chain, combinations
import time
from datetime import datetime

import torch
import torch.nn as nn

try:
    import moxing as mox
    open = mox.file.File
except Exception:
    pass


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_file(path):
    da_df = pd.read_csv(
        path, index_col=False, header=None
    )
    return da_df


def save_data(da_df, path):
    da_df.to_csv(path)
    print("File saved in {}.".format(path))


def load_pickle(fpath):
    with open(fpath, "rb") as fr:
        data = pickle.load(fr)
    return data


def save_pickle(data, fpath):
    with open(fpath, "wb") as fw:
        pickle.dump(data, fw)
    return data


def load_json(fpath):
    with open(fpath, "r") as fr:
        data = json.load(fr)
    return data


def save_json(data, fpath):
    with open(fpath, "w") as fr:
        data = json.dump(data, fr)
    return data


def append_to_logs(fpath, logs):
    with open(fpath, "a", encoding="utf-8") as fa:
        for log in logs:
            fa.write("{}\n".format(log))
        fa.write("\n")


def format_logs(logs):
    def formal_str(x):
        if isinstance(x, int):
            return str(x)
        elif isinstance(x, float):
            return "{:.5f}".format(x)
        else:
            return str(x)

    logs_str = []
    for key, elems in logs.items():
        log_str = "[{}]: ".format(key)
        log_str += " ".join([formal_str(e) for e in elems])
        logs_str.append(log_str)
    return logs_str


def listfiles(fdir):
    for root, dirs, files in os.walk(fdir):
        print(root, dirs, files)


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


def prediction_mask(logits, label):
    pred = torch.argmax(logits, dim=1)
    mask = (pred == label).float()
    return mask


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(
            m.weight, mode='fan_out', nonlinearity='relu'
        )
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        try:
            nn.init.constant_(m.bias, 0)
        except Exception:
            pass

# 定义矩阵分解函数
def Matrix_decomposition(R, P, Q, N, M, K, alpha=0.0002, beta=0.02):
    Q = Q.T  # Q 矩阵转置
    loss_list = []  # 存储每次迭代计算的 loss 值
    for step in range(5000):
        # 更新 R^
        for i in range(N):
            for j in range(M):
                if R[i][j] != 0:
                    # 计算损失函数
                    error = R[i][j]
                    for k in range(K):
                        error -= P[i][k] * Q[k][j]
                    # 优化 P,Q 矩阵的元素
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * error * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * error * P[i][k] - beta * Q[k][j])

        loss = 0.0
        # 计算每一次迭代后的 loss 大小，就是原来 R 矩阵里面每个非缺失值跟预测值的平方损失
        for i in range(N):
            for j in range(M):
                if R[i][j] != 0:
                    # 计算 loss 公式加号的左边
                    data = 0
                    for k in range(K):
                        data = data + P[i][k] * Q[k][j]
                    loss = loss + math.pow(R[i][j] - data, 2)
                    # 得到完整 loss 值
                    for k in range(K):
                        loss = loss + beta / 2 * (P[i][k] * P[i][k] + Q[k][j] * Q[k][j])
                    loss_list.append(loss)
        # plt.scatter(step, loss)
        # 输出 loss 值
        if (step + 1) % 1000 == 0:
            print("loss={:}".format(loss))
        # 判断
        if loss < 0.001:
            print(loss)
            break
    # plt.show()
    return P, Q

#通过上一轮全局模型计算本地客户端原型
def preprototype(self, model, train_loader, test_loader):
    classList = [[] for _ in range(self.args.n_classes)]  # 记录当前本地客户端中每个类中预测成功的特征
    class_dataNums = [0 for _ in range(self.args.n_classes)]  # 统计当前本地客户端每个类样本数量

    if self.args.local_steps is not None:
        n_total_bs = self.args.local_steps
    elif self.args.local_epochs is not None:
        n_total_bs = max(
            int(self.args.local_epochs * len(train_loader)), 5
        )
    else:
        raise ValueError(
            "local_steps and local_epochs must not be None together"
        )

    loader_iter = iter(train_loader)

    for t in range(n_total_bs + 1):
        if t in [0, n_total_bs]:
            # per_acc = self.test(
            #     model=model,
            #     loader=test_loader,
            # )
            per_acc = 0

        if t >= n_total_bs:
            break

        try:
            batch_x, batch_y = next(loader_iter)
        except Exception:
            loader_iter = iter(train_loader)
            batch_x, batch_y = next(loader_iter)

        if self.args.cuda:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

        hs, logits = model(batch_x)

        _, prediction = torch.max(logits, 1)

        # 每个类预测成功的特征
        # for m in range(batch_x.shape[0]):
        #     pred = prediction[m]
        #     label = batch_y[m]
        #     # 如果预测成功就把特征数据hs记录到classList
        #     if pred == label:
        #         classList[label].append(hs[m])

        # 每个类预测成功的特征
        for m in range(batch_x.shape[0]):
            pred = prediction[m]
            label = batch_y[m]
            # 如果预测成功就把特征数据hs记录到classList
            if pred == label:
                if len(classList[label]) == 0:
                    classList[label] = hs[m]
                    class_dataNums[label] += 1
                else:
                    init_matrix = classList[label][0]
                    if init_matrix.shape == hs[m].shape:
                        classList[label] = torch.add(init_matrix, hs[m])
                        class_dataNums[label] += 1

    class_mean = []
    # 进行特征平均值计算
    # meanFeature, featureNumofClasses = getMeanFeature(self.args.n_classes)
    for label in range(self.args.n_classes):
        if len(classList[label]) != 0:
            # print(classList[label])
            end_matrix = torch.div(classList[label], class_dataNums[label])
            class_mean.append(end_matrix)
        else:
            class_mean.append(None)

    # for n in range(self.args.n_classes):
    #     cnt = 0
    #     if classList[n]:
    #         # print("list 中，每个种类预测成功的数量。种类：", str(n), "数量：", str(len(list[n])) )
    #         cnt = 1
    #         init_matrix = classList[n][0]
    #         # 特征张量求和并统计数量
    #         for k in range(1, len(classList[n])):
    #             if init_matrix.shape == classList[n][k].shape:
    #                 init_matrix = torch.add(init_matrix, classList[n][k])
    #                 cnt += 1
    #         # 计算平均特征
    #         end_matrix = torch.div(init_matrix, cnt)
    #         # 记录类特征
    #         class_mean.append(end_matrix)
    #     else:
    #         class_mean.append(None)
    #     class_dataNums[n] = cnt

    return class_mean, class_dataNums

#动量梯度下降(gradient descent with Momentum)
def MGD(pro_model, cur_model, m, a):
    b = 1 - a

    reduce_model = cur_model - pro_model   #reduce_w = w(t+1) - wt

    m_alpha = m * a
    m_beta = reduce_model * b
    res = m_alpha + m_beta                                  #m = a * m + b * reduce_w

    finish_model = pro_model - res              #w(t+1) = wt - m
    return finish_model, res

def model_subtraction(model1, model2):    #模型相减（model1 - model2）
    tmp = model2.clone()
    subtraction_state_dict = {}
    for name, param in model1.state_dict().items():
        subtraction_value = model1.state_dict()[name] - model2.state_dict()[name]
        subtraction_state_dict[name] = subtraction_value
    model2.load_state_dict(subtraction_state_dict, strict=False)     #将计算后的dict转化为model形式
    # model_same(model2, tmp)
    return model2

def model_constant_multiplication(model, constant):  #模型常数相乘（model * p）
    tmp = model
    constantMult_state_dict = {}
    for name, param in model.state_dict().items():
        subtraction_value = model.state_dict()[name] * constant
        constantMult_state_dict[name] = subtraction_value
    model.load_state_dict(constantMult_state_dict, strict=False)
    # model_same(model, tmp)
    return model

def model_addition(model1, model2):    #模型相加（model1 + model2）
    tmp = model2
    subtraction_state_dict = {}
    for name, param in model1.state_dict().items():
        subtraction_value = model1.state_dict()[name] + model2.state_dict()[name]
        subtraction_state_dict[name] = subtraction_value
    model2.load_state_dict(subtraction_state_dict, strict=False)     #将计算后的dict转化为model形式
    # model_same(model2, tmp)
    return model2

def model_constant_division(model, constant):
    tmp = model
    constantMult_state_dict = {}
    for name, param in model.state_dict().items():
        subtraction_value = model.state_dict()[name] / constant
        constantMult_state_dict[name] = subtraction_value
    model.load_state_dict(constantMult_state_dict, strict=False)
    # model_same(model, tmp)
    return model

def model_dot(model1, model2):
    # 获取两个模型的参数字典
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    # 初始化点乘结果
    dot_product = 0.0

    # 遍历模型的参数，并逐元素相乘并累加
    for name in state_dict1:
        if name in state_dict2:
            param1 = state_dict1[name]
            param2 = state_dict2[name]
            elementwise_product = param1 * param2
            dot_product += elementwise_product.sum()
    return dot_product.item()

def model_norm(model):
    # 获取模型的参数
    params = model.parameters()

    # 计算二范数
    norm = 0.0
    for param in params:
        norm += torch.sum(param ** 2)

    norm = torch.sqrt(norm)

    return norm.item()

def model_same(model1, model2):
    # 获取两个模型的状态字典
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    # 比较两个模型的状态字典是否一致
    are_equal = True
    for key1, key2 in zip(state_dict1.keys(), state_dict2.keys()):
        if state_dict1[key1].shape != state_dict2[key2].shape:
            are_equal = False
            break
        if not torch.equal(state_dict1[key1], state_dict2[key2]):
            are_equal = False
            break

    if are_equal:
        print("两个模型参数一致")
        return True
    else:
        print("两个模型参数不一致")
        return False

def completion_matrix(filename):
    fileload = '/data/yaominghao/code/FedRepo/result/'
    contribution_matrix = torch.FloatTensor(
        np.load(fileload + filename + '.npy', allow_pickle=True))

    final = []
    for i in range(len(contribution_matrix)):
        cur = contribution_matrix[0]
        # print(cur)

        # 使用PyTorch中的SVD函数进行矩阵分解
        U, S, V = torch.svd(cur)

        # 重构原始矩阵
        reconstructed_A = U @ torch.diag(S) @ V.t()

        print("round ", i)

        final.append(reconstructed_A.tolist())

    np.save(fileload + 'final/' + filename + "_final.npy", final)
    return final

def shapley(utility, local_idx, N):
    """
    :param utility: a dictionary with keys being tuples. (1,2,3) means that the trainset 1,2 and 3 are used,
    and the values are the accuracies from training on a combination of these trainsets
    :param N: total number of data contributors
    :returns the dictionary with the shapley values of the data, eg: {1: 0.2, 2: 0.4, 3: 0.4}
    """
    shapley_dict = {}
    for i in local_idx:
        shapley_dict[i] = 0
    for key in utility:
        if key != ():
            for contributor in key:
                # print('contributor:', contributor, key) # print check
                marginal_contribution = utility[key] - utility[tuple(i for i in key if i != contributor)]
                # print('marginal:', marginal_contribution) # print check
                shapley_dict[contributor] += marginal_contribution / ((comb(N-1, len(key)-1))*N)
    return shapley_dict


def powersettool(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def time_predict(start_time, total_rounds):
    curtime = time.time()
    time_diff = curtime - start_time
    total_cost = time_diff * total_rounds

    days, remainder = divmod(total_cost, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"++++++++++++++++++++++预计训练：{days}天, {hours}小时, {minutes}分钟, {seconds}秒+++++++++++++++++++++++++++++++++++++++++")

def time_diff(start_time, endtime):
    time_diff = endtime - start_time

    days, remainder = divmod(time_diff, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"++++++++++++++++++++++时间间隔：{days}天, {hours}小时, {minutes}分钟, {seconds}秒+++++++++++++++++++++++++++++++++++++++++")

    return time_diff

def get_dict_ID(dict):
    ID_list = []
    for id in dict.keys():
        ID_list.append(id)

    return ID_list


def get_dict_value(dict):
    value_list = []
    for value in dict.values():
        value_list.append(value)

    return value_list




