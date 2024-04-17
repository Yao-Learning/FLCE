import copy
import numpy as np
import math
import fmodule

import torch
import torch.nn as nn
from torch.linalg import norm
import torch.nn.functional as F

from utils import Averager
from utils import count_acc
from utils import append_to_logs
from utils import format_logs

from tools import construct_dataloaders
from tools import construct_optimizer
from tools import flatten, unflatten

from tools import cal_client_contribution
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from utils import time_diff
from datetime import datetime
import warnings
import time

from fmodule import _modeldict_zeroslike


class CGSV():
    def __init__(
        self, csets, gset, model, args
    ):
        self.csets = csets
        self.gset = gset
        self.model = model
        self.args = args

        self.alpha = 0.5
        self.beta = 0.5
        self.Gamma = 0.5

        self.clients = list(csets.keys())

        # construct dataloaders
        self.train_loaders, self.test_loaders, self.glo_test_loader = \
            construct_dataloaders(
                self.clients, self.csets, self.gset, self.args
            )

        self.history_local_model = []
        self.history_agg_update = self.model
        self.rs = []

        self.logs = {
            "ROUNDS": [],
            "LOSSES": [],
            "GLO_TACCS": [],
            # "LOCAL_TACCS": [],
            "GLO_precision": [],
            "GLO_recall": [],
            "GLO_f1": [],
            # "GLO_auc": []
        }

    def train(self):
        client_total_time = 0
        server_total_time = 0
        start_time = time.time()       #客户端开始时间
        print('#################开始训练时间为{}##################'.format(datetime.fromtimestamp(start_time)))

        ALLContribution_Matrix = []

        #所有客户端模型初始化
        for i in range(len(self.clients)):
            self.history_local_model.append(self.model)

        #计算初始权重
        clients_weight = []
        for i in range(len(self.clients)):
            clients_weight.append(len(self.csets[i][0]))
        clients_weight = [i/sum(clients_weight) for i in clients_weight]


        print(clients_weight)

        # Training
        for r in range(1, self.args.max_round + 1):
            n_sam_clients = int(self.args.c_ratio * len(self.clients))
            sam_clients = np.random.choice(
                self.clients, n_sam_clients, replace=False
            )

            local_updates = {}
            local_models = {}

            avg_loss = Averager()
            all_per_accs = []

            clients_start_time = time.time()
            for client in sam_clients:
                #获取每个客户端的更新
                local_model, local_update, per_accs, loss = self.update_local(
                    r=r,
                    model=copy.deepcopy(self.model),
                    train_loader=self.train_loaders[client],
                    test_loader=self.test_loaders[client],
                )

                local_updates[client] = copy.deepcopy(local_update)
                local_models[client] = copy.deepcopy(local_model)

                avg_loss.add(loss)
                all_per_accs.append(per_accs)

            train_loss = avg_loss.item()
            per_accs = list(np.array(all_per_accs).mean(axis=0))

            #计算贡献值
            Allclient_contribution = cal_client_contribution(self.model, self.args.n_clients, local_models)
            ALLContribution_Matrix.append(Allclient_contribution)

            clients_end_time = time.time()    #客户端结束时间
            clients_per_time = clients_end_time - clients_start_time
            client_total_time += clients_per_time

            time_diff(clients_start_time, clients_end_time)

            agg_gradient, rs, q_ratios = self.update_global(
                r=r,
                global_model=self.model,
                local_updates=local_updates,
                local_models=local_models,
                clients_weight=clients_weight,
            )

            local_masked_model = self.update_local_mask(
                aggregated_gradient=agg_gradient,
                q_ratios=q_ratios,
                local_models=local_models,
            )

            #更新各个客户端模型
            for i in range(len(self.clients)):
                if i in local_masked_model:
                    self.history_local_model[i] = local_masked_model[i]

            end_time = time.time()      #服务端结束时间

            if r % self.args.test_round == 0:
                # global test loader
                glo_test_acc, glo_test_precision, glo_test_recall, glo_test_f1 = self.test(
                    model=self.model,
                    loader=self.glo_test_loader,
                )

                # add to log
                self.logs["ROUNDS"].append(r)
                self.logs["LOSSES"].append(train_loss)
                self.logs["GLO_TACCS"].append(glo_test_acc)
                self.logs["GLO_precision"].append(glo_test_precision)
                self.logs["GLO_recall"].append(glo_test_recall)
                self.logs["GLO_f1"].append(glo_test_f1)

                # np.save("/data/yaominghao/code/FedRepo/result/contribution_matrix_Allmydatarefedfv_" + self.args.filename + '.npy', ALLContribution_Matrix)
                np.save(
                    "/data/yaominghao/logs_2024/log202403/Contribution Martix/contribution_matrix_" + 'CGSV_' + self.args.filename + '_mydata' + '.npy',
                    ALLContribution_Matrix)
                print("contribution_matrix save success!")

                All_per_time = end_time - clients_start_time
                server_per_time = All_per_time - clients_per_time
                server_total_time += server_per_time


                print('#################共训练了{}s#################'.format(end_time - start_time))

                print('#################本轮客户端共训练了{}s#################'.format(clients_per_time))
                print('#################本轮服务端共训练了{}s#################'.format(server_per_time))

                print('#################客户端共训练了{}s#################'.format(client_total_time))
                print('#################服务端共训练了{}s#################'.format(server_total_time))

                print("[R:{}] [Ls:{}] [TeAc:{}] [PAcBeg:{} PAcAft:{}] [pre: {}] [recall: {}] [f1_score: {}]".format(
                    r, train_loss, glo_test_acc, per_accs[0], per_accs[-1], glo_test_precision, glo_test_recall, glo_test_f1
                ))

    def update_local(self, r, model, train_loader, test_loader):
        # lr = min(r / 10.0, 1.0) * self.args.lr
        lr = self.args.lr

        per_local_model = copy.deepcopy(model)       #获取训练前的本地原始模型

        optimizer = construct_optimizer(
            model, lr, self.args
        )

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

        model.train()

        loader_iter = iter(train_loader)

        avg_loss = Averager()
        per_accs = []

        for t in range(n_total_bs + 1):
            if t in [0, n_total_bs]:
                per_acc, precision, recall, f1 = self.test(
                    model=model,
                    loader=test_loader,
                )
                per_accs.append(per_acc)

            if t >= n_total_bs:
                break

            model.train()
            try:
                batch_x, batch_y = next(loader_iter)
            except Exception:
                loader_iter = iter(train_loader)
                batch_x, batch_y = next(loader_iter)

            if self.args.cuda:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            hs, logits = model(batch_x)

            criterion = nn.CrossEntropyLoss()

            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), self.args.max_grad_norm
            )
            optimizer.step()

            avg_loss.add(loss.item())

        loss = avg_loss.item()
        print('CGSV loss', loss)

        cur_local_model = copy.deepcopy(model)    #获取训练后的模型
        local_update = cur_local_model - per_local_model    #获取更新

        gradient = copy.deepcopy(local_update)

        #归一化
        # local_update_flatten = flatten(local_update)
        # norm_local_update_flatten = norm(local_update_flatten) + 1e-7
        #
        # gradient_flatten = torch.multiply(torch.tensor(self.Gamma), torch.div(local_update_flatten, norm_local_update_flatten))
        #
        # param_state_dict = {}
        # for name, param in gradient.state_dict().items():
        #     n_params = len(param.view(-1))
        #     value = torch.as_tensor(gradient_flatten[:n_params]).reshape(param.size())
        #     param_state_dict[name] = value
        #     gradient_flatten = gradient_flatten[n_params:]
        #
        # gradient.load_state_dict(param_state_dict, strict=False)

        return cur_local_model, gradient, per_accs, loss

    def update_global(self, r, global_model, local_updates, local_models, clients_weight):
        cur_local_updates = [[] for _ in range(len(local_updates))]
        cur_clients_weight = [0 for _ in range(len(local_updates))]

        cur_global_model = copy.deepcopy(global_model)

        rs_dict, qs_dict = [], []
        rs = torch.zeros(len(self.clients))
        past_phis = []
        weights = []

        #第一轮时以数据量聚合
        if r == 1:
            weights = clients_weight
            for i in range(len(local_updates)):
                idx, model = list(local_updates.items())[i]
                cur_local_updates[i] = model
                cur_clients_weight[i] = weights[idx]

            cur_clients_weight = [i / sum(cur_clients_weight) for i in cur_clients_weight]
            print(cur_clients_weight)
            aggregated_update = self.aggregate(cur_local_updates, cur_clients_weight)
            self.model = cur_global_model + aggregated_update
        else:
            aggregated_update = self.history_agg_update


        flatten_agg_update = flatten(aggregated_update)

        cosine_distance = []
        for client in range(len(self.clients)):
            if client in local_models:
                flatten_local_update = flatten(local_updates[client])
                tmp = F.cosine_similarity(flatten_local_update, flatten_agg_update, 0, 1e-10)
                cosine_distance.append(tmp)
            else:
                cosine_distance.append(0)

        rs = torch.tensor(cosine_distance)
        # rs = self.alpha * rs + (1-self.alpha) * torch.tensor(cosine_distance)

        # rs = torch.clamp(rs, min=1e-3)  # make sure the rs do not go negative
        rs = torch.div(rs, rs.sum())  # normalize the weights to 1

        q_ratios = torch.tanh(self.beta * rs)
        q_ratios = torch.div(q_ratios, torch.max(q_ratios))

        qs_dict.append(q_ratios)
        rs_dict.append(rs)

        weights = rs.tolist()
        # weights = clients_weight
        for i in range(len(local_updates)):
            idx, model = list(local_updates.items())[i]
            cur_local_updates[i] = model
            cur_clients_weight[i] = weights[idx]

        cur_clients_weight = [i / sum(cur_clients_weight) for i in cur_clients_weight]
        print(cur_clients_weight)

        #使用余弦作为比例再聚合一次作为下一轮的基准
        aggregated_update = self.aggregate(cur_local_updates, cur_clients_weight)
        self.history_agg_update = aggregated_update

        self.model = cur_global_model + aggregated_update

        return aggregated_update, rs, q_ratios

    def update_local_mask(self, aggregated_gradient, q_ratios, local_models):
        q_ratios = [i - min(q_ratios) for i in q_ratios]
        reward_gradient_dict = {}

        for i in range(len(self.clients)):
            if i in local_models:
                reward_gradient_dict[i] = self.mask_grad_update_by_order(aggregated_gradient, mask_percentile=q_ratios[i], mode='layer')

        for j in range(len(self.clients)):
            if j in local_models:
                local_models[j] = local_models[j] + reward_gradient_dict[j]

        return local_models


    # def test(self, model, loader):
    #     model.eval()
    #
    #     acc_avg = Averager()
    #
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y) in enumerate(loader):
    #             if self.args.cuda:
    #                 batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
    #             _, logits = model(batch_x)
    #             acc = count_acc(logits, batch_y)
    #             acc_avg.add(acc)
    #
    #     acc = acc_avg.item()
    #     return acc

    def test(self, model, loader):
        model.eval()

        acc_avg = Averager()
        precision_avg = Averager()
        recall_avg = Averager()
        f1_avg = Averager()
        auc_avg = Averager()

        roc_auc = dict()

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(loader):
                if self.args.cuda:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                _, logits = model(batch_x)
                acc = count_acc(logits, batch_y)

                y_pred = torch.argmax(logits, dim=1).to('cpu')
                y_scores = torch.nn.functional.softmax(logits, dim=1).to('cpu')
                y_scores = y_scores.numpy()
                # print(y_scores.shape[1])
                y_test = batch_y.to('cpu')
                class_100 = np.arange(100)
                class_10 = np.arange(10)
                # y_test_one_hot = label_binarize(y_test, classes=np.unique(y_test))
                y_test_one_hot = label_binarize(y_test, classes=class_10)
                # print('y_test', np.unique(y_test))
                # print(y_test_one_hot.shape[1])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    precision = precision_score(y_test, y_pred, average='macro')

                recall = recall_score(y_test, y_pred, average='macro')
                f1 = f1_score(y_test, y_pred, average='macro')

                # roc_auc = dict()
                #
                # # 计算每个类别的 ROC 曲线和 AUC
                # for j in range(self.args.n_classes):
                #     # print('j', j)
                #     y_tmp = y_test_one_hot[:, j]
                #     # print(y_tmp)
                #     y_score_tmp = y_scores[:, j]
                #     # print(y_score_tmp)
                #     auc_score = roc_auc_score(y_tmp, y_score_tmp)
                #     roc_auc[j] = auc_score
                #
                # mean_auc = np.mean(list(roc_auc.values()))

                acc_avg.add(acc)
                precision_avg.add(precision)
                recall_avg.add(recall)
                f1_avg.add(f1)
                # auc_avg.add(mean_auc)


        acc = acc_avg.item()
        precision = precision_avg.item()
        recall = recall_avg.item()
        f1 = f1_avg.item()
        # auc = auc_avg.item()


        # for i, auc_score in enumerate(roc_auc.values()):
        #     print(f'AUC for class {i}: {auc_score}')



        return acc, precision, recall, f1
        # return acc


    def save_logs(self, fpath):
        all_logs_str = []
        all_logs_str.append(str(self.args))

        logs_str = format_logs(self.logs)
        all_logs_str.extend(logs_str)

        append_to_logs(fpath, all_logs_str)

    def aggregate(self, models, p):
        return fmodule._model_average(models, p)

    def mask_grad_update_by_order(self, grad_update, mask_order=None, mask_percentile=None, mode='all'):
        if mode == 'all':
            # mask all but the largest <mask_order> updates (by magnitude) to zero
            all_update_mod = torch.cat([update.data.view(-1).abs()
                                        for update in grad_update])
            if not mask_order and mask_percentile is not None:
                mask_order = int(len(all_update_mod) * mask_percentile)

            if mask_order == 0:
                return self.mask_grad_update_by_magnitude(grad_update, float('inf'))
            else:
                topk, indices = torch.topk(all_update_mod, mask_order)
                return self.mask_grad_update_by_magnitude(grad_update, topk[-1])

        elif mode == 'layer':  # layer wise largest-values criterion
            grad_update = copy.deepcopy(grad_update)

            mask_percentile = max(0, mask_percentile)

            param_state_dict = {}
            for name, param in grad_update.state_dict().items():
                layer_mod = param.view(-1).abs()
                if mask_percentile is not None:
                    mask_order = math.ceil(len(layer_mod) * mask_percentile)

                if mask_order == 0:
                    param_state_dict[name] = torch.zeros(param.shape)
                else:
                    topk, indices = torch.topk(layer_mod, min(mask_order, len(layer_mod) - 1))
                    if len(topk) < 1:
                        param_state_dict[name] = param
                    else:
                        param[param.abs() < topk[-1]] = 0
                        param_state_dict[name] = param
                    # for i in param.shape:
                    #     if param_state_dict[name][i] < topk[-1]:
                    #         param_state_dict[name][i] = 0
                    # param_state_dict[name][param.abs() < topk[-1]] = 0

            grad_update.load_state_dict(param_state_dict, strict=False)
            return grad_update

    def mask_grad_update_by_magnitude(self, grad_update, mask_constant):
        # mask all but the updates with larger magnitude than <mask_constant> to zero
        # print('Masking all gradient updates with magnitude smaller than ', mask_constant)
        grad_update = copy.deepcopy(grad_update)
        for i, update in enumerate(grad_update):
            grad_update[i].data[update.data.abs() < mask_constant] = 0
        return grad_update

