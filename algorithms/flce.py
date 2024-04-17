import copy
import numpy as np
import fmodule
import warnings

import torch
import torch.nn as nn
import time
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt

import sys
sys.path.append("..")

from utils import Averager
from utils import count_acc
from utils import append_to_logs
from utils import format_logs
from utils import preprototype
from utils import time_predict, time_diff

from tools import construct_dataloaders
from tools import construct_optimizer
from tools import client_similarity, client_similarity_alone, client_similarity_V, client_similarity_diff, client_similarity_prodiff, client_similarity_M, client_similarity_P
from tools import cal_client_full_contri, cal_prototypes_full_contri

from SupConLoss import SupConLoss
# from loss import SupConLoss2


class FLCE():
    def __init__(
        self, csets, gset, model, args
    ):
        self.csets = csets
        self.gset = gset
        self.model = model
        self.args = args

        self.clients = list(csets.keys())

        # construct dataloaders
        self.train_loaders, self.test_loaders, self.glo_test_loader = \
            construct_dataloaders(
                self.clients, self.csets, self.gset, self.args
            )

        self.client_models = {}
        for client in self.clients:
            self.client_models[client] = copy.deepcopy(
                model
            )

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
        start_time = time.time()
        print('#################开始训练时间为{}##################'.format(datetime.fromtimestamp(start_time)))

        clients_weight = []
        for i in range(len(self.clients)):
            clients_weight.append(len(self.csets[i][0]))

        clients_weight = [i/sum(clients_weight) for i in clients_weight]
        print(clients_weight)

        # Training
        # global ALLglobal_local_accs
        ALLContribution_Matrix = []

        clients_prototypes_weight = [[] for i in range(self.args.n_clients)]
        pre_clients_prototype_list = [[] for i in range(self.args.n_clients)]  # 只有第一轮在客户端计算，其他轮次在服务端计算

        for r in range(1, self.args.max_round + 1):
            pre_global_prototype = []

            if r == 2:
                time_predict(start_time, self.args.max_round)

            n_sam_clients = int(self.args.c_ratio * len(self.clients))     #随机选择客户端

            # if r == 1 or r == 2 or r == 3:
            #     n_sam_clients = len(self.clients)                 #设置第一轮全部选中
            # else:
            #     n_sam_clients = int(self.args.c_ratio * len(self.clients))

            sam_clients = np.random.choice(
                self.clients, n_sam_clients, replace=False
            )

            if r == 1:                                       #设置分多轮全部选中
                sam_clients = [0,1,2,3,4,5,6,7,8,9]
            elif r == 2:
                sam_clients = [10,11,12,13,14,15,16,17,18,19]
            elif r == 3:
                sam_clients = [20,21,22,23,24,25,26,27,28,29]
            elif r == 4:
                sam_clients = [30,31,32,33,34,35,36,37,38,39]
            elif r == 5:
                sam_clients = [40,41,42,43,44,45,46,47,48,49]
            else:
                sam_clients = np.random.choice(
                    self.clients, n_sam_clients, replace=False
                )


            # if r == 1:                                       #设置分多轮全部选中
            #     sam_clients = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
            # elif r == 2:
            #     sam_clients = [20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
            # elif r == 3:
            #     sam_clients = [40,41,42,43,44,45,46,47,48,49,0,1,2,3,4,5,6,7,8,9]
            # else:
            #     sam_clients = np.random.choice(
            #         self.clients, n_sam_clients, replace=False
            #     )

            # if r == 1:                                       #设置分多轮全部选中
            #     sam_clients = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
            # elif r == 2:
            #     sam_clients = [30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,0,1,2,3,4,5,6,7,8,9]
            # else:
            #     sam_clients = np.random.choice(
            #         self.clients, n_sam_clients, replace=False
            #     )


            # if r == 1:                                       #设置分多轮全部选中
            #     sam_clients = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
            # elif r == 2:
            #     sam_clients = [40,41,42,43,44,45,46,47,48,49,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
            # else:
            #     sam_clients = np.random.choice(
            #         self.clients, n_sam_clients, replace=False
            #     )

            print("clients列表", sam_clients)

            local_models = {}

            avg_loss = Averager()
            all_per_accs = []

            clients_features = [[] for i in range(self.args.n_clients)]
            clients_class_num = [[] for i in range(self.args.n_clients)]

            # ALLglobal_local_accs = []

            clients_start_time = time.time()    #客户端训练开始

            for client in sam_clients:
                # to cuda
                if self.args.cuda is True:
                    self.client_models[client].cuda()

            for client in sam_clients:
                if r == 1:     #只有第一轮用初始化的全局模型计算原型
                    #用分发的全局模型收集客户端原型(用之前的原型)
                    pre_client_prototype, preclient_classnum = preprototype(self, copy.deepcopy(self.model), self.train_loaders[client], self.test_loaders[client])
                    pre_clients_prototype_list[client] = pre_client_prototype

                local_model, per_accs, loss, class_mean, class_num = self.update_local(    #下载全局模型通过本地数据集训练
                    r=r,
                    model=copy.deepcopy(self.model),
                    local_model=copy.deepcopy(self.client_models[client]),
                    train_loader=self.train_loaders[client],
                    test_loader=self.test_loaders[client],
                )

                local_models[client] = copy.deepcopy(local_model)
                avg_loss.add(loss)
                all_per_accs.append(per_accs)

                #将每个client的特征和数量记录到全局
                clients_features[client] = class_mean
                clients_class_num[client] = class_num


            clients_end_time = time.time()       #客户端训练结束
            clients_per_time = clients_end_time - clients_start_time
            client_total_time += clients_per_time

            time_diff(clients_start_time, clients_end_time)

            #计算每个client与全局prototype的相似度(贡献)
            all_similarities = client_similarity_P(pre_clients_prototype_list, clients_features, clients_class_num, self.args.n_classes, self.args.n_clients, r)

            pre_clients_prototype_list = cal_prototypes_full_contri(all_similarities, clients_features, self.args.n_classes, self.args.n_clients)

            client_proportion = cal_client_full_contri(all_similarities, self.args.n_classes, self.args.n_clients)

            train_loss = avg_loss.item()
            per_accs = list(np.array(all_per_accs).mean(axis=0))


            print("round", r)
            print("client_proportion", client_proportion)
            # print("similarities", all_similarities)
            # print("acc", per_accs)

            # ALLglobal_local_accs.append(per_accs)
            ALLContribution_Matrix.append(all_similarities)


            self.update_global(
                r=r,
                global_model=self.model,
                local_models=local_models,
                p=client_proportion
            )

            end_time = time.time()     #服务端结束时间

            cur_glo_test_acc = self.test(
                model=self.model,
                loader=self.glo_test_loader,
            )
            print('第{}轮的globe acc为{}'.format(r, cur_glo_test_acc))

            if r % self.args.test_round == 0:
                # global test loader
                glo_test_acc, glo_test_precision, glo_test_recall, glo_test_f1= self.test(
                    model=self.model,
                    loader=self.glo_test_loader,
                )

                print('F1 Score: ', glo_test_f1)

                # add to log
                self.logs["ROUNDS"].append(r)
                self.logs["LOSSES"].append(train_loss)
                self.logs["GLO_TACCS"].append(glo_test_acc)
                self.logs["GLO_precision"].append(glo_test_precision)
                self.logs["GLO_recall"].append(glo_test_recall)
                self.logs["GLO_f1"].append(glo_test_f1)
                # self.logs["GLO_auc"].append(glo_test_auc)

                # self.logs["LOCAL_TACCS"].extend(per_accs)


                np.save("/data/yaominghao/logs_2024/logs_202401/Contribution Martix/contribution_matrix_" + 'FLCE_' + self.args.filename + '_Allmydata' + '.npy', ALLContribution_Matrix)      #cifar10
                # np.save("/data/yaominghao/code/FedRepo/result/contribution_matrix_fedproavg_CE_CL+M_" + 'Allmydata' + '.npy', ALLContribution_Matrix)
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


    def update_local(self, r, model, local_model, train_loader, test_loader):
        # glo_model = copy.deepcopy(model)
        # glo_model.eval()
        # local_model.eval()

        classList = [[] for _ in range(self.args.n_classes)]   #记录当前本地客户端中每个类中预测成功的特征
        class_dataNums = [0 for _ in range(self.args.n_classes)]    #统计当前本地客户端每个类样本数量

        # lr = min(r / 10.0, 1.0) * self.args.lr
        lr = self.args.lr

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

        #模型训练模式
        for t in range(n_total_bs + 1):
            if t in [0, n_total_bs +10]:    #看下per_acc最后一轮
                # per_acc = self.test(
                #     model=model,
                #     loader=test_loader,
                # )
                per_acc = 0
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

            total = sum([param.nelement() for param in model.parameters()])
            print('Number of parameter', total)

            # hs1, _ = glo_model(batch_x)
            # hs0, _ = local_model(batch_x)

            # moon loss
            # ct_loss = self.contrastive_loss(
            #     hs, hs0.detach(), hs1.detach()
            # )

            CL_criterion = SupConLoss()
            # CL_criterion = SupConLoss2(device=torch.device('cuda:1'))


            CL_loss_tmp = CL_criterion(hs, labels=batch_y)

            # CL_loss_tmp = CL_criterion(hs.view(hs.shape[0], hs.shape[1], -1), labels=None)
            if torch.isnan(CL_loss_tmp):            #nan值去掉
                CL_loss = 0
            else:
                CL_loss = CL_loss_tmp

            _, prediction = torch.max(logits, 1)

            # 每个类预测成功的特征
            # for m in range(batch_x.shape[0]):
            #     pred = prediction[m]
            #     label = batch_y[m]
            #     # 如果预测成功就把特征数据hs记录到classList
            #     if pred == label:
            #         classList[label].append(hs[m])

            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, batch_y)

            alpha = 1
            # total_loss = loss                            #CE_loss
            total_loss = loss + alpha * CL_loss        #CE+CL_loss

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), self.args.max_grad_norm
            )
            optimizer.step()

            avg_loss.add(total_loss.item())

        model.eval()
        #模型评估模式
        for t in range(n_total_bs + 1):       #NonIID数据量超过
            if t in [0, n_total_bs+10]:
                # per_acc = self.test(
                #     model=model,
                #     loader=test_loader,
                # )
                per_acc=0
                per_accs.append(per_acc)

            if t >= n_total_bs:
                break

            # model.eval()
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

            # nn.utils.clip_grad_norm_(
            #     model.parameters(), self.args.max_grad_norm
            # )

        class_mean = []
        # 进行特征平均值计算
        # meanFeature, featureNumofClasses = getMeanFeature(self.args.n_classes)
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

        for label in range(self.args.n_classes):
            if len(classList[label]) != 0:
                end_matrix = torch.div(classList[label], class_dataNums[label])
                class_mean.append(end_matrix)
            else:
                class_mean.append(None)

        total_loss = avg_loss.item()

        print('fedproavg loss', total_loss)
        return model, per_accs, total_loss, class_mean, class_dataNums

    def update_global(self, r, global_model, local_models, p):
        # self.model = self.aggregate(local_models, p)
        mean_state_dict = {}
        for name, param in global_model.state_dict().items():
            vs = []
            for client in local_models.keys():
                # vs.append(local_models[client].state_dict()[name])
                vs.append(local_models[client].state_dict()[name] * p[client] * len(local_models))   #按比例权重分配时应对客户端数量相乘方便后续取平均
            vs = torch.stack(vs, dim=0)

            try:
                mean_value = vs.mean(dim=0)
            except Exception:
                # for BN's cnt
                mean_value = (1.0 * vs).mean(dim=0).long()
            mean_state_dict[name] = mean_value

        global_model.load_state_dict(mean_state_dict, strict=False)

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
                y_test_one_hot = label_binarize(y_test, classes=np.unique(y_test))
                # y_test_one_hot = label_binarize(y_test, classes=class_100)
                # print('y_test', np.unique(y_test))
                # print(y_test_one_hot.shape[1])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    precision = precision_score(y_test, y_pred, average='macro')

                recall = recall_score(y_test, y_pred, average='macro')
                f1 = f1_score(y_test, y_pred, average='macro')

                roc_auc = dict()

                # 计算每个类别的 ROC 曲线和 AUC
                # for j in range(self.args.n_classes):
                #     # print('j', j)
                #     y_tmp = y_test_one_hot[:, j]
                #     # print(y_tmp)
                #     y_score_tmp = y_scores[:, j]
                #     # print(y_score_tmp)
                #     auc_score = roc_auc_score(y_tmp, y_score_tmp)
                #     roc_auc[j] = auc_score

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

    # def contrastive_loss(self, hs, hs0, hs1):
    #     cs = nn.CosineSimilarity(dim=-1)
    #     sims0 = cs(hs, hs0)
    #     sims1 = cs(hs, hs1)
    #
    #     sims = 2.0 * torch.stack([sims0, sims1], dim=1)
    #     labels = torch.LongTensor([1] * hs.shape[0])
    #     labels = labels.to(hs.device)
    #
    #     criterion = nn.CrossEntropyLoss()
    #     ct_loss = criterion(sims, labels)
    #     return ct_loss


