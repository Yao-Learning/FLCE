import copy
import numpy as np
import math
import fmodule
import time
from datetime import datetime

import torch
import torch.nn as nn

from utils import Averager
from utils import count_acc
from utils import append_to_logs
from utils import format_logs
from utils import time_predict, time_diff


from tools import construct_dataloaders
from tools import construct_optimizer

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import warnings


class FedFa():
    def __init__(
        self, csets, gset, model, args
    ):
        self.csets = csets
        self.gset = gset
        self.model = model
        self.args = args
        self.ms = self.model.zeros_like()                      
        self.mc = [self.model.zeros_like() for _ in range(self.args.n_clients)]       
        self.alpha = 0.5
        self.clients_gamma = 0.5           
        self.server_gamma = 0.9            
        self.b_round = 3                

        self.clients = list(csets.keys())

        self.history_weight = []

        # construct dataloaders
        self.train_loaders, self.test_loaders, self.glo_test_loader = \
            construct_dataloaders(
                self.clients, self.csets, self.gset, self.args
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

        all_partNum_list = [0 for _ in range(self.args.n_clients)]  
        # Training
        for r in range(1, self.args.max_round + 1):
            if r == 2:
                time_predict(start_time, self.args.max_round)

            print("rounds: ", r)
            n_sam_clients = int(self.args.c_ratio * len(self.clients))
            sam_clients = np.random.choice(
                self.clients, n_sam_clients, replace=False
            )

            print("select clients", sam_clients)

            local_models = {}

            avg_loss = Averager()
            all_per_accs = []
            cur_per_accs_list = [0 for _ in range(self.args.n_clients)]     
            partNum_list = [0 for _ in range(self.args.n_clients)]     

            for i in range(len(sam_clients)):
                all_partNum_list[sam_clients[i]] += 1     
                partNum_list[sam_clients[i]] = all_partNum_list[sam_clients[i]]    

            clients_start_time = time.time()
            for client in sam_clients:
                pro_model = copy.deepcopy(self.model)      
                local_model, per_accs, loss = self.update_local(
                    r=r,
                    model=copy.deepcopy(self.model),
                    train_loader=self.train_loaders[client],
                    test_loader=self.test_loaders[client],
                )

                dw = local_model - self.model
                # calculate m = γm+(1-γ)dw
                self.mc[client] = self.clients_gamma * self.mc[client] + (1 - self.clients_gamma) * dw
                local_model = local_model - self.mc[client] * self.args.lr

                # print("clients success!")
                # local_models[client] = copy.deepcopy(MGD_model)   
                local_models[client] = copy.deepcopy(local_model)
                cur_per_accs_list[client] = per_accs[1]     

                avg_loss.add(loss)
                all_per_accs.append(per_accs)

            clients_end_time = time.time()
            clients_per_time = clients_end_time - clients_start_time
            client_total_time += clients_per_time

            time_diff(clients_start_time, clients_end_time)

            train_loss = avg_loss.item()
            per_accs = list(np.array(all_per_accs).mean(axis=0))

            # np.save("/data/yaominghao/code/FedRepo/result/fedfa" + self.args.filename + '.npy', self.history_weight)



            end_time = time.time()
            All_per_time = end_time - clients_start_time
            server_per_time = All_per_time - clients_per_time
            server_total_time += server_per_time

            self.update_global(
                r=r,
                global_model=self.model,
                local_models=local_models,
                ACCs=cur_per_accs_list,
                Fs=partNum_list
            )

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
                # self.logs["GLO_auc"].append(glo_test_auc)
                # self.logs["LOCAL_TACCS"].extend(per_accs)

                np.save(
                    "/data/yaominghao/logs_2024/log202403/Contribution Martix/contribution_matrix_" + 'FedFa_' + self.args.filename + '_mydata' + '.npy',
                    self.history_weight)

                end_time = time.time()
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
        return model, per_accs, loss

    def update_global(self, r, global_model, local_models, ACCs, Fs):
        beta = 1 - self.alpha

        # pro_model = copy.deepcopy(global_model)

        print("当前轮次客户端参与频次： ", Fs)

        ACC = [0 for _ in range(len(local_models))]
        F = [0 for _ in range(len(local_models))]
        client_models = [[] for _ in range(len(local_models))]
        for i in range(len(local_models)):
            idx, model = list(local_models.items())[i]
            client_models[i] = model
            ACC[i] = ACCs[idx]
            F[i] = Fs[idx]

        sum_acc = np.sum(ACC)
        sum_f = np.sum(F)
        ACCinf = [-np.log2(1.0*acc/sum_acc+0.000001) for acc in ACC]
        Finf = [-np.log2(1-1.0*f/sum_f+0.00001) for f in F]
        sum_acc = np.sum(ACCinf)
        sum_f = np.sum(Finf)
        ACCinf = [acc/sum_acc for acc in ACCinf]
        Finf = [f/sum_f for f in Finf]
        # calculate weight = αACCi_inf+βfi_inf
        p = [self.alpha * accinf + beta*finf for accinf, finf in zip(ACCinf, Finf)]

        idx = local_models.keys()
        client_weight = dict(zip(idx, p))
        all_weight = []
        for i in range(len(self.clients)):
            if i in local_models:
                all_weight.append(client_weight[i])
            else:
                all_weight.append(0)

        self.history_weight.append(all_weight)


        wnew = self.aggregate(client_models, p)

        if r % self.b_round != 0:
            self.model = wnew
        else:
            dw = wnew - self.model
            # calculate m = γm+(1-γ)dw
            self.ms = self.server_gamma * self.ms + (1 - self.server_gamma) * dw
            self.model = wnew - self.ms * self.args.lr

    def aggregate(self, models, p):
        return fmodule._model_average(models, p)


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
