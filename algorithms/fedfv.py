import copy
import numpy as np
import math
import fmodule

import torch
import torch.nn as nn

from utils import Averager
from utils import count_acc
from utils import append_to_logs
from utils import format_logs, time_diff
from utils import model_addition, model_subtraction, model_constant_multiplication, model_dot, model_norm,model_constant_division

from tools import construct_dataloaders
from tools import construct_optimizer
from tools import cal_client_contribution

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from utils import time_diff
from datetime import datetime
import warnings
import time



class FedFv():
    def __init__(
        self, csets, gset, model, args
    ):
        self.csets = csets
        self.gset = gset
        self.model = model
        self.args = args
        self.client_last_sample_round = [-1 for i in range(self.args.n_clients)]
        self.client_grads_history = [0 for i in range(self.args.n_clients)]

        self.clients = list(csets.keys())

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

        ALLContribution_Matrix = []
        # Training
        for r in range(1, self.args.max_round + 1):
            n_sam_clients = int(self.args.c_ratio * len(self.clients))
            sam_clients = np.random.choice(
                self.clients, n_sam_clients, replace=False
            )

            local_models = {}

            avg_loss = Averager()
            all_per_accs = []
            client_losses = []    #compute per_loss
            client_models = []

            clients_start_time = time.time()
            for client in sam_clients:
                local_model, per_accs, cur_client_losses = self.update_local(
                    r=r,
                    model=copy.deepcopy(self.model),
                    train_loader=self.train_loaders[client],
                    test_loader=self.test_loaders[client],
                )
                client_losses.append(cur_client_losses)
                client_models.append(copy.deepcopy(local_model))

                local_models[client] = copy.deepcopy(local_model)
                local_models_dict = {k: v for k, v in zip(sam_clients, client_models)}

                loss = sum(client_losses)/len(sam_clients)    #compute avg_loss
                avg_loss.add(loss)
                all_per_accs.append(per_accs)

            #计算贡献值
            Allclient_contribution = cal_client_contribution(self.model, self.args.n_clients, local_models)
            ALLContribution_Matrix.append(Allclient_contribution)

            clients_end_time = time.time()
            clients_per_time = clients_end_time - clients_start_time
            client_total_time += clients_per_time

            time_diff(clients_start_time, clients_end_time)

            end_time = time.time()
            All_per_time = end_time - clients_start_time
            server_per_time = All_per_time - clients_per_time
            server_total_time += server_per_time

            train_loss = avg_loss.item()
            per_accs = list(np.array(all_per_accs).mean(axis=0))

            print("rounds:  ", r)
            print("Acc", per_accs)

            self.update_global(
                r=r,
                global_model=self.model,
                local_models=local_models_dict,
                losses = client_losses,
                alpha = 0.1,
                tau = 1
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

                # np.save("/data/yaominghao/code/FedRepo/result/contribution_matrix_Allmydatarefedfv_" + self.args.filename + '.npy', ALLContribution_Matrix)
                np.save(
                    "/data/yaominghao/logs_2024/log202403/Contribution Martix/contribution_matrix_" + 'FedFV_' + self.args.filename + '_mydata' + '.npy',
                    ALLContribution_Matrix)
                print("contribution_matrix save success!")

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

    def update_global(self, r, global_model, local_models, losses, alpha, tau):
        ws = [[] for _ in range(len(local_models))]
        for i in range(len(local_models)):
            idx, model = list(local_models.items())[i]
            ws[i] = model

        grads = [self.model - w for w in ws]          #grads = global_model - local_model

        # update GH
        for cid, gi in zip(local_models, grads):
            self.client_grads_history[cid] = gi
            self.client_last_sample_round[cid] = r

        # project grads
        order_grads = copy.deepcopy(grads)
        order = [_ for _ in range(len(order_grads))]

        # sort client gradients according to their losses in ascending orders
        tmp = sorted(list(zip(losses, order)), key=lambda x: x[0])
        order = [x[1] for x in tmp]

        # keep the original direction for clients with the αm largest losses
        keep_original = []
        if alpha > 0:
            keep_original = order[math.ceil((len(order) - 1) * (1 - alpha)):]

        # mitigate internal conflicts by iteratively projecting gradients
        for i in range(len(order_grads)):
            if i in keep_original: continue
            for j in order:
                if (j == i):
                    continue
                else:
                    # calculate the dot of gi and gj
                    dot = grads[j].dot(order_grads[i])
                    if dot < 0:
                        order_grads[i] = order_grads[i] - grads[j] * dot / (grads[j].norm()**2)

        # aggregate projected grads
        gt = fmodule._model_average(order_grads)

        # mean_state_dict = {}
        #
        # for name, param in global_model.state_dict().items():
        #     vs = []
        #     for client in range(len(order_grads)):
        #         vs.append(order_grads[client].state_dict()[name])
        #     vs = torch.stack(vs, dim=0)
        #
        #     try:
        #         mean_value = vs.mean(dim=0)
        #     except Exception:
        #         # for BN's cnt
        #         mean_value = (1.0 * vs).mean(dim=0).long()
        #     mean_state_dict[name] = mean_value
        #
        # global_model.load_state_dict(mean_state_dict, strict=False)
        # gt = copy.deepcopy(global_model)

        # gt = fmodule._model_average(order_grads)

        # mitigate external conflicts
        if r >= tau:
            for k in range(tau-1, -1, -1):
                # calculate outside conflicts
                gcs = [self.client_grads_history[cid] for cid in range(self.args.n_clients) if self.client_last_sample_round[cid] == r - k and gt.dot(self.client_grads_history[cid]) < 0]
                if gcs:
                    g_con = fmodule._model_sum(gcs)
                    dot = gt.dot(g_con)
                    if dot < 0:
                        gt = gt - g_con*dot/(g_con.norm()**2)

        # ||gt||=||1/m*Σgi||
        # for name, param in global_model.state_dict().items():
        #     vs = []
        #     for client in range(len(grads)):
        #         vs.append(grads[client].state_dict()[name])
        #     vs = torch.stack(vs, dim=0)
        #
        #     try:
        #         mean_value = vs.mean(dim=0)
        #     except Exception:
        #         # for BN's cnt
        #         mean_value = (1.0 * vs).mean(dim=0).long()
        #     mean_state_dict[name] = mean_value
        #
        # global_model.load_state_dict(mean_state_dict, strict=False)
        gnorm = fmodule._model_average(grads).norm()
        # gnorm = fmodule._model_average(grads).norm()
        gt = gt/gt.norm()*gnorm

        self.model = self.model - gt



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
